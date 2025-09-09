"""
LangGraph 기반 Main Orchestrator
이미지 워크플로우에 따른 체계적인 Supervisor 구현
"""

import os
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from common.short_term.redis_memory import RedisLangGraphMemory
from common.milvus_manager import MilvusDBManager
from teacher.teacher import Teacher, TeacherState
from farmer.farmer import Farmer, RouterState
from langgraph.checkpoint.postgres import PostgresSaver

class MainState(TypedDict):
    """라우터 상태 정의 - LangGraph StateGraph 호환"""
    user_query: str
    user_id: str
    chat_id: str
    session_key: str
    
    # 메모리 데이터
    existing_questions: Annotated[List[Dict], "기존 문제 목록"]
    locked_service: Optional[str]
    short_term_data: Annotated[Dict[str, Any], "숏텀 메모리 데이터"]
    
    # MilvusDB 데이터
    milvus_data: Annotated[Dict[str, Any], "MilvusDB에서 주입된 데이터"]
    
    # 분류 결과
    is_relevant: bool
    classified_service: str
    service_consistent: bool
    
    # 실행 결과
    teacher_result: Optional[Dict]
    farmer_result: Optional[Dict]
    final_response: str


class MainOrchestrator:
    """메인 오케스트레이터 - LangGraph StateGraph 기반"""
    
    def __init__(self, user_id: str, chat_id: str):
        self.user_id = user_id
        self.chat_id = chat_id
        self.session_key = f"{user_id}_{chat_id}"
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # MilvusDB 관리자 초기화
        self.milvus_manager = MilvusDBManager()
        
        # MilvusDB 연결
        if not self.milvus_manager.connect():
            print("⚠️ MilvusDB 연결 실패 - 일부 기능이 제한될 수 있습니다.")
        
        # 메모리 시스템 초기화
        self.memory = RedisLangGraphMemory(
            user_id=user_id,
            service="supervisor",
            chat_id=chat_id
        )
        
        # 서비스 초기화
        self.teacher = Teacher(user_id, "teacher", chat_id)
        self.farmer = Farmer()
        
        # LangGraph StateGraph 생성
        self.graph = self._create_graph()
        
        print(f"✅ MainOrchestrator 초기화 완료 (session: {self.session_key})")
        print(f"🔗 MilvusDB 연결 상태: {'✅ 연결됨' if self.milvus_manager.is_connected else '❌ 연결 안됨'}")
    
    def load_recent_questions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 생성된 문제들을 불러오기 (added_count 사용)"""
        try:
            if not self.memory:
                print("⚠️ 메모리가 초기화되지 않았습니다.")
                return []
            
            # 숏텀 메모리에서 added_count 가져오기
            short_term_data = self.load_short_term_memory()
            teacher_data = short_term_data.get("teacher", {})
            added_count = teacher_data.get("added_count", 0)
            
            print(f"🔍 Teacher added_count: {added_count}")
            
            if added_count == 0:
                print("📝 추가된 문제가 없습니다.")
                return []
            
            # 실제로는 added_count만큼만 가져오기
            actual_limit = min(limit, added_count)
            print(f"🔍 Redis에서 최근 {actual_limit}개 문제 조회 중...")
            
            # Redis에서 questions:{session_key}:* 패턴으로 키 검색
            pattern = f"questions:{self.session_key}:*"
            keys = self.memory.redis.keys(pattern)
            print(f"🔍 조회된 키들: {keys}")
            
            if not keys:
                print("📝 저장된 문제가 없습니다.")
                return []
            
            # 최근 생성된 순서로 정렬 (키에 타임스탬프가 포함되어 있다면)
            keys.sort(reverse=True)
            
            # 각 문제의 상세 정보 가져오기 (added_count만큼만)
            questions = []
            for i, key in enumerate(keys[:actual_limit]):
                try:
                    print(f"🔍 문제 {i+1} 상세 정보 조회 중... (키: {key})")
                    question_data = self.memory.redis.hgetall(key)
                    print(f"🔍 문제 데이터: {question_data}")
                    
                    # Redis에서 가져온 데이터를 직접 사용 (이미 문자열)
                    question_text = question_data.get("question", "")
                    options_str = question_data.get("options", "[]")
                    answer = question_data.get("answer", "")
                    explanation = question_data.get("explanation", "")
                    subject = question_data.get("subject", "unknown")
                    
                    # options JSON 파싱
                    try:
                        options = json.loads(options_str) if options_str else []
                    except:
                        options = []
                    
                    questions.append({
                        "qid": key if isinstance(key, str) else key.decode("utf-8"),
                        "question": question_text,
                        "options": options if isinstance(options, list) else [],
                        "answer": answer,
                        "explanation": explanation,
                        "subject": subject,
                        "created_at": int(time.time()),  # 현재 시간으로 설정
                        "updated_at": int(time.time())
                    })
                except Exception as e:
                    print(f"⚠️ 문제 {key} 로드 중 오류: {e}")
                    continue
            
            print(f"📖 최근 {len(questions)}개 문제를 불러왔습니다. (added_count: {added_count})")
            return questions
            
        except Exception as e:
            print(f"❌ 최근 문제 조회 실패: {e}")
            return []
    
    def _create_graph(self) -> StateGraph:
        """LangGraph StateGraph 생성"""
        workflow = StateGraph(MainState)
        
        # 노드 추가
        workflow.add_node("load_memory", self.load_memory_data)
        workflow.add_node("inject_milvus_data", self.inject_milvus_data)
        workflow.add_node("classify_question_and_service", self.classify_question_and_service)
        workflow.add_node("validate_classification", self._refine_classification_if_needed)
        workflow.add_node("check_consistency", self.check_final_service_consistency)
        workflow.add_node("update_session", self.update_session_data)
        workflow.add_node("execute_teacher", self.teacher_app)
        workflow.add_node("execute_farmer", self.execute_farmer)
        workflow.add_node("merge_teacher_result", self.merge_teacher_result)
        workflow.add_node("save_memory", self.save_memory_data)
        workflow.add_node("finalize_response", self.finalize_response)
        workflow.add_node("handle_irrelevant", self.handle_irrelevant_question)
        workflow.add_node("handle_inconsistency", self.handle_service_inconsistency)
        
        # 시작점 설정
        workflow.set_entry_point("load_memory")
        
        # 엣지 추가
        workflow.add_edge("load_memory", "inject_milvus_data")
        workflow.add_edge("inject_milvus_data", "classify_question_and_service")
        workflow.add_conditional_edges(
            "classify_question_and_service",
            self._route_after_classification,
            {
                "irrelevant": "handle_irrelevant",
                "relevant": "validate_classification"
            }
        )
        workflow.add_edge("validate_classification", "check_consistency")
        workflow.add_conditional_edges(
            "check_consistency",
            self._route_after_consistency_check,
            {
                "inconsistent": "handle_inconsistency",
                "consistent": "update_session"
            }
        )
        workflow.add_conditional_edges(
            "update_session",
            self._route_to_service,
            {
                "teacher": "execute_teacher",
                "farmer": "execute_farmer"
            }
        )
        workflow.add_edge("execute_teacher", "merge_teacher_result")
        workflow.add_edge("merge_teacher_result", "save_memory")
        workflow.add_edge("execute_farmer", "save_memory")
        workflow.add_edge("save_memory", "finalize_response")
        workflow.add_edge("handle_irrelevant", "save_memory")
        workflow.add_edge("handle_inconsistency", "save_memory")
        workflow.add_edge("finalize_response", END)

        db_url = os.getenv("DATABASE_URL")
        checkpointer = PostgresSaver.from_conn_string(db_url) if db_url else MemorySaver()

        return workflow.compile(checkpointer=checkpointer)
    
    def visualize_graph(self, output_path: str = "supervisor_graph.png"):
        """그래프 시각화"""
        try:
            # 그래프 이미지 생성
            image = self.graph.get_graph().draw_mermaid_png()
            
            # 파일로 저장
            with open(output_path, "wb") as f:
                f.write(image)
            
            print(f"📊 그래프 시각화 저장: {output_path}")
        except Exception as e:
            print(f"❌ 그래프 시각화 실패: {e}")
    
    # 라우팅 함수들
    def _route_after_classification(self, state: MainState) -> str:
        """분류 후 라우팅"""
        return "irrelevant" if not state["is_relevant"] else "relevant"
    
    def _route_after_consistency_check(self, state: MainState) -> str:
        """일관성 검사 후 라우팅"""
        return "inconsistent" if not state["service_consistent"] else "consistent"
    
    def _route_to_service(self, state: MainState) -> str:
        """서비스로 라우팅"""
        return state["classified_service"]
    
    def load_memory_data(self, state: MainState) -> MainState:
        """1. 메모리 데이터 로드"""
        print("📚 메모리 데이터 로드 중...")
        
        try:
            # 기존 문제들 로드
            existing_questions = self.get_questions_from_redis()
            
            # 락된 서비스 확인
            locked_service = self.get_locked_service()
            
            # 숏텀 메모리에서 서비스별 데이터 로드
            short_term_data = self.load_short_term_memory()
            
            state["existing_questions"] = existing_questions
            state["locked_service"] = locked_service
            state["short_term_data"] = short_term_data
            
            print(f"📚 {len(existing_questions)}개 기존 문제 로드")
            print(f"🔒 락된 서비스: {locked_service}")
            print(f"📝 숏텀 메모리 데이터: {short_term_data}")
            
        except Exception as e:
            print(f"❌ 메모리 로드 실패: {e}")
            state["existing_questions"] = []
            state["locked_service"] = None
            state["short_term_data"] = {}
        
        return state
    
    def inject_milvus_data(self, state: MainState) -> MainState:
        """2. MilvusDB 연결 정보 주입"""
        print("🔗 MilvusDB 연결 정보 주입 중...")
        
        try:
            # MilvusDB가 연결되어 있지 않으면 빈 데이터로 진행
            if not self.milvus_manager.is_connected:
                print("⚠️ MilvusDB가 연결되지 않음 - 빈 연결 정보로 진행")
                state["milvus_data"] = {
                    "teacher": {"connection_status": False},
                    "farmer": {"connection_status": False},
                    "connection_status": False
                }
                return state
            
            # MilvusDB 연결 정보 제공 (직렬화 가능한 형태로)
            state["milvus_data"] = {
                "connection_status": True,
                "host": self.milvus_manager.host,
                "port": self.milvus_manager.port,
                "embedding_model_name": self.milvus_manager.embedding_model_name
            }
            
            print(f"✅ MilvusDB 연결 정보 주입 완료")
            
        except Exception as e:
            print(f"❌ MilvusDB 연결 정보 주입 실패: {e}")
            state["milvus_data"] = {
                "milvus_manager": None,
                "connection_status": False,
                "error": str(e)
            }
        
        return state
    
    def classify_question_and_service(self, state: MainState) -> MainState:
        """2. 질문 관련성 검사 및 서비스 분류 - 통합 LLM 기반"""
        print("🔍 질문 분석 및 서비스 분류 중...")
        
        user_query = state["user_query"]
        
        prompt = f"""
        다음 사용자 질문을 분석하여 관련성과 적절한 서비스를 한 번에 분류해주세요.

        **저희 서비스 범위:**

        **1. Teacher 서비스 (교육/학습 관련)**
        - **문제 생성**: IT 관련 시험 문제 자동 생성
          * 정보처리기사, 컴퓨터활용능력, ITQ, 워드프로세서, 스프레드시트 등
          * 데이터베이스, 소프트웨어, 프로그래밍, 정보시스템, 하드웨어, 네트워크, 보안, 알고리즘 등
          * 객관식, 주관식 문제 생성
        - **답변 처리**: 사용자 답변 분석 및 피드백
          * 정답 확인 및 해설 제공
          * 오답 분석 및 학습 포인트 제시
          * 단계별 문제 풀이 과정 안내
        - **채점 기능**: 자동 채점 및 성적 관리
          * 객관식 문제 정답 여부 판단
          * 점수 계산 및 성적 통계
          * 학습 진도 추적
        - **학습 지원**: IT 지식 및 개념 설명
          * IT 용어 사전 및 개념 설명
          * 학습 자료 제공 및 추천
          * 개인별 학습 계획 수립

        **2. Farmer 서비스 (농업/재배 관련)**
        - **작물 재배**: 시설작물 재배 방법 및 관리
          * 토마토, 고추, 배추, 상추, 오이, 딸기, 가지, 파프리카 등
          * 파종, 정식, 수확 시기 및 방법
          * 생육 단계별 관리 포인트
        - **시설 관리**: 온실 및 농업 시설 관리
          * 온실, 하우스, 관수시설, 환기시설, 난방시설 관리
          * 온도, 습도, 조도 등 환경 제어
          * 시설 점검 및 유지보수 가이드
        - **재해 대응**: 병해충 및 자연재해 대응
          * 병해충 진단 및 방제 방법
          * 자연재해 예방 및 대응 전략
          * 예방적 관리 방법
        - **기상 정보**: 기상 데이터 기반 농업 관리
          * 온도, 습도, 강수량, 일조량 분석
          * 기상 예보 기반 재배 계획 수립
          * 이상 기상 대응 방안
        - **수확/유통**: 수확 및 판매 전략
          * 수확 시기 및 방법 최적화
          * 저장 및 보관 방법
          * 시장 가격 정보 및 판매 전략
        - **작물 추천**: 지역별, 계절별 작물 추천
          * 지역 특성에 맞는 작물 선택
          * 계절별 재배 가능 작물
          * 수익성 분석 기반 작물 추천

        **관련 없는 질문 예시:**
        - 일반적인 대화, 인사말 ("안녕하세요", "고마워요" 등)
        - 다른 분야의 질문 (의료, 법률, 요리, 여행, 쇼핑 등)
        - 개인적인 상담이나 심리적 문제
        - 뉴스나 정치 관련 질문
        - 게임이나 엔터테인먼트 관련 질문

        **질문: "{user_query}"**

        위 서비스 범위를 고려하여 다음 중 하나로 분류해주세요:
        - teacher: Teacher 서비스가 적합한 IT/컴퓨터 관련 교육, 학습, 시험, 문제, 채점, 용어 설명 등
        - farmer: Farmer 서비스가 적합한 농업, 재배, 작물, 시설, 재해, 기상, 수확, 시장 등
        - irrelevant: 두 서비스 모두 해당하지 않는 질문

        분류 결과만 출력하세요 (teacher/farmer/irrelevant):
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            classification_result = response.content.strip().lower()
            
            # 유효성 검사
            if classification_result not in ["teacher", "farmer", "irrelevant"]:
                classification_result = "irrelevant"
                print(f"⚠️ LLM 응답이 유효하지 않음, 기본값 사용: {classification_result}")
            
            # 관련성과 서비스 분류 동시 설정
            is_relevant = (classification_result != "irrelevant")
            state["is_relevant"] = is_relevant
            state["classified_service"] = classification_result
            
            print(f"🔍 질문 분석 결과: {'✅ 관련' if is_relevant else '❌ 무관'} → {classification_result}")
            
        except Exception as e:
            print(f"❌ LLM 질문 분석 실패: {e}")
            # 오류 시 기본값 설정
            state["is_relevant"] = True
            state["classified_service"] = "irrelevant"
            print("🔍 기본값 사용: 관련 → irrelevant")
        
        return state
    
    def _validate_classification_confidence(self, state: MainState) -> bool:
        """분류 결과의 신뢰성 검증"""
        user_query = state["user_query"].lower()
        classified_service = state["classified_service"]
        
        # 경계선상의 키워드들 (두 서비스 모두에 관련될 수 있는)
        ambiguous_keywords = [
            "데이터", "분석", "관리", "시스템", "정보", "기술", "자동화", 
            "모니터링", "제어", "최적화", "효율성", "생산성"
        ]
        
        # Teacher 관련 강한 키워드들
        teacher_strong_keywords = [
            "문제", "시험", "채점", "정답", "학습", "공부", "교육", "강의",
            "정보처리기사", "컴퓨터활용능력", "itq", "데이터베이스", "프로그래밍",
            "소프트웨어", "하드웨어", "네트워크", "보안", "알고리즘"
        ]
        
        # Farmer 관련 강한 키워드들
        farmer_strong_keywords = [
            "재배", "농작물", "작물", "토마토", "고추", "배추", "상추", "오이", "딸기",
            "온실", "하우스", "시설", "관수", "환기", "병해", "해충", "방제",
            "수확", "시장", "가격", "판매", "유통", "기상", "온도", "습도"
        ]
        
        # 강한 키워드가 있는지 확인
        has_teacher_strong = any(keyword in user_query for keyword in teacher_strong_keywords)
        has_farmer_strong = any(keyword in user_query for keyword in farmer_strong_keywords)
        has_ambiguous = any(keyword in user_query for keyword in ambiguous_keywords)
        
        # 분류 신뢰도 평가
        if classified_service == "teacher" and has_teacher_strong:
            return True  # 높은 신뢰도
        elif classified_service == "farmer" and has_farmer_strong:
            return True  # 높은 신뢰도
        elif has_ambiguous and not (has_teacher_strong or has_farmer_strong):
            return False  # 낮은 신뢰도 (경계선상)
        else:
            return True  # 기본적으로 신뢰
    
    def _refine_classification_if_needed(self, state: MainState) -> MainState:
        """필요시 분류 결과 재검토"""
        if not self._validate_classification_confidence(state):
            print("🤔 분류 신뢰도가 낮음, 재검토 중...")
            
            user_query = state["user_query"]
            
            # 재검토 프롬프트
            refinement_prompt = f"""
            다음 질문의 분류를 다시 한번 신중하게 검토해주세요.
            
            **현재 분류 결과**: {state["classified_service"]}
            
            **질문**: "{user_query}"
            
            이 질문이 정말로 {state["classified_service"]} 서비스에 적합한지 다시 생각해보세요.
            
            **Teacher 서비스**: IT/컴퓨터 관련 교육, 학습, 시험, 문제, 채점, 용어 설명
            **Farmer 서비스**: 농업, 재배, 작물, 시설, 재해, 기상, 수확, 시장
            
            만약 다른 서비스가 더 적합하다고 생각되면 변경해주세요.
            
            최종 분류 결과만 출력하세요 (teacher/farmer/irrelevant):
            """
            
            try:
                response = self.llm.invoke([HumanMessage(content=refinement_prompt)])
                refined_service = response.content.strip().lower()
                
                if refined_service in ["teacher", "farmer", "irrelevant"]:
                    if refined_service != state["classified_service"]:
                        print(f"🔄 분류 결과 변경: {state['classified_service']} → {refined_service}")
                        state["classified_service"] = refined_service
                    else:
                        print("✅ 분류 결과 유지")
                else:
                    print("⚠️ 재검토 결과가 유효하지 않음, 기존 결과 유지")
                    
            except Exception as e:
                print(f"❌ 분류 재검토 실패: {e}")
        
        return state
    
    def check_final_service_consistency(self, state: MainState) -> MainState:
        """4. 서비스 일관성 검사"""
        print("🔒 서비스 일관성 검사 중...")
        
        locked_service = state["locked_service"]
        classified_service = state["classified_service"]
        
        if locked_service is None:
            # 락이 없으면 새로 설정 가능
            state["service_consistent"] = True
            print("🔒 새 세션 시작 가능")
        elif locked_service == classified_service:
            # 같은 서비스면 일관성 있음
            state["service_consistent"] = True
            print(f"🔒 일관성 있음: {locked_service}")
        else:
            # 다른 서비스면 일관성 없음
            state["service_consistent"] = False
            print(f"🔒 일관성 없음: {locked_service} → {classified_service}")
        
        return state
    
    def update_session_data(self, state: MainState) -> MainState:
        """5. 세션 데이터 업데이트"""
        print("📝 세션 데이터 업데이트 중...")
        
        if state["service_consistent"] and state["classified_service"] != "irrelevant":
            # 서비스 락 설정
            self.set_locked_service(state["classified_service"])
            print(f"🔒 서비스 락 설정: {state['classified_service']}")
        
        return state
    
    def execute_farmer(self, state: MainState) -> MainState:
        """6. Farmer 서비스 실행"""
        print("🌾 Farmer 서비스 실행 중...")
        
        try:
            # 숏텀 메모리에서 Farmer 데이터 로드
            short_term_data = state.get("short_term_data", {})
            farmer_data = short_term_data.get("farmer", {})
            
            # MilvusDB 연결 정보 로드
            milvus_data = state.get("milvus_data", {})
            
            # Farmer 상태 준비 (기존 데이터 + 숏텀 메모리 데이터 + MilvusDB 연결 정보)
            farmer_state = {
                "query": state["user_query"],
                "selected_crop": farmer_data.get("selected_crop", ""),
                # MilvusDB 연결 정보 주입
                "milvus_data": milvus_data
            }
            
            # Farmer 그래프 실행
            result = self.farmer.invoke(farmer_state)
            
            state["farmer_result"] = result
            print("🌾 Farmer 실행 완료")
            
        except Exception as e:
            print(f"❌ Farmer 실행 실패: {e}")
            state["farmer_result"] = {"error": str(e)}
        
        return state
    
    def teacher_app(self, state: MainState) -> MainState:
        """7. Teacher 서비스 실행"""
        print("👨‍🏫 Teacher 서비스 실행 중...")
        
        try:
            # 숏텀 메모리에서 Teacher 데이터 로드
            short_term_data = state.get("short_term_data", {})
            teacher_data = short_term_data.get("teacher", {})
            
            # MilvusDB 연결 정보 로드
            milvus_data = state.get("milvus_data", {})
            
            # Teacher 상태 준비 (기존 문제 + 숏텀 메모리 데이터 + MilvusDB 연결 정보)
            teacher_state = {
                "user_query": state["user_query"],
                "intent": self._determine_teacher_intent(state["user_query"]),
                "shared": {
                    "question": [q.get("question", "") for q in state["existing_questions"]],
                    "options": [q.get("options", []) for q in state["existing_questions"]],
                    "answer": [q.get("answer", "") for q in state["existing_questions"]],
                    "explanation": [q.get("explanation", "") for q in state["existing_questions"]],
                    "subject": [q.get("subject", "") for q in state["existing_questions"]],
                    "added_count": teacher_data.get("added_count", 0)  # 최근 추가된 문항 수
                },
                "routing": {
                    "output_mode": "pdf"  # 비대화형 모드
                },
                # MilvusDB 연결 정보 주입
                "milvus_data": milvus_data
            }
            
            # Teacher 그래프 실행 (config 추가)
            config = {
                "configurable": {
                    "thread_id": f"{self.user_id}_{self.chat_id}",
                    "checkpoint_ns": "teacher",
                    "checkpoint_id": f"teacher_{self.session_key}"
                }
            }
            result = self.teacher.graph.invoke(teacher_state, config)
            
            
            state["teacher_result"] = result
            print("👨‍🏫 Teacher 실행 완료")
            
        except Exception as e:
            print(f"❌ Teacher 실행 실패: {e}")
            state["teacher_result"] = {"error": str(e)}
        
        return state
    
    def merge_teacher_result(self, state: MainState) -> MainState:
        """8. Teacher 결과 병합"""
        print("🔄 Teacher 결과 병합 중...")
        
        if state["teacher_result"] and "error" not in state["teacher_result"]:
            # 새로 생성된 문제들 저장
            shared_data = state["teacher_result"].get("shared", {})
            new_questions = []
            
            questions = shared_data.get("question", [])
            options = shared_data.get("options", [])
            answers = shared_data.get("answer", [])
            explanations = shared_data.get("explanation", [])
            subjects = shared_data.get("subject", [])
            
            for i in range(len(questions)):
                if questions[i]:  # 빈 문제가 아닌 경우만
                    new_questions.append({
                        "question": questions[i],
                        "options": options[i] if i < len(options) else [],
                        "answer": answers[i] if i < len(answers) else "1",
                        "explanation": explanations[i] if i < len(explanations) else "",
                        "subject": subjects[i] if i < len(subjects) else ""
                    })
            
            if new_questions:
                saved_count = self.save_questions_to_redis_with_dedup(new_questions)
                print(f"💾 {saved_count}개 새 문제 저장")
        
        return state
    
    def save_memory_data(self, state: MainState) -> MainState:
        """9. 메모리 데이터 저장"""
        print("💾 메모리 데이터 저장 중...")
        
        try:
            # 서비스별 결과를 숏텀 메모리에 저장
            if state["classified_service"] == "teacher" and state["teacher_result"]:
                self.save_teacher_short_term_data_from_state(state["teacher_result"])
            elif state["classified_service"] == "farmer" and state["farmer_result"]:
                self.save_farmer_short_term_data_from_state(state["farmer_result"])
            
            # 채팅 히스토리 저장
            self.save_chat_history(state["user_query"], state["final_response"])
            
        except Exception as e:
            print(f"❌ 메모리 저장 실패: {e}")
        
        return state
    
    def finalize_response(self, state: MainState) -> MainState:
        """10. 최종 응답 생성"""
        print("📝 최종 응답 생성 중...")
        
        if not state["is_relevant"]:
            state["final_response"] = "❌ 이 질문은 저희 서비스 범위를 벗어납니다. 교육/학습 또는 농업/재배 관련 질문을 해주세요."
        elif not state["service_consistent"]:
            state["final_response"] = f"❌ 현재 {state['locked_service']} 서비스가 활성화되어 있습니다. 세션을 초기화하려면 'clear'를 입력하세요."
        elif state["classified_service"] == "teacher" and state["teacher_result"]:
            state["final_response"] = self._format_teacher_response(state["teacher_result"])
        elif state["classified_service"] == "farmer" and state["farmer_result"]:
            state["final_response"] = self._format_farmer_response(state["farmer_result"])
        else:
            state["final_response"] = "❌ 서비스 실행 중 오류가 발생했습니다."
        
        return state
    
    def handle_irrelevant_question(self, state: MainState) -> MainState:
        """무관한 질문 처리"""
        print("❌ 무관한 질문 처리")
        state["final_response"] = "❌ 이 질문은 저희 서비스 범위를 벗어납니다."
        return state
    
    def handle_service_inconsistency(self, state: MainState) -> MainState:
        """서비스 불일치 처리"""
        print("❌ 서비스 불일치 처리")
        state["final_response"] = f"❌ 현재 {state['locked_service']} 서비스가 활성화되어 있습니다. 세션을 초기화하려면 'clear'를 입력하세요."
        return state
    
    def process_query(self, user_query: str) -> str:
        """메인 쿼리 처리 - LangGraph 워크플로우 실행"""
        print(f"\n📝 사용자 질문: {user_query}")
        
        # 특별 명령어 처리
        if user_query.lower() == "clear":
            self.clear_session()
            return "🧹 세션이 초기화되었습니다."
        
        # 초기 상태 설정
        initial_state = MainState(
            user_query=user_query,
            user_id=self.user_id,
            chat_id=self.chat_id,
            session_key=self.session_key,
            existing_questions=[],
            locked_service=None,
            short_term_data={},
            milvus_data={},
            is_relevant=False,
            classified_service="",
            service_consistent=False,
            teacher_result=None,
            farmer_result=None,
            final_response=""
        )
        
        # LangGraph 실행
        try:
            config = {
                "configurable": {
                    "thread_id": self.session_key,
                    "checkpoint_ns": "supervisor",
                    "checkpoint_id": f"supervisor_{self.session_key}"
                }
            }
            
            # 그래프 실행
            final_state = self.graph.invoke(initial_state, config)
            
            return final_state["final_response"]
            
        except Exception as e:
            print(f"❌ LangGraph 워크플로우 실행 실패: {e}")
            return f"❌ 시스템 오류가 발생했습니다: {e}"
    
    # 헬퍼 메서드들
    def _determine_teacher_intent(self, user_query: str) -> str:
        """Teacher intent 자동 분류"""
        if any(keyword in user_query for keyword in ["정답은", "답은", "1,2,3", "1, 2, 3"]):
            return "solution"
        elif any(keyword in user_query for keyword in ["채점", "점수", "맞았나", "정답 확인"]):
            return "score"
        else:
            return "generate"
    
    def _format_teacher_response(self, teacher_result: Dict) -> str:
        """Teacher 응답 포맷팅"""
        if "error" in teacher_result:
            return f"❌ Teacher 실행 오류: {teacher_result['error']}"
        
        # 1. llm_response가 있으면 우선적으로 사용 (가장 자연스러운 응답)
        llm_response = teacher_result.get("llm_response", "").strip()
        if llm_response:
            print(f"📝 Teacher llm_response 사용: {llm_response[:100]}...")
            return llm_response
        
        # 2. llm_response가 없으면 intent와 데이터에 따라 적절한 메시지 생성
        intent = teacher_result.get("intent", "").lower()
        shared_data = teacher_result.get("shared", {})
        questions = shared_data.get("question", [])
        answers = shared_data.get("answer", [])
        explanations = shared_data.get("explanation", [])
        
        print(f"📝 Teacher intent 기반 응답 생성: {intent}")
        
        if intent == "generate" and questions:
            response = "✅ 문제가 생성되었습니다!\n\n"
            for i, question in enumerate(questions):
                if question:
                    response += f"문제 {i+1}: {question[:100]}...\n"
            response += "\n답변을 입력하려면 '정답은 1,2,3' 형식으로 입력하세요."
            return response
        
        elif intent == "solution" and questions and answers:
            response = "✅ 문제 풀이가 완료되었습니다!\n\n"
            for i, (question, answer, explanation) in enumerate(zip(questions, answers, explanations)):
                if question and answer:
                    response += f"문제 {i+1}: {question[:100]}...\n"
                    response += f"정답: {answer}\n"
                    if explanation:
                        response += f"해설: {explanation[:100]}...\n"
                    response += "\n"
            return response
        
        elif intent == "score" and questions and answers:
            response = "✅ 채점이 완료되었습니다!\n\n"
            for i, (question, answer) in enumerate(zip(questions, answers)):
                if question and answer:
                    response += f"문제 {i+1}: {question[:100]}...\n"
                    response += f"정답: {answer}\n\n"
            return response
        
        elif intent == "analyze" and questions:
            response = "✅ 문제 분석이 완료되었습니다!\n\n"
            for i, question in enumerate(questions):
                if question:
                    response += f"문제 {i+1}: {question[:100]}...\n"
            return response
        
        elif questions:
            # intent가 명확하지 않지만 문제가 있는 경우
            response = "✅ Teacher 서비스가 실행되었습니다!\n\n"
            for i, question in enumerate(questions):
                if question:
                    response += f"문제 {i+1}: {question[:100]}...\n"
            return response
        
        return "✅ Teacher 서비스가 실행되었습니다."
    
    def _format_farmer_response(self, farmer_result: Dict) -> str:
        """Farmer 응답 포맷팅"""
        if "error" in farmer_result:
            return f"❌ Farmer 실행 오류: {farmer_result['error']}"
        
        # 1. output이 있으면 우선적으로 사용 (가장 자연스러운 응답)
        output = farmer_result.get("output", "").strip()
        if output:
            print(f"📝 Farmer output 사용: {output[:100]}...")
            return output
        
        # 2. output이 없으면 기본 메시지
        print("📝 Farmer 기본 응답 생성")
        return "✅ Farmer 서비스가 실행되었습니다."
    
    # Redis 관련 메서드들
    def get_locked_service(self) -> Optional[str]:
        """락된 서비스 조회"""
        try:
            locked = self.memory.redis.get(f"lock:{self.session_key}")
            if locked:
                if isinstance(locked, bytes):
                    return locked.decode('utf-8')
                return str(locked)
            return None
        except Exception as e:
            print(f"❌ 락 확인 실패: {e}")
            return None
    
    def set_locked_service(self, service: str):
        """서비스 락 설정"""
        try:
            self.memory.redis.setex(f"lock:{self.session_key}", 3600, service)
            print(f"🔒 서비스 락 설정: {service}")
        except Exception as e:
            print(f"❌ 락 설정 실패: {e}")
    
    def clear_session(self):
        """세션 초기화"""
        try:
            # 락 해제
            self.memory.redis.delete(f"lock:{self.session_key}")
            
            # 문제 데이터 삭제
            pattern = f"questions:{self.session_key}:*"
            keys = self.memory.redis.keys(pattern)
            if keys:
                self.memory.redis.delete(*keys)
                print(f"🗑️ {len(keys)}개 문제 데이터 삭제")
            
            # 사용자 답변 삭제
            self.memory.redis.delete(f"user_answers:{self.session_key}")
            
            print("🧹 세션 초기화 완료 (락 해제 + 데이터 삭제)")
        except Exception as e:
            print(f"❌ 세션 초기화 실패: {e}")
    
    def save_questions_to_redis_with_dedup(self, questions: List[Dict]) -> int:
        """문제 저장 (중복 제거)"""
        saved_count = 0
        try:
            for question_data in questions:
                question_text = question_data.get("question", "")
                options = question_data.get("options", [])
                
                # 중복 검사를 위한 해시 생성
                content_hash = hashlib.md5(
                    f"{question_text}_{json.dumps(options, sort_keys=True)}".encode()
                ).hexdigest()
                
                # Redis에 저장
                key = f"questions:{self.session_key}:{content_hash}"
                self.memory.redis.hset(key, mapping={
                    "question": question_text,
                    "options": json.dumps(options),
                    "answer": question_data.get("answer", "1"),
                    "explanation": question_data.get("explanation", ""),
                    "subject": question_data.get("subject", "")
                })
                self.memory.redis.expire(key, 3600)  # 1시간 TTL
                saved_count += 1
            
            print(f"💾 {saved_count}개 문제 저장 완료")
        except Exception as e:
            print(f"❌ 문제 저장 실패: {e}")
        
        return saved_count
    
    def get_questions_from_redis(self) -> List[Dict]:
        """Redis에서 문제 조회"""
        try:
            pattern = f"questions:{self.session_key}:*"
            keys = self.memory.redis.keys(pattern)
            questions = []
            
            for key in keys:
                data = self.memory.redis.hgetall(key)
                if data:
                    # bytes 처리
                    question_data = {}
                    for k, v in data.items():
                        # Redis에서 가져온 데이터는 이미 문자열
                        
                        if k == "options":
                            try:
                                question_data[k] = json.loads(v)
                            except:
                                question_data[k] = []
                        else:
                            question_data[k] = v
                    
                    questions.append(question_data)
            
            return questions
        except Exception as e:
            print(f"❌ 문제 조회 실패: {e}")
            return []
    
    # 숏텀 메모리 관련 메서드들
    def load_short_term_memory(self) -> Dict[str, Any]:
        """숏텀 메모리에서 서비스별 데이터 로드"""
        try:
            key = f"short_term:{self.session_key}"
            data = self.memory.redis.get(key)
            if data:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                return json.loads(data)
            return {}
        except Exception as e:
            print(f"❌ 숏텀 메모리 로드 실패: {e}")
            return {}
    
    def save_short_term_memory(self, data: Dict[str, Any]) -> None:
        """숏텀 메모리에 서비스별 데이터 저장"""
        try:
            key = f"short_term:{self.session_key}"
            self.memory.redis.setex(key, 3600, json.dumps(data, ensure_ascii=False))  # 1시간 TTL
            print(f"💾 숏텀 메모리 저장 완료: {key}")
        except Exception as e:
            print(f"❌ 숏텀 메모리 저장 실패: {e}")
    
    def save_teacher_short_term_data_from_state(self, teacher_result: Dict[str, Any]) -> None:
        """Teacher 서비스 결과를 숏텀 메모리에 저장 (상태에서 직접 추출)"""
        try:
            shared_data = teacher_result.get("shared", {})
            
            # 기존 숏텀 메모리 데이터 로드
            short_term_data = self.load_short_term_memory()
            
            # Teacher 데이터 업데이트 (added_count는 shared에서 직접 가져옴)
            teacher_data = {
                "questions": shared_data.get("question", []),
                "options": shared_data.get("options", []),
                "answer": shared_data.get("answer", []),
                "explanation": shared_data.get("explanation", []),
                "subject": shared_data.get("subject", []),
                "added_count": shared_data.get("added_count", 0)  # shared에서 직접 가져옴
            }
            
            short_term_data["teacher"] = teacher_data
            self.save_short_term_memory(short_term_data)
            
            print(f"💾 Teacher 숏텀 메모리 저장: {len(teacher_data['questions'])}개 문제, added_count={teacher_data['added_count']}")
            
        except Exception as e:
            print(f"❌ Teacher 숏텀 메모리 저장 실패: {e}")
    
    def save_farmer_short_term_data_from_state(self, farmer_result: Dict[str, Any]) -> None:
        """Farmer 서비스 결과를 숏텀 메모리에 저장 (상태에서 직접 추출)"""
        try:
            # 기존 숏텀 메모리 데이터 로드
            short_term_data = self.load_short_term_memory()
            
            # Farmer 데이터 업데이트
            farmer_data = {
                # "selected_crop": farmer_result.get("selected_crop", "")
            }
            
            short_term_data["farmer"] = farmer_data
            self.save_short_term_memory(short_term_data)
            
            print(f"💾 Farmer 숏텀 메모리 저장: selected_crop={farmer_data['selected_crop']}")
            
        except Exception as e:
            print(f"❌ Farmer 숏텀 메모리 저장 실패: {e}")
    
    def save_chat_history(self, user_query: str, response: str) -> None:
        """채팅 히스토리 저장"""
        try:
            key = f"chat_history:{self.session_key}"
            chat_entry = {
                "user_query": user_query,
                "response": response,
                "timestamp": int(time.time())
            }
            
            # 기존 히스토리 로드
            existing_history = self.memory.redis.lrange(key, 0, -1)
            history = []
            for entry in existing_history:
                if isinstance(entry, bytes):
                    entry = entry.decode('utf-8')
                history.append(json.loads(entry))
            
            # 새 엔트리 추가
            history.append(chat_entry)
            
            # 최근 50개만 유지
            if len(history) > 50:
                history = history[-50:]
            
            # Redis에 저장
            self.memory.redis.delete(key)  # 기존 데이터 삭제
            for entry in history:
                self.memory.redis.rpush(key, json.dumps(entry, ensure_ascii=False))
            
            self.memory.redis.expire(key, 3600)  # 1시간 TTL
            
        except Exception as e:
            print(f"❌ 채팅 히스토리 저장 실패: {e}")
    
    def get_chat_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """채팅 히스토리 조회"""
        try:
            key = f"chat_history:{self.session_key}"
            history = self.memory.redis.lrange(key, -limit, -1)  # 최근 N개
            
            result = []
            for entry in history:
                if isinstance(entry, bytes):
                    entry = entry.decode('utf-8')
                result.append(json.loads(entry))
            
            return result
        except Exception as e:
            print(f"❌ 채팅 히스토리 조회 실패: {e}")
            return []


def main():
    """CLI 인터페이스"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Main Orchestrator")
    parser.add_argument("--user_id", default="test_user", help="사용자 ID")
    parser.add_argument("--chat_id", default="test_chat", help="채팅 ID")
    parser.add_argument("--query", help="실행할 쿼리 (한 번만 실행)")
    
    args = parser.parse_args()
    
    # 오케스트레이터 초기화
    orchestrator = MainOrchestrator(args.user_id, args.chat_id)
    
    # 그래프 시각화 생성 (선택적)
    try:
        orchestrator.visualize_graph("supervisor_langgraph.png")
    except Exception as e:
        print(f"⚠️ 그래프 시각화 실패 (계속 진행): {e}")
    
    if args.query:
        # 한 번만 실행
        result = orchestrator.process_query(args.query)
        print(f"\n결과: {result}")
    else:
        # 대화형 모드
        print("🚀 Main Orchestrator 시작 (종료: 'quit')")
        while True:
            try:
                user_input = input("\n사용자: ").strip()
                if user_input.lower() in ['quit', 'exit', '종료']:
                    break
                
                if user_input:
                    result = orchestrator.process_query(user_input)
                    print(f"\n시스템: {result}")
            except KeyboardInterrupt:
                print("\n👋 종료합니다.")
                break


if __name__ == "__main__":
    main()