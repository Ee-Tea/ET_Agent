import os
from typing import TypedDict, List, Dict, Literal, Optional, Tuple, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import connections
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
import json, re
from langchain_openai import ChatOpenAI
from ..base_agent import BaseAgent
from datetime import datetime

# LangGraph HITL 관련 import 추가
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

# 검색 에이전트 import (retrieve_agent 연동)
try:
    from ..retrieve.retrieve_agent import retrieve_agent
    SEARCH_AGENT_AVAILABLE = True
    print("✅ 검색 에이전트 import 성공")
except ImportError as e:
    SEARCH_AGENT_AVAILABLE = False
    print(f"⚠️ 검색 에이전트를 import할 수 없습니다: {e}")
    print("   - 검색 기능은 비활성화됩니다.")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ✅ 상태 정의 (HITL 기능 추가)
class SolutionState(TypedDict):
    # 사용자 입력
    user_input_txt: str

    # 문제리스트, 문제, 보기
    user_problem: str
    user_problem_options: List[str]
    
    # vectorstore는 상태에 저장하지 않음 (직렬화 문제 방지)

    retrieved_docs: List[Document]
    similar_questions_text : str

    # 문제 해답/풀이/과목 생성
    generated_answer: str         # 해답
    generated_explanation: str   # 풀이
    generated_subject: str

    # HITL 관련 상태
    user_feedback: str           # 사용자 피드백
    feedback_type: str           # 피드백 유형 (comprehension, clarification, improvement)
    search_results: str          # 검색 에이전트 결과
    improved_explanation: str    # 개선된 풀이
    interaction_count: int       # 상호작용 횟수
    max_interactions: int        # 최대 상호작용 횟수
    
    # 품질 평가 관련 상태
    quality_scores: Dict[str, float]  # 세부 품질 점수들
    total_quality_score: float        # 총 품질 점수
    quality_threshold: float          # 품질 임계값
    
    results: List[Dict]
    validated: bool
    retry_count: int             # 검증 실패 시 재시도 횟수

    chat_history: List[str]
    
class SolutionAgent(BaseAgent):
    """Human-in-the-Loop가 포함된 문제 해답/풀이 생성 에이전트"""

    def __init__(self, max_interactions: int = 5, hitl_mode: str = "smart"):
        self.max_interactions = max_interactions
        self.hitl_mode = hitl_mode  # "auto", "manual", "smart"
        self.memory = self._create_custom_checkpointer()  # 체크포인터 먼저 초기화
        self.graph = self._create_graph()  # 그 다음 그래프 생성
        
    @property
    def name(self) -> str:
        return "SolutionAgent"

    @property
    def description(self) -> str:
        return "사용자와의 상호작용을 통해 풀이를 개선하는 문제 해답/풀이 생성 에이전트입니다."

    def _llm(self, temperature: float = 0):
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
            model=OPENAI_LLM_MODEL,
            temperature=temperature,
        )

    def _create_graph(self) -> StateGraph:
        """HITL 워크플로우 그래프 생성"""

        print("📚 HITL LangGraph 흐름 구성 중...")
        
        graph = StateGraph(SolutionState)

        # 기본 노드들
        graph.add_node("search_similarity", self._search_similar_questions)
        graph.add_node("generate_solution", self._generate_solution)
        graph.add_node("validate", self._validate_solution)
        
        # HITL 노드들
        graph.add_node("collect_user_feedback", self._collect_user_feedback)
        graph.add_node("process_feedback", self._process_feedback)
        graph.add_node("search_additional_info", self._search_additional_info)
        graph.add_node("improve_explanation", self._improve_explanation)
        graph.add_node("store", self._store_to_vector_db)

        # 워크플로우 설정
        graph.set_entry_point("search_similarity")
        graph.add_edge("search_similarity", "generate_solution")
        graph.add_edge("generate_solution", "validate")
        
        # HITL 조건부 엣지
        graph.add_conditional_edges(
            "validate", 
            self._route_after_validation,
            {"ok": "store", "feedback_needed": "collect_user_feedback", "retry": "generate_solution"}
        )
        
        graph.add_conditional_edges(
            "collect_user_feedback",
            self._route_after_feedback,
            {"continue": "store", "improve": "process_feedback", "search": "search_additional_info"}
        )
        
        graph.add_edge("process_feedback", "improve_explanation")
        graph.add_edge("search_additional_info", "improve_explanation")
        graph.add_edge("improve_explanation", "collect_user_feedback")

        return graph.compile(checkpointer=self.memory)
    
    def _route_after_validation(self, state: SolutionState) -> str:
        """검증 후 라우팅 결정 (HITL 모드에 따라)"""
        if state["validated"]:
            return "ok"
        elif state.get("retry_count", 0) < 3:
            return "retry"
        else:
            # HITL 모드에 따라 결정
            if self.hitl_mode == "auto":
                return "ok"  # 자동 모드에서는 검증 실패해도 통과
            elif self.hitl_mode == "smart":
                # 스마트 모드: 풀이 품질을 평가하여 결정
                return self._smart_hitl_decision(state)
            else:  # manual 모드
                return "feedback_needed"
    
    def _smart_hitl_decision(self, state: SolutionState) -> str:
        """스마트 HITL: 풀이 품질을 평가하여 HITL 적용 여부 결정"""
        # 다차원 품질 평가 수행
        quality_score = self._evaluate_solution_quality(state)
        
        print(f"📊 풀이 품질 점수: {quality_score:.2f}/100")
        
        # 품질 점수에 따른 HITL 적용 여부 결정
        if quality_score >= 80:
            print("✅ 품질이 높음 - 자동 통과")
            return "ok"
        elif quality_score >= 60:
            print("⚠️ 품질이 보통 - HITL 적용")
            return "feedback_needed"
        else:
            print("❌ 품질이 낮음 - HITL 필수 적용")
            return "feedback_needed"
    
    def _evaluate_solution_quality(self, state: SolutionState) -> float:
        """다차원 풀이 품질 평가 (0-100점)"""
        llm = self._llm(0)
        
        # 1. 정확성 평가 (30점)
        accuracy_score = self._evaluate_accuracy(state, llm)
        
        # 2. 완성도 평가 (25점)
        completeness_score = self._evaluate_completeness(state, llm)
        
        # 3. 이해도 평가 (25점)
        clarity_score = self._evaluate_clarity(state, llm)
        
        # 4. 논리성 평가 (20점)
        logic_score = self._evaluate_logic(state, llm)
        
        # 가중 평균 계산
        total_score = (
            accuracy_score * 0.30 +
            completeness_score * 0.25 +
            clarity_score * 0.25 +
            logic_score * 0.20
        )
        
        # 품질 점수들을 state에 저장
        state["quality_scores"] = {
            "accuracy": accuracy_score,
            "completeness": completeness_score,
            "clarity": clarity_score,
            "logic": logic_score
        }
        state["total_quality_score"] = total_score
        
        print(f"📈 품질 세부 점수:")
        print(f"   정확성: {accuracy_score:.1f}/100 (가중치: 30%)")
        print(f"   완성도: {completeness_score:.1f}/100 (가중치: 25%)")
        print(f"   이해도: {clarity_score:.1f}/100 (가중치: 25%)")
        print(f"   논리성: {logic_score:.1f}/100 (가중치: 20%)")
        print(f"   총점: {total_score:.1f}/100")
        
        return total_score
    
    def _evaluate_accuracy(self, state: SolutionState, llm) -> float:
        """정확성 평가 (30점)"""
        prompt = f"""
        다음 풀이의 정확성을 평가해주세요:
        
        문제: {state['user_problem']}
        보기: {state['user_problem_options']}
        정답: {state['generated_answer']}
        풀이: {state['generated_explanation']}
        
        다음 기준으로 평가하세요:
        1. 정답이 올바른가? (10점)
        2. 풀이 과정이 정확한가? (10점)
        3. 기술적 내용이 정확한가? (10점)
        
        각 항목별로 점수를 매기고, 총점을 계산하여 0-100 사이의 숫자로만 답변하세요.
        """
        
        try:
            response = llm.invoke(prompt)
            score_text = response.content.strip()
            # 숫자만 추출
            score_match = re.search(r'(\d+)', score_text)
            if score_match:
                return min(100, max(0, float(score_match.group(1))))
            return 70  # 기본값
        except:
            return 70
    
    def _evaluate_completeness(self, state: SolutionState, llm) -> float:
        """완성도 평가 (25점)"""
        prompt = f"""
        다음 풀이의 완성도를 평가해주세요:
        
        문제: {state['user_problem']}
        풀이: {state['generated_explanation']}
        
        다음 기준으로 평가하세요:
        1. 핵심 개념을 모두 포함하는가? (10점)
        2. 단계별 설명이 충분한가? (10점)
        3. 예시나 비유가 적절한가? (5점)
        
        각 항목별로 점수를 매기고, 총점을 계산하여 0-100 사이의 숫자로만 답변하세요.
        """
        
        try:
            response = llm.invoke(prompt)
            score_text = response.content.strip()
            score_match = re.search(r'(\d+)', score_text)
            if score_match:
                return min(100, max(0, float(score_match.group(1))))
            return 70
        except:
            return 70
    
    def _evaluate_clarity(self, state: SolutionState, llm) -> float:
        """이해도 평가 (25점)"""
        prompt = f"""
        다음 풀이의 이해도를 평가해주세요:
        
        풀이: {state['generated_explanation']}
        
        다음 기준으로 평가하세요:
        1. 문장이 명확하고 읽기 쉬운가? (10점)
        2. 전문 용어가 적절히 설명되었는가? (10점)
        3. 전체적인 흐름이 자연스러운가? (5점)
        
        각 항목별로 점수를 매기고, 총점을 계산하여 0-100 사이의 숫자로만 답변하세요.
        """
        
        try:
            response = llm.invoke(prompt)
            score_text = response.content.strip()
            score_match = re.search(r'(\d+)', score_text)
            if score_match:
                return min(100, max(0, float(score_match.group(1))))
            return 70
        except:
            return 70
    
    def _evaluate_logic(self, state: SolutionState, llm) -> float:
        """논리성 평가 (20점)"""
        prompt = f"""
        다음 풀이의 논리성을 평가해주세요:
        
        문제: {state['user_problem']}
        풀이: {state['generated_explanation']}
        
        다음 기준으로 평가하세요:
        1. 논리적 추론이 올바른가? (10점)
        2. 인과관계가 명확한가? (10점)
        
        각 항목별로 점수를 매기고, 총점을 계산하여 0-100 사이의 숫자로만 답변하세요.
        """
        
        try:
            response = llm.invoke(prompt)
            score_text = response.content.strip()
            score_match = re.search(r'(\d+)', score_text)
            if score_match:
                return min(100, max(0, float(score_match.group(1))))
            return 70
        except:
            return 70
    
    def _route_after_feedback(self, state: SolutionState) -> str:
        """사용자 피드백 후 라우팅 결정"""
        feedback = state.get("user_feedback", "").lower()
        
        if "만족" in feedback or "좋음" in feedback or "이해" in feedback:
            return "continue"
        elif "검색" in feedback or "찾아" in feedback or "설명" in feedback:
            return "search"
        else:
            return "improve"
    
    #----------------------------------------nodes------------------------------------------------------

    def _search_similar_questions(self, state: SolutionState) -> SolutionState:
        print("\n🔍 [1단계] 유사 문제 검색 시작")
        print(state["user_problem"], state["user_problem_options"])
        
        # vectorstore는 외부에서 전달받아야 함 (직렬화 문제 방지)
        vectorstore = getattr(self, '_current_vectorstore', None)
        if vectorstore is None:
            print("⚠️ 벡터스토어가 없어 유사 문제 검색을 건너뜁니다.")
            state["retrieved_docs"] = []
            state["similar_questions_text"] = ""
            print("🔍 [1단계] 유사 문제 검색 함수 종료 (건너뜀)")
            return state
        
        try:
            results = vectorstore.similarity_search(state["user_problem"], k=3)
        except Exception as e:
            print(f"⚠️ 유사 문제 검색 실패: {e}")
            results = []
        
        similar_questions = []
        for i, doc in enumerate(results):
            metadata = doc.metadata
            options = json.loads(metadata.get("options", "[]"))
            answer = metadata.get("answer", "")
            explanation = metadata.get("explanation", "")
            subject = metadata.get("subject", "기타")

            formatted = f"""[유사문제 {i+1}]
                문제: {doc.page_content}
                보기:
                """ + "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)]) + f"""
                정답: {answer}
                풀이: {explanation}
                과목: {subject}
                """
            similar_questions.append(formatted)
        
        state["retrieved_docs"] = results
        state["similar_questions_text"] = "\n\n".join(similar_questions) 

        print(f"유사 문제 {len(results)}개 검색 완료.")
        print("🔍 [1단계] 유사 문제 검색 함수 종료")
        return state

    def _generate_solution(self, state: SolutionState) -> SolutionState:
        print("\n✏️ [2단계] 해답 및 풀이 생성 시작")

        llm_gen = self._llm(0.5)  

        similar_problems = state.get("similar_questions_text", "")
        print("유사 문제들:\n", similar_problems[:100])

        prompt = f"""
            사용자가 입력한 질문:
            {state['user_input_txt']}

            다음은 사용자가 입력한 문제:
            {state['user_problem']}
            {state['user_problem_options']}

            아래는 이 문제와 유사한 문제들:
            {similar_problems}

            1. 사용자가 입력한 문제의 **정답**의 보기 번호를 정답으로 작성해 주세요.
            2. 이어서 그 정답인 근거를 담은 **풀이 과정**을 상세히 설명해 주세요.
            3. 이 문제의 과목을 정보처리기사 과목 5개 중에서 가장 적합한 것으로 지정해 주세요. 유사 문제들의 과목을 참고해도 좋습니다. [소프트웨어설계, 소프트웨어개발, 데이터베이스구축, 프로그래밍언어활용, 정보시스템구축관리]

            출력 형식:
            정답: ...
            풀이: ...
            과목: ...
        """

        response = llm_gen.invoke(prompt)
        result = response.content.strip()
        print("🧠 LLM 응답 완료")

        answer_match = re.search(r"정답:\s*(.+)", result)
        explanation_match = re.search(r"풀이:\s*(.+)", result, re.DOTALL)
        subject_match = re.search(r"과목:\s*(.+)", result)
        
        state["generated_answer"] = answer_match.group(1).strip() if answer_match else ""
        state["generated_explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        state["generated_subject"] = subject_match.group(1).strip() if subject_match else "기타"

        state["chat_history"].append(f"Q: {state['user_input_txt']}\nP: {state['user_problem']}\nA: {state['generated_answer']}\nE: {state['generated_explanation']}")

        return state

    # ✅ 정합성 검증 (간단히 길이 기준 사용)

    def _validate_solution(self, state: SolutionState) -> SolutionState:
        print("\n🧐 [3단계] 정합성 검증 시작")
        
        llm = self._llm(0)

        validation_prompt = f"""
        사용자 요구사항: {state['user_input_txt']}

        문제 질문: {state['user_problem']}
        문제 보기: {state['user_problem_options']}

        생성된 정답: {state['generated_answer']}
        생성된 풀이: {state['generated_explanation']}
        생성된 과목: {state['generated_subject']}

        생성된 해답과 풀이, 과목이 문제와 사용자 요구사항에 맞고, 논리적 오류나 잘못된 정보가 없습니까?
        적절하다면 '네', 그렇지 않다면 '아니오'로만 답변하세요.
        """

        validation_response = llm.invoke(validation_prompt)
        result_text = validation_response.content.strip().lower()

        # ✅ '네'가 포함된 응답일 경우에만 유효한 풀이로 판단
        print("📌 검증 응답:", result_text)
        
        # manual 모드에서는 항상 HITL 적용
        if self.hitl_mode == "manual":
            state["validated"] = False
            print(" Manual 모드: 강제로 HITL 적용")
        else:
            state["validated"] = "네" in result_text
        
        if not state["validated"]:
            state["retry_count"] = state.get("retry_count", 0) + 1
            print(f"⚠️ 검증 실패 (재시도 {state['retry_count']}/5)")
        else:
            print("✅ 검증 결과: 통과")
            
        return state

    def _collect_user_feedback(self, state: SolutionState) -> SolutionState:
        """사용자로부터 풀이에 대한 피드백 수집 (interrupt 사용)"""
        print("\n💬 [HITL] 사용자 피드백 수집")
        
        # 현재 풀이 상태 표시
        print(f"\n📝 현재 풀이:")
        print(f"정답: {state['generated_answer']}")
        print(f"풀이: {state['generated_explanation']}")
        print(f"과목: {state['generated_subject']}")
        
        # interrupt를 사용하여 사용자 입력 받기
        feedback_query = {
            "query": "풀이에 대한 의견을 자유롭게 입력해주세요",
            "examples": {
                "이해됨": ["이해가 됩니다", "좋습니다", "만족합니다", "충분합니다", "괜찮습니다"],
                "더 쉬운 풀이 필요": ["더 쉽게 설명해주세요", "복잡해요", "어려워요", "간단하게", "초보자도 이해할 수 있게"],
                "용어 설명 필요": ["이 용어가 뭔지 모르겠어요", "설명이 부족해요", "용어를 더 자세히", "개념 설명 추가"]
            },
            "current_problem": state['user_problem'],
            "current_answer": state['generated_answer'],
            "current_explanation": state['generated_explanation']
        }
        
        print("\n💭 풀이에 대한 의견을 자유롭게 입력해주세요:")
        print("예시:")
        print("- '이해가 됩니다', '좋습니다', '만족합니다' → 이해됨")
        print("- '더 쉽게 설명해주세요', '복잡해요', '어려워요' → 더 쉬운 풀이 필요")
        print("- '이 용어가 뭔지 모르겠어요', '설명이 부족해요' → 용어 설명 필요")
        
        # interrupt를 사용하여 실행 일시 중단 및 사용자 입력 대기
        human_response = interrupt(feedback_query)
        
        # Command 객체에서 사용자 입력 추출
        if isinstance(human_response, dict) and "data" in human_response:
            user_input = human_response["data"]
        else:
            user_input = str(human_response)
        
        if not user_input:
            # 입력이 없으면 기본값 설정
            user_input = "풀이를 더 쉽게 설명해주세요"
            print(f"⚠️ 입력이 없어 기본값을 사용합니다: {user_input}")
        
        state["user_feedback"] = user_input
        state["interaction_count"] = state.get("interaction_count", 0) + 1
        
        print(f"✅ 피드백 수집 완료: {user_input}")
        
        return state

    def _route_after_feedback(self, state: SolutionState) -> str:
        """사용자 피드백을 LLM이 분석하여 라우팅 결정"""
        print(f"\n🧠 [HITL] 사용자 피드백 분석 중...")
        
        llm = self._llm(0.1)  # 일관된 분류를 위해 낮은 temperature 사용
        
        analysis_prompt = f"""
        사용자가 제시한 풀이에 대한 피드백을 분석하여 다음 3가지 카테고리 중 하나로 분류해주세요:

        사용자 피드백: {state['user_feedback']}

        분류 기준:
        1. "이해됨" - 사용자가 풀이를 이해했고 만족하는 경우
           예시: "이해가 됩니다", "좋습니다", "만족합니다", "충분합니다", "괜찮습니다"
        
        2. "더 쉬운 풀이 필요" - 사용자가 풀이가 너무 복잡하거나 어렵다고 느끼는 경우
           예시: "더 쉽게 설명해주세요", "복잡해요", "어려워요", "간단하게", "초보자도 이해할 수 있게"
        
        3. "용어 설명 필요" - 사용자가 특정 용어나 개념에 대한 추가 설명을 요청하는 경우
           예시: "이 용어가 뭔지 모르겠어요", "설명이 부족해요", "용어를 더 자세히", "개념 설명 추가"

        위 3가지 중 하나로만 답변하세요. 답변은 반드시 다음 중 하나여야 합니다:
        - 이해됨
        - 더 쉬운 풀이 필요  
        - 용어 설명 필요
        """
        
        try:
            response = llm.invoke(analysis_prompt)
            feedback_type = response.content.strip()
            
            print(f"📊 LLM 분석 결과: {feedback_type}")
            
            # 분류 결과를 state에 저장
            if "이해됨" in feedback_type:
                state["feedback_type"] = "comprehension"
                print("✅ 사용자 의도: 이해됨 → 바로 저장으로 진행")
                return "continue"
            elif "더 쉬운 풀이 필요" in feedback_type:
                state["feedback_type"] = "improvement"
                print("⚠️ 사용자 의도: 더 쉬운 풀이 필요 → 풀이 개선 진행")
                return "improve"
            elif "용어 설명 필요" in feedback_type:
                state["feedback_type"] = "clarification"
                print("🔍 사용자 의도: 용어 설명 필요 → 검색 노드 실행")
                return "search"
            else:
                # 명확하지 않은 경우 기본값으로 개선 진행
                print("⚠️ 명확하지 않은 피드백 → 기본값으로 개선 진행")
                state["feedback_type"] = "improvement"
                return "improve"
                
        except Exception as e:
            print(f"⚠️ LLM 분석 중 오류 발생: {e}")
            print("⚠️ 기본값으로 개선 진행")
            state["feedback_type"] = "improvement"
            return "improve"

    def _process_feedback(self, state: SolutionState) -> SolutionState:
        """사용자 피드백을 분석하고 처리 방향 결정"""
        print(f"\n🔄 [HITL] 피드백 처리: {state['feedback_type']}")
        
        feedback_type = state.get("feedback_type", "improvement")
        
        if feedback_type == "improvement":
            print("📝 풀이를 더 쉽게 개선하겠습니다.")
        elif feedback_type == "clarification":
            print("🔍 특정 용어에 대해 더 자세히 설명하겠습니다.")
        elif feedback_type == "search":
            print("🔎 관련 정보를 검색하겠습니다.")
        
        return state

    def _search_additional_info(self, state: SolutionState) -> SolutionState:
        """검색 에이전트를 사용하여 추가 정보 검색"""
        print(f"\n🔎 [HITL] 추가 정보 검색 시작")
        
        if not SEARCH_AGENT_AVAILABLE:
            print("⚠️ 검색 에이전트를 사용할 수 없습니다.")
            state["search_results"] = "검색 에이전트를 사용할 수 없습니다."
            return state
        
        try:
            # 검색 쿼리 구성 - 유저 피드백 포함
            user_feedback = state.get('user_feedback', '')
            search_query = f"{state['user_problem']} {state['generated_explanation']} {user_feedback}"
            
            print(f"🔍 검색 쿼리: {search_query[:100]}...")
            print(f"💬 유저 피드백: {user_feedback}")
            
            # 검색 에이전트 실행 (invoke 메서드 사용)
            search_results = retrieve_agent().invoke({
                "retrieval_question": search_query
            })
            
            if search_results and "answer" in search_results:
                # 검색 결과를 텍스트로 변환
                answer_content = search_results.get("answer", "")
                state["search_results"] = f"[검색결과]\n{answer_content}"
                print(f"✅ 검색 완료: 답변 길이 {len(answer_content)}자")
            else:
                state["search_results"] = "검색 결과를 찾을 수 없습니다."
                print("⚠️ 검색 결과가 없습니다.")
                
        except Exception as e:
            print(f"⚠️ 검색 중 오류 발생: {e}")
            state["search_results"] = f"검색 중 오류가 발생했습니다: {e}"
        
        return state

    def _improve_explanation(self, state: SolutionState) -> SolutionState:
        """사용자 피드백과 검색 결과를 바탕으로 풀이 개선"""
        print(f"\n✨ [HITL] 풀이 개선 시작")
        
        llm = self._llm(0.3)
        feedback_type = state.get("feedback_type", "improvement")
        
        # 피드백 유형에 따른 개선 프롬프트 구성
        if feedback_type == "improvement":
            improvement_prompt = f"""
            원본 문제: {state['user_problem']}
            원본 보기: {state['user_problem_options']}
            원본 정답: {state['generated_answer']}
            원본 풀이: {state['generated_explanation']}
            
            사용자 피드백: {state['user_feedback']}
            
            위 정보를 바탕으로 풀이를 더 쉽고 이해하기 쉽게 개선해주세요:
            
            개선 요구사항:
            1. 복잡한 용어를 쉬운 말로 바꾸기
            2. 단계별로 명확하게 설명하기
            3. 초보자도 이해할 수 있도록 간단하게
            4. 필요시 비유나 예시 추가하기
            
            출력 형식:
            정답: [정답]
            풀이: [개선된 풀이]
            과목: [과목]
            개선사항: [어떤 부분을 어떻게 개선했는지 설명]
            """
            
        elif feedback_type == "clarification":
            improvement_prompt = f"""
            원본 문제: {state['user_problem']}
            원본 보기: {state['user_problem_options']}
            원본 정답: {state['generated_answer']}
            원본 풀이: {state['generated_explanation']}
            
            사용자 피드백: {state['user_feedback']}
            
            추가 검색 결과:
            {state.get('search_results', '검색 결과 없음')}
            
            위 정보를 바탕으로 풀이를 개선해주세요:
            
            개선 요구사항:
            1. 사용자가 요청한 용어나 개념에 대한 상세한 설명 추가
            2. 검색 결과를 활용하여 관련 개념 보강
            3. 용어의 정의와 예시 제공
            4. 전체적인 맥락에서 이해할 수 있도록 설명
            5. 사용자 피드백에서 언급된 구체적인 부분에 집중하여 설명
             
            출력 형식:
            정답: [정답]
            풀이: [개선된 풀이]
            과목: [과목]
            개선사항: [어떤 부분을 어떻게 개선했는지 설명]
            """
            
        else:  # 기본 개선
            improvement_prompt = f"""
            원본 문제: {state['user_problem']}
            원본 보기: {state['user_problem_options']}
            원본 정답: {state['generated_answer']}
            원본 풀이: {state['generated_explanation']}
            
            사용자 피드백: {state['user_feedback']}
            
            위 정보를 바탕으로 풀이를 개선해주세요:
            
            개선 요구사항:
            1. 사용자 피드백에 맞게 적절히 개선
            2. 이해하기 쉽고 명확하게 설명
            3. 필요한 경우 추가 정보나 예시 제공
            
            출력 형식:
            정답: [정답]
            풀이: [개선된 풀이]
            과목: [과목]
            개선사항: [어떤 부분을 어떻게 개선했는지 설명]
            """
        
        try:
            response = llm.invoke(improvement_prompt)
            result = response.content.strip()
            
            # 개선된 결과 파싱
            answer_match = re.search(r"정답:\s*(.+)", result)
            explanation_match = re.search(r"풀이:\s*(.+)", result, re.DOTALL)
            subject_match = re.search(r"과목:\s*(.+)", result)
            improvement_match = re.search(r"개선사항:\s*(.+)", result, re.DOTALL)
            
            if answer_match:
                state["generated_answer"] = answer_match.group(1).strip()
            if explanation_match:
                state["improved_explanation"] = explanation_match.group(1).strip()
            if subject_match:
                state["generated_subject"] = subject_match.group(1).strip()
            if improvement_match:
                improvement_note = improvement_match.group(1).strip()
            else:
                improvement_note = f"사용자 피드백({feedback_type})에 따라 풀이를 개선했습니다."
            
            # 개선된 풀이를 메인 풀이로 설정
            if state.get("improved_explanation"):
                state["generated_explanation"] = state["improved_explanation"]
            
            # 채팅 히스토리에 개선 과정 추가
            state["chat_history"].append(f"개선({feedback_type}): {improvement_note}")
            
            print(f"✅ 풀이 개선 완료: {improvement_note}")
            print(f"📝 개선 유형: {feedback_type}")
            
        except Exception as e:
            print(f"⚠️ 풀이 개선 중 오류 발생: {e}")
            state["improved_explanation"] = state["generated_explanation"]
        
        return state

    # ✅ 임베딩 후 벡터 DB 저장
    def _store_to_vector_db(self, state: SolutionState) -> SolutionState:  
        print("\n🧩 [4단계] 임베딩 및 벡터 DB 저장 시작")

        # vectorstore는 외부에서 전달받아야 함 (직렬화 문제 방지)
        vectorstore = getattr(self, '_current_vectorstore', None)
        if vectorstore is None:
            print("⚠️ 벡터스토어가 없어 DB 저장을 건너뜁니다.")
        else:
            try:
                # 중복 문제 확인
                similar = vectorstore.similarity_search(state["user_problem"], k=1)
                if similar and state["user_problem"].strip() in similar[0].page_content:
                    print("⚠️ 동일한 문제가 존재하여 저장 생략")
                else:
                    # 문제, 해답, 풀이를 각각 metadata로 저장
                    doc = Document(
                        page_content=state["user_problem"],
                        metadata={
                            "options": json.dumps(state.get("user_problem_options", [])), 
                            "answer": state["generated_answer"],
                            "explanation": state["generated_explanation"],
                            "subject": state["generated_subject"],
                        }
                    )
                    vectorstore.add_documents([doc])
                    print("✅ 문제+해답+풀이 저장 완료")
            except Exception as e:
                print(f"⚠️ 벡터 DB 저장 중 오류 발생: {e}")
                print("   - 결과만 저장하고 계속 진행합니다.")

        # 결과를 state에 저장 (항상 실행)
        print(f"\n📝 결과 저장 시작:")
        print(f"   - 현재 문제: {state['user_problem'][:50]}...")
        print(f"   - 생성된 정답: {state['generated_answer'][:30]}...")
        print(f"   - 검증 상태: {state['validated']}")
        print(f"   - 상호작용 횟수: {state.get('interaction_count', 0)}")
        
        item = {
            "user_problem": state["user_problem"],
            "user_problem_options": state["user_problem_options"],
            "generated_answer": state["generated_answer"],
            "generated_explanation": state["generated_explanation"],
            "generated_subject": state["generated_subject"],
            "validated": state["validated"],
            "interaction_count": state.get("interaction_count", 0),
            "user_feedback": state.get("user_feedback", ""),
            "quality_scores": state.get("quality_scores", {}),
            "total_quality_score": state.get("total_quality_score", 0.0),
            "chat_history": state.get("chat_history", [])
        }
        
        state["results"].append(item)
        print(f"✅ 결과 저장 완료: {len(state['results'])}개")
        
        return state

    def invoke(
            self, 
            user_input_txt: str,
            user_problem: str,
            user_problem_options: List[str],
            vectorstore: Optional[Milvus] = None,
            recursion_limit: int = 1000,
        ) -> Dict:
        
        print(f"🚀 HITL 모드: {self.hitl_mode}")
        
        # 단일 문제 처리
        return self._process_single_problem(
            user_input_txt, user_problem, user_problem_options, vectorstore, recursion_limit
        )
    
    def invoke_batch(
            self,
            problems: List[Dict[str, Any]],  # [{"problem": "...", "options": [...], "input_txt": "..."}]
            vectorstore: Optional[Milvus] = None,
            recursion_limit: int = 1000,
            batch_feedback: bool = True,  # 배치 단위로 피드백 수집
        ) -> Dict[str, Any]:
        """
        여러 문제를 배치로 처리 (HITL 최적화)
        
        Args:
            problems: 문제 리스트
            vectorstore: 벡터스토어
            recursion_limit: 재귀 제한
            batch_feedback: 배치 단위 피드백 여부
        """
        print(f"🚀 배치 처리 시작: {len(problems)}개 문제, HITL 모드: {self.hitl_mode}")
        
        if self.hitl_mode == "auto":
            # 자동 모드: 모든 문제를 자동으로 처리
            return self._process_batch_auto(problems, vectorstore, recursion_limit)
        elif batch_feedback and self.hitl_mode in ["manual", "smart"]:
            # 배치 피드백 모드: 모든 문제 처리 후 한 번에 피드백
            return self._process_batch_with_feedback(problems, vectorstore, recursion_limit)
        else:
            # 개별 HITL 모드: 문제별로 개별 처리
            return self._process_batch_individual(problems, vectorstore, recursion_limit)
    
    def _process_single_problem(
            self,
            user_input_txt: str,
            user_problem: str,
            user_problem_options: List[str],
            vectorstore: Optional[Milvus] = None,
            recursion_limit: int = 1000,
        ) -> Dict:

        # ✅ Milvus 연결 및 벡터스토어 생성
        if vectorstore is None:
            try:
                embedding_model = HuggingFaceEmbeddings(
                    model_name="jhgan/ko-sroberta-multitask",
                    model_kwargs={"device": "cpu"}
                )

                if "default" in connections.list_connections():
                    connections.disconnect("default")
                connections.connect(alias="default", host="localhost", port="19530")

                vectorstore = Milvus(
                    embedding_function=embedding_model,
                    collection_name="problems",
                    connection_args={"host": "localhost", "port": "19530"}
                )
                print("✅ Milvus 벡터스토어 연결 성공")
            except Exception as e:
                print(f"⚠️ Milvus 연결 실패: {e}")
                print("   - 벡터스토어 없이 실행을 계속합니다.")
                vectorstore = None
        
        # vectorstore를 인스턴스 변수로 설정 (직렬화 문제 방지)
        self._current_vectorstore = vectorstore
        
        initial_state: SolutionState = {
            "user_input_txt": user_input_txt,
            "user_problem": user_problem,
            "user_problem_options": user_problem_options,
            "retrieved_docs": [],
            "similar_questions_text": "",
            "generated_answer": "",
            "generated_explanation": "",
            "generated_subject": "",
            "validated": False,
            "retry_count": 0,
            "user_feedback": "",
            "feedback_type": "",
            "search_results": "",
            "improved_explanation": "",
            "interaction_count": 0,
            "max_interactions": self.max_interactions,
            "quality_scores": {},
            "total_quality_score": 0.0,
            "quality_threshold": 80.0,  # 기본 임계값
            "results": [],
            "chat_history": []
        }
        
        final_state = self.graph.invoke(
            initial_state, 
            config={
                "recursion_limit": recursion_limit,
                "configurable": {
                    "thread_id": "default",
                    "checkpoint_id": "default"
                }
            }
        )

        # 결과 확인 및 디버깅
        results = final_state.get("results", [])
        print(f"   - 총 결과 수: {len(results)}")
        
        if results:
            for i, result in enumerate(results):
                print(f"   - 결과 {i+1}: {result.get('user_problem', '')[:30]}...")
                print(f"     상호작용 횟수: {result.get('interaction_count', 0)}")
        else:
            print("   ⚠️ results가 비어있습니다!")
            print(f"   - final_state 내용: {final_state}")
        
        return final_state
    
    def _process_batch_auto(self, problems: List[Dict[str, Any]], vectorstore, recursion_limit) -> Dict[str, Any]:
        """자동 모드: 모든 문제를 자동으로 처리 (HITL 없음)"""
        print("🤖 자동 모드: 모든 문제를 자동으로 처리합니다.")
        
        results = []
        for i, problem in enumerate(problems):
            print(f"\n📝 문제 {i+1}/{len(problems)} 처리 중...")
            
            # HITL을 비활성화하고 자동 처리
            original_mode = self.hitl_mode
            self.hitl_mode = "auto"
            
            try:
                result = self._process_single_problem(
                    problem.get("input_txt", ""),
                    problem.get("problem", ""),
                    problem.get("options", []),
                    vectorstore,
                    recursion_limit
                )
                results.append(result)
            finally:
                self.hitl_mode = original_mode
        
        return {
            "mode": "auto",
            "total_problems": len(problems),
            "processed_problems": len(results),
            "results": results
        }
    
    def _process_batch_with_feedback(self, problems: List[Dict[str, Any]], vectorstore, recursion_limit) -> Dict[str, Any]:
        """배치 피드백 모드: 모든 문제 처리 후 한 번에 피드백 수집"""
        print("💬 배치 피드백 모드: 모든 문제를 처리한 후 피드백을 수집합니다.")
        
        # 1단계: 모든 문제를 자동으로 처리
        batch_results = []
        for i, problem in enumerate(problems):
            print(f"\n📝 문제 {i+1}/{len(problems)} 자동 처리 중...")
            
            # 임시로 자동 모드로 설정
            original_mode = self.hitl_mode
            self.hitl_mode = "auto"
            
            try:
                result = self._process_single_problem(
                    problem.get("input_txt", ""),
                    problem.get("problem", ""),
                    problem.get("options", []),
                    vectorstore,
                    recursion_limit
                )
                batch_results.append({
                    "problem_data": problem,
                    "result": result,
                    "needs_improvement": not result.get("validated", False)
                })
            finally:
                self.hitl_mode = original_mode
        
        # 2단계: 개선이 필요한 문제들만 사용자에게 피드백 요청
        improvement_candidates = [r for r in batch_results if r["needs_improvement"]]
        
        if improvement_candidates:
            print(f"\n🔍 {len(improvement_candidates)}개 문제에 대해 개선이 필요합니다.")
            print("사용자 피드백을 수집하여 문제를 개선하겠습니다.")
            
            # 사용자에게 배치 피드백 요청
            self._collect_batch_feedback(improvement_candidates)
            
            # 피드백을 바탕으로 문제 개선
            for candidate in improvement_candidates:
                self._improve_problem_with_feedback(candidate)
        else:
            print("✅ 모든 문제가 자동으로 처리되었습니다.")
        
        return {
            "mode": "batch_feedback",
            "total_problems": len(problems),
            "auto_processed": len(batch_results) - len(improvement_candidates),
            "improved_with_feedback": len(improvement_candidates),
            "results": batch_results
        }
    
    def _process_batch_individual(self, problems: List[Dict[str, Any]], vectorstore, recursion_limit) -> Dict[str, Any]:
        """개별 HITL 모드: 문제별로 개별 HITL 처리"""
        print("👤 개별 HITL 모드: 각 문제마다 개별적으로 피드백을 수집합니다.")
        
        results = []
        for i, problem in enumerate(problems):
            print(f"\n📝 문제 {i+1}/{len(problems)} HITL 처리 중...")
            
            result = self._process_single_problem(
                problem.get("input_txt", ""),
                problem.get("problem", ""),
                problem.get("options", []),
                vectorstore,
                recursion_limit
            )
            results.append(result)
        
        return {
            "mode": "individual_hitl",
            "total_problems": len(problems),
            "results": results
        }
    
    def _collect_batch_feedback(self, improvement_candidates: List[Dict]) -> None:
        """배치 단위로 사용자 피드백 수집"""
        print(f"\n💬 {len(improvement_candidates)}개 문제에 대한 배치 피드백을 수집합니다.")
        
        # 문제 요약 제공
        for i, candidate in enumerate(improvement_candidates):
            problem = candidate["problem_data"]
            result = candidate["result"]
            print(f"\n문제 {i+1}: {problem.get('problem', '')[:50]}...")
            print(f"현재 풀이: {result.get('generated_explanation', '')[:100]}...")
        
        # 실제 환경에서는 사용자 입력을 받아야 함
        print("\n💭 전체적으로 어떤 부분을 개선하면 좋을지 피드백을 주세요.")
        print("예시: '풀이를 더 쉽게', '용어 설명 추가', '전체적으로 만족' 등")
        
        # 시뮬레이션을 위한 자동 피드백
        batch_feedback = "풀이를 더 쉽게 설명해주고, 중요한 용어에 대한 설명을 추가해주세요."
        print(f"📝 배치 피드백: {batch_feedback}")
        
        # 각 문제에 배치 피드백 적용
        for candidate in improvement_candidates:
            candidate["batch_feedback"] = batch_feedback
    
    def _improve_problem_with_feedback(self, candidate: Dict) -> None:
        """배치 피드백을 바탕으로 개별 문제 개선"""
        problem = candidate["problem_data"]
        result = candidate["result"]
        batch_feedback = candidate.get("batch_feedback", "")
        
        print(f"\n✨ 문제 개선 중: {problem.get('problem', '')[:50]}...")
        
        # LLM을 사용하여 배치 피드백을 바탕으로 풀이 개선
        llm = self._llm(0.3)
        
        improvement_prompt = f"""
        다음 문제의 풀이를 사용자 피드백에 따라 개선해주세요:
        
        문제: {problem.get('problem', '')}
        보기: {problem.get('options', [])}
        현재 풀이: {result.get('generated_explanation', '')}
        
        사용자 피드백: {batch_feedback}
        
        위 피드백을 반영하여 풀이를 개선해주세요.
        
        출력 형식:
        개선된 풀이: [개선된 풀이 내용]
        개선 사항: [어떤 부분을 어떻게 개선했는지 설명]
        """
        
        try:
            response = llm.invoke(improvement_prompt)
            improved_explanation = response.content.strip()
            
            # 개선된 풀이 파싱
            explanation_match = re.search(r"개선된 풀이:\s*(.+)", improved_explanation, re.DOTALL)
            if explanation_match:
                candidate["improved_explanation"] = explanation_match.group(1).strip()
                print("✅ 문제 개선 완료")
            else:
                candidate["improved_explanation"] = result.get("generated_explanation", "")
                print("⚠️ 풀이 개선 실패, 원본 유지")
                
        except Exception as e:
            print(f"⚠️ 풀이 개선 중 오류: {e}")
            candidate["improved_explanation"] = result.get("generated_explanation", "")

    def _create_custom_checkpointer(self):
        """vectorstore 직렬화 문제를 해결하는 커스텀 체크포인터"""
        from langgraph.checkpoint.memory import InMemorySaver
        
        # 기본 InMemorySaver 사용 (안정적이고 호환성 좋음)
        return InMemorySaver()


if __name__ == "__main__":
    print("🚀 HITL 피드백 시스템 테스트")
    print("=" * 50)
    
    # HITL 모드 선택
    print("\n🎯 HITL 모드를 선택하세요:")
    print("1. auto - 자동 모드 (HITL 없음)")
    print("2. smart - 스마트 모드 (품질에 따라 자동 결정)")
    print("3. manual - 수동 모드 (항상 HITL)")
    
    mode_choice = input("모드를 선택하세요 (1-3, 기본값: 3): ").strip()
    
    if mode_choice == "1":
        hitl_mode = "auto"
    elif mode_choice == "2":
        hitl_mode = "smart"
    else:
        hitl_mode = "manual"
    
    print(f"\n✅ 선택된 모드: {hitl_mode}")
    
    # 벡터스토어 연결 시도 (선택사항)
    vectorstore = None
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={"device": "cpu"}
        )

        if "default" in connections.list_connections():
            connections.disconnect("default")
        connections.connect(alias="default", host="localhost", port="19530")

        vectorstore = Milvus(
            embedding_function=embedding_model,
            collection_name="problems",
            connection_args={"host": "localhost", "port":"19530"}
        )
        print("✅ Milvus 벡터스토어 연결 성공")
    except Exception as e:
        print(f"⚠️ Milvus 연결 실패: {e}")
        print("   - 벡터스토어 없이 테스트를 진행합니다.")
        vectorstore = None
    
    agent = SolutionAgent(max_interactions=5, hitl_mode=hitl_mode)

    # 테스트용 기본값 제공
    print(f"\n📝 테스트 문제를 입력하거나 기본값을 사용하세요:")
    
    user_input_txt = input("❓ 사용자 질문 (기본값: 프로세스와 스레드의 차이점을 이해하고 싶습니다): ").strip()
    if not user_input_txt:
        user_input_txt = "프로세스와 스레드의 차이점을 이해하고 싶습니다"
    
    user_problem = input("❓ 사용자 문제 (기본값: 프로세스와 스레드의 차이점으로 올바른 것은?): ").strip()
    if not user_problem:
        user_problem = "프로세스와 스레드의 차이점으로 올바른 것은?"
    
    user_problem_options_raw = input("❓ 사용자 보기 (쉼표로 구분, 기본값: 프로세스는 독립적인 메모리 공간을 가지며 스레드는 프로세스 내에서 메모리를 공유한다,프로세스와 스레드는 모두 독립적인 메모리 공간을 가진다,프로세스는 메모리를 공유하고 스레드는 독립적인 메모리 공간을 가진다,프로세스와 스레드는 모두 메모리를 공유한다): ").strip()
    if not user_problem_options_raw:
        user_problem_options_raw = "프로세스는 독립적인 메모리 공간을 가지며 스레드는 프로세스 내에서 메모리를 공유한다,프로세스와 스레드는 모두 독립적인 메모리 공간을 가진다,프로세스는 메모리를 공유하고 스레드는 독립적인 메모리 공간을 가진다,프로세스와 스레드는 모두 메모리를 공유한다"
    
    user_problem_options = [opt.strip() for opt in user_problem_options_raw.split(",") if opt.strip()]

    print(f"\n🚀 에이전트 실행 중...")
    print(f"모드: {hitl_mode}")
    print(f"문제: {user_problem}")
    print(f"보기 수: {len(user_problem_options)}개")
    
    try:
        final_state = agent.invoke(
            user_input_txt=user_input_txt,
            user_problem=user_problem,
            user_problem_options=user_problem_options,
            vectorstore=vectorstore,
        )

        print(f"\n" + "=" * 50)
        print(f"🎯 최종 결과:")
        print(f"문제: {final_state.get('user_problem', '')}")
        print(f"정답: {final_state.get('generated_answer', '')}")
        print(f"풀이: {final_state.get('generated_explanation', '')}")
        print(f"과목: {final_state.get('generated_subject', '')}")
        print(f"상호작용 횟수: {final_state.get('interaction_count', 0)}")
        print(f"사용자 피드백: {final_state.get('user_feedback', '')}")
        print(f"피드백 유형: {final_state.get('feedback_type', '')}")
        
        # 품질 점수 출력
        quality_scores = final_state.get('quality_scores', {})
        if quality_scores:
            print(f"\n📊 품질 점수:")
            for key, score in quality_scores.items():
                print(f"  {key}: {score:.1f}/100")
            print(f"  총점: {final_state.get('total_quality_score', 0):.1f}/100")
        
        # 채팅 히스토리 출력
        chat_history = final_state.get('chat_history', [])
        if chat_history:
            print(f"\n💬 상호작용 히스토리:")
            for i, chat in enumerate(chat_history, 1):
                print(f"  {i}. {chat[:100]}...")
                
    except Exception as e:
        print(f"❌ 에이전트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 테스트 완료!")
    print(f"\n💡 추가 테스트를 원하시면 'python test_hitl_feedback.py'를 실행하세요.")