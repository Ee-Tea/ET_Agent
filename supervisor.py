"""
LangGraph 기반 Main Orchestrator
이미지 워크플로우에 따른 체계적인 Supervisor 구현
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Optional, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from common.short_term.redis_memory import RedisLangGraphMemory
from teacher.teacher import Teacher
from farmer.farmer import Farmer


class MainState(TypedDict):
    """라우터 상태 정의"""
    user_query: str
    user_id: str
    chat_id: str
    session_key: str
    
    # 메모리 데이터
    existing_questions: List[Dict]
    locked_service: Optional[str]
    
    # 분류 결과
    is_relevant: bool
    classified_service: str
    service_consistent: bool
    
    # 실행 결과
    teacher_result: Optional[Dict]
    farmer_result: Optional[Dict]
    final_response: str


class MainOrchestrator:
    """메인 오케스트레이터 - 이미지 워크플로우 기반"""
    
    def __init__(self, user_id: str, chat_id: str):
        self.user_id = user_id
        self.chat_id = chat_id
        self.session_key = f"{user_id}_{chat_id}"
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY=REDACTED = RedisLangGraphMemory(
            user_id=user_id,
            service="supervisor",
            chat_id=chat_id
        )
        
        # 서비스 초기화
        self.teacher = Teacher(user_id, "teacher", chat_id)
        self.farmer = Farmer()
        
        print(f"✅ MainOrchestrator 초기화 완료 (session: {self.session_key})")
    
    def load_memory_data(self, state: MainState) -> MainState:
        """1. 메모리 데이터 로드"""
        print("📚 메모리 데이터 로드 중...")
        
        try:
            # 기존 문제들 로드
            existing_questions = self.get_questions_from_redis()
            
            # 락된 서비스 확인
            locked_service = self.get_locked_service()
            
            state["existing_questions"] = existing_questions
            state["locked_service"] = locked_service
            
            print(f"📚 {len(existing_questions)}개 기존 문제 로드")
            print(f"🔒 락된 서비스: {locked_service}")
            
        except Exception as e:
            print(f"❌ 메모리 로드 실패: {e}")
            state["existing_questions"] = []
            state["locked_service"] = None
        
        return state
    
    def check_question_relevance(self, state: MainState) -> MainState:
        """2. 질문 관련성 검사"""
        print("🔍 질문 관련성 검사 중...")
        
        user_query = state["user_query"].lower()
        
        # 관련 키워드 체크
        relevant_keywords = [
            # Teacher 관련
            "문제", "시험", "학습", "공부", "정답", "채점", "점수", "용어", "설명",
            "데이터베이스", "소프트웨어", "프로그래밍", "정보시스템",
            # Farmer 관련  
            "재배", "농작물", "작물", "토마토", "고추", "배추", "상추", "시설", "온실",
            "재해", "병해", "해충", "날씨", "기상", "수확", "시장", "가격"
        ]
        
        is_relevant = any(keyword in user_query for keyword in relevant_keywords)
        state["is_relevant"] = is_relevant
        
        print(f"🔍 질문 관련성: {'✅ 관련' if is_relevant else '❌ 무관'}")
        return state
    
    def classify_service(self, state: MainState) -> MainState:
        """3. 서비스 분류"""
        print("🤖 서비스 분류 중...")
        
        user_query = state["user_query"]
        
        prompt = f"""
        다음 사용자 질문을 분석하여 적절한 서비스를 분류해주세요.

        **Teacher 서비스 키워드:**
        - 문제 생성: "문제", "시험", "문제 만들어줘", "3문제", "5문제"
        - 답변 입력: "정답은", "답은", "1,2,3", "1, 2, 3"
        - 채점 요청: "채점", "점수", "맞았나", "정답 확인"
        - 학습 도움: "학습", "공부", "설명", "용어", "개념"
        - 과목 관련: "데이터베이스", "소프트웨어", "프로그래밍", "정보시스템"

        **Farmer 서비스 키워드:**
        - 작물 재배: "재배", "농작물", "작물", "토마토", "고추", "배추", "상추"
        - 시설 관리: "시설", "온실", "하우스", "관수", "환기"
        - 재해 대응: "재해", "병해", "해충", "방제", "예방"
        - 기상 정보: "날씨", "기상", "온도", "습도", "강수"
        - 수확/시장: "수확", "시장", "가격", "판매", "유통"

        질문: "{user_query}"

        위 키워드를 참고하여 다음 중 하나로 분류해주세요:
        - teacher: 교육/학습 관련 질문
        - farmer: 농업/재배 관련 질문
        - irrelevant: 위 두 분야와 무관한 질문

        분류 결과만 출력하세요 (teacher/farmer/irrelevant):
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            classified_service = response.content.strip().lower()
            
            # 유효성 검사
            if classified_service not in ["teacher", "farmer", "irrelevant"]:
                classified_service = "irrelevant"
            
            state["classified_service"] = classified_service
            print(f"🤖 분류 결과: {classified_service}")
            
        except Exception as e:
            print(f"❌ 서비스 분류 실패: {e}")
            state["classified_service"] = "irrelevant"
        
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
            # Farmer 그래프 실행
            result = self.farmer.invoke({
                "query": state["user_query"]
            })
            
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
            # Teacher 상태 준비
            teacher_state = {
                "user_query": state["user_query"],
                "intent": self._determine_teacher_intent(state["user_query"]),
                "shared": {
                    "question": [q.get("question", "") for q in state["existing_questions"]],
                    "options": [q.get("options", []) for q in state["existing_questions"]],
                    "answer": [q.get("answer", "") for q in state["existing_questions"]],
                    "explanation": [q.get("explanation", "") for q in state["existing_questions"]],
                    "subject": [q.get("subject", "") for q in state["existing_questions"]]
                },
                "routing": {
                    "output_mode": "pdf"  # 비대화형 모드
                }
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
        
        # Farmer 결과가 있으면 저장 (필요시)
        if state["farmer_result"] and "error" not in state["farmer_result"]:
            # Farmer 결과 저장 로직 (필요시 구현)
            pass
        
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
        """메인 쿼리 처리 - 워크플로우 실행"""
        print(f"\n📝 사용자 질문: {user_query}")
        
        # 특별 명령어 처리
        if user_query.lower() == "clear":
            self.clear_session()
            return "🧹 세션이 초기화되었습니다."
        
        # 초기 상태 설정
        state = MainState(
            user_query=user_query,
            user_id=self.user_id,
            chat_id=self.chat_id,
            session_key=self.session_key,
            existing_questions=[],
            locked_service=None,
            is_relevant=False,
            classified_service="",
            service_consistent=False,
            teacher_result=None,
            farmer_result=None,
            final_response=""
        )
        
        # 워크플로우 실행
        try:
            # 1. 메모리 데이터 로드
            state = self.load_memory_data(state)
            
            # 2. 질문 관련성 검사
            state = self.check_question_relevance(state)
            
            if not state["is_relevant"]:
                state = self.handle_irrelevant_question(state)
                state = self.save_memory_data(state)
                return state["final_response"]
            
            # 3. 서비스 분류
            state = self.classify_service(state)
            
            # 4. 서비스 일관성 검사
            state = self.check_final_service_consistency(state)
            
            if not state["service_consistent"]:
                state = self.handle_service_inconsistency(state)
                state = self.save_memory_data(state)
                return state["final_response"]
            
            # 5. 세션 데이터 업데이트
            state = self.update_session_data(state)
            
            # 6-7. 서비스 실행
            if state["classified_service"] == "teacher":
                state = self.teacher_app(state)
                state = self.merge_teacher_result(state)
            elif state["classified_service"] == "farmer":
                state = self.execute_farmer(state)
            
            # 8. 메모리 데이터 저장
            state = self.save_memory_data(state)
            
            # 9. 최종 응답 생성
            state = self.finalize_response(state)
            
            return state["final_response"]
            
        except Exception as e:
            print(f"❌ 워크플로우 실행 실패: {e}")
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
        
        llm_response = teacher_result.get("llm_response", "")
        if llm_response:
            return llm_response
        
        # 문제 생성 결과가 있는 경우
        shared_data = teacher_result.get("shared", {})
        questions = shared_data.get("question", [])
        
        if questions:
            response = "✅ 문제가 생성되었습니다!\n\n"
            for i, question in enumerate(questions):
                if question:
                    response += f"문제 {i+1}: {question[:100]}...\n"
            response += "\n답변을 입력하려면 '정답은 1,2,3' 형식으로 입력하세요."
            return response
        
        return "✅ Teacher 서비스가 실행되었습니다."
    
    def _format_farmer_response(self, farmer_result: Dict) -> str:
        """Farmer 응답 포맷팅"""
        if "error" in farmer_result:
            return f"❌ Farmer 실행 오류: {farmer_result['error']}"
        
        output = farmer_result.get("output", "")
        if output:
            return output
        
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
                        if isinstance(k, bytes):
                            k = k.decode('utf-8')
                        if isinstance(v, bytes):
                            v = v.decode('utf-8')
                        
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