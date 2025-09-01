#!/usr/bin/env python3
"""
메인 오케스트레이터
사용자 입력을 분석하여 farmer 또는 teacher 서비스를 선택하고 실행하는 메인 오케스트레이터
"""

import os
import sys
import time
import json
from typing import Dict, Any, Optional, List, TypedDict, NotRequired
from typing_extensions import Annotated
from dotenv import load_dotenv
from openai import OpenAI

from langgraph.graph import StateGraph, END, START
# from langgraph.graph.message import Message  # 사용하지 않는 import 제거
# from langgraph.prebuilt import MemorySaver  # 사용하지 않는 import 제거
from langgraph.checkpoint.memory import MemorySaver as LangGraphMemorySaver
from langgraph.types import interrupt

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from teacher.teacher import Teacher, TeacherState
from common.short_term.redis_memory import RedisLangGraphMemory

# 환경 변수 로드
load_dotenv()


# LLM 설정
llm_api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
llm_base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")


# Redis 설정
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6380"))

# OpenAI 클라이언트 초기화
client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)

class SimpleLLM:
    def __init__(self, client: OpenAI, model: str, temperature: float = 0.2):
        self._client = client
        self._model = model
        self._temperature = temperature
    
    def invoke(self, prompt: str):
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=32
        )
        class _R:
            pass
        r = _R()
        r.content = resp.choices[0].message.content.strip()
        return r

class MainState(TypedDict):

    """메인 오케스트레이터 상태"""
    user_query: str                    # 사용자 입력
    user_id: str                       # 사용자 ID
    chat_id: str                       # 채팅 ID
    service_classification: str        # 서비스 분류 결과 (farmer/teacher)
    classification_confidence: float   # 분류 신뢰도
    classification_reason: str         # 분류 이유
    llm_response: str                  # LLM 응답
    error_message: NotRequired[str]    # 오류 메시지
    hitl_required: bool                # HITL 필요 여부
    hitl_data: NotRequired[Dict]       # HITL 데이터
    teacher_state: NotRequired[TeacherState]  # Teacher 상태
    farmer_state: NotRequired[Dict]    # Farmer 상태
    final_response: str                # 최종 응답
    session_data: Dict[str, Any]       # 세션 데이터

class MainOrchestrator:
    """메인 오케스트레이터 클래스"""
    
    def __init__(self, user_id: str = "default_user", chat_id: str = "default_chat"):
        """
        메인 오케스트레이터 초기화
        
        Args:
            user_id: 사용자 ID
            chat_id: 채팅 ID
        """
        self.user_id = user_id
        self.chat_id = chat_id
        
        # 메모리 초기화
        try:
            self.memory = RedisLangGraphMemory(
                user_id=user_id,
                service="main_orchestrator",
                chat_id=chat_id,
                redis_host="localhost",
                redis_port=6380
            )
        except Exception as e:
            print(f"⚠️ Redis 연결 실패: {e}")
            print("📝 메모리 기반으로 실행합니다.")
            self.memory = LangGraphMemorySaver()
        
        # 에이전트 초기화
        self.teacher = Teacher(
            user_id=user_id,
            service="teacher",
            chat_id=chat_id,
            init_agents=True
        )

        # LLM for classification
        self.llm = SimpleLLM(client=client, model=llm_model, temperature=llm_temperature)
        
        # 그래프 생성 및 컴파일
        self.graph = self._create_graph()

        self.app = self.graph.compile(checkpointer=self.memory)
    
    def classify_service(self, state: MainState) -> MainState:
        """
        사용자 입력을 분석하여 서비스를 분류하는 노드 (키워드 기반 휴리스틱)
        """
        user_query = state["user_query"]
        
        try:
            # LLM을 사용한 서비스 분류
            classification_prompt = f"""
사용자의 질문을 분석하여 어떤 서비스가 필요한지 분류해주세요.

질문: {user_query}

다음 중 하나로만 답변해주세요:
- "farmer": 농업, 재배, 작물, 농사 관련 질문
- "teacher": 자격증, 시험, 학습, 교육 관련 질문

답변:
"""
            
            response = self.llm.invoke(classification_prompt)
            service_classification = response.content.strip().lower()
            
            # 응답 정규화
            if "farmer" in service_classification:
                service_classification = "farmer"
            elif "teacher" in service_classification:
                service_classification = "teacher"
            else:
                # 기본값
                service_classification = "teacher"
            
            print(f"🤖 LLM 기반 분류: {service_classification}")
            
            return {
                **state,
                "service_classification": service_classification,
                "classification_confidence": 1.0,  # 단순화를 위해 고정값
                "classification_reason": "LLM 분류 결과"
            }
            
        except Exception as e:
            print(f"❌ 서비스 분류 중 오류: {e}")
            return {
                **state,
                "service_classification": "teacher",  # 기본값
                "classification_confidence": 0.0,
                "classification_reason": f"오류로 인한 기본값: {str(e)}"
            }
    
    def should_use_hitl(self, state: MainState) -> str:
        """
        HITL 사용 여부를 결정하는 조건부 엣지
        
        Args:
            state: 현재 상태
            
        Returns:
            str: 다음 노드 이름
        """
        service = state.get("service_classification", "teacher")
        
        # 단순한 분류 결과에 따라 라우팅
        if service == "farmer":
            return "execute_farmer"
        elif service == "teacher":
            return "execute_teacher"
        else:
            # 기본값
            return "execute_teacher"
    
    def hitl_confirmation(self, state: MainState) -> MainState:
        """
        HITL 확인 노드
        
        Args:
            state: 현재 상태
            
        Returns:
            MainState: HITL 데이터가 포함된 상태
        """
        service = state["service_classification"]
        reason = state["classification_reason"]
        confidence = state["classification_confidence"]
        
        hitl_message = f"""
분류 결과를 확인해주세요:

사용자 질문: {state['user_query']}
분류된 서비스: {service}
신뢰도: {confidence:.2f}
분류 이유: {reason}

이 분류가 맞나요? (Y/N)
"""
        
        return {
            **state,
            "hitl_required": True,
            "hitl_data": {
                "message": hitl_message,
                "current_service": service,
                "user_query": state["user_query"]
            }
        }
    
    def execute_service(self, state: MainState) -> MainState:
        """
        선택된 서비스를 실행하는 노드
        
        Args:
            state: 현재 상태
            
        Returns:
            MainState: 실행 결과가 포함된 상태
        """
        service = state["service_classification"]
        user_query = state["user_query"]
        
        try:
            if service == "teacher":
                # Teacher 서비스 실행
                teacher_state = {
                    "user_query": user_query,
                    "intent": "",
                    "shared": {},
                    "work": {},
                    "retrieval": {},
                    "generation": {},
                    "solution": {},
                    "score": {},
                    "analysis": {},
                    "history": [],
                    "session": {},
                    "artifacts": {},
                    "routing": {},
                    "llm_response": ""
                }
                
                result = self.teacher.execute(teacher_state)
                
                return {
                    **state,
                    "teacher_state": result,
                    "final_response": result.get("llm_response", "Teacher 서비스 실행 완료"),
                    "hitl_required": False
                }
                
            elif service == "farmer":
                # Farmer 서비스 실행 (기본 응답)
                farmer_response = f"Farmer 서비스: {user_query}에 대한 농업 관련 답변을 제공합니다."
                
                return {
                    **state,
                    "farmer_state": {"response": farmer_response},
                    "final_response": farmer_response,
                    "hitl_required": False
                }
            
            else:
                return {
                    **state,
                    "error_message": f"알 수 없는 서비스: {service}",
                    "final_response": "서비스를 찾을 수 없습니다.",
                    "hitl_required": False
                }
                
        except Exception as e:
            print(f"❌ 서비스 실행 중 오류: {e}")
            return {
                **state,
                "error_message": str(e),
                "final_response": f"서비스 실행 중 오류가 발생했습니다: {str(e)}",
                "hitl_required": False
            }
    
    def execute_teacher(self, state: MainState) -> MainState:
        """
        Teacher 서비스를 실행하는 노드
        
        Args:
            state: 현재 상태
            
        Returns:
            MainState: Teacher 실행 결과가 포함된 상태
        """
        user_query = state["user_query"]
        
        try:
            # Teacher 서비스 실행을 위한 상태 준비
            teacher_state = {
                "user_query": user_query,
                "intent": state.get("intent", ""),
                "shared": state.get("shared", {}),
                "work": state.get("work", {}),
                "retrieval": state.get("retrieval", {}),
                "generation": state.get("generation", {}),
                "solution": state.get("solution", {}),
                "score": state.get("score", {}),
                "analysis": state.get("analysis", {}),
                "history": state.get("history", []),
                "session": state.get("session", {}),
                "artifacts": state.get("artifacts", {}),
                "routing": state.get("routing", {})
            }
            
            # Teacher 실행 (invoke 사용)
            result = self.teacher.graph.invoke(teacher_state)
            
            # 결과를 메인 상태에 병합
            return {
                **state,
                "intent": result.get("intent", ""),
                "shared": result.get("shared", {}),
                "work": result.get("work", {}),
                "retrieval": result.get("retrieval", {}),
                "generation": result.get("generation", {}),
                "solution": result.get("solution", {}),
                "score": result.get("score", {}),
                "analysis": result.get("analysis", {}),
                "history": result.get("history", []),
                "session": result.get("session", {}),
                "artifacts": result.get("artifacts", {}),
                "routing": result.get("routing", {}),
                "final_response": result.get("llm_response", "Teacher 서비스 실행 완료"),
                "hitl_required": False
            }
                
        except Exception as e:
            print(f"❌ Teacher 서비스 실행 중 오류: {e}")
            return {
                **state,
                "error_message": str(e),
                "final_response": f"Teacher 서비스 실행 중 오류가 발생했습니다: {str(e)}",
                "hitl_required": False
            }
    
    def execute_farmer(self, state: MainState) -> MainState:
        """
        Farmer 서비스를 실행하는 노드
        
        Args:
            state: 현재 상태
            
        Returns:
            MainState: Farmer 실행 결과가 포함된 상태
        """
        user_query = state["user_query"]
        
        try:
            # Farmer 서비스 실행 (기본 응답)
            farmer_response = f"Farmer 서비스: {user_query}에 대한 농업 관련 답변을 제공합니다."
            
            return {
                **state,
                "farmer_state": {"response": farmer_response},
                "final_response": farmer_response,
                "hitl_required": False
            }
                
        except Exception as e:
            print(f"❌ Farmer 서비스 실행 중 오류: {e}")
            return {
                **state,
                "error_message": str(e),
                "final_response": f"Farmer 서비스 실행 중 오류가 발생했습니다: {str(e)}",
                "hitl_required": False
            }
    
    def process_hitl_response(self, state: MainState, hitl_response: str) -> MainState:
        """
        HITL 응답을 처리하는 노드
        
        Args:
            state: 현재 상태
            hitl_response: HITL 응답
            
        Returns:
            MainState: 처리된 상태
        """
        response = hitl_response.strip().upper()
        
        if response in ["Y", "YES", "맞음", "맞습니다"]:
            # 분류가 맞다고 확인됨
            return {
                **state,
                "hitl_required": False,
                "classification_confidence": 1.0
            }
        elif response in ["N", "NO", "아니요", "틀림"]:
            # 분류가 틀렸다고 확인됨 - 반대 서비스로 변경
            current_service = state["service_classification"]
            new_service = "teacher" if current_service == "farmer" else "farmer"
            
            return {
                **state,
                "service_classification": new_service,
                "classification_confidence": 1.0,
                "classification_reason": "HITL을 통한 수동 수정",
                "hitl_required": False
            }
        else:
            # 잘못된 응답 - 기본값 사용
            return {
                **state,
                "hitl_required": False,
                "classification_confidence": 0.5
            }
    
    def finalize_response(self, state: MainState) -> MainState:
        """
        최종 응답을 정리하는 노드
        
        Args:
            state: 현재 상태
            
        Returns:
            MainState: 정리된 상태
        """
        # 세션 데이터 업데이트
        session_data = state.get("session_data", {})
        session_data.update({
            "last_query": state["user_query"],
            "last_service": state["service_classification"],
            "last_response": state["final_response"],
            "timestamp": str(int(time.time()))
        })
        
        return {
            **state,
            "session_data": session_data
        }
    
    def _create_graph(self) -> StateGraph:
        """
        LangGraph 워크플로우 생성
        
        Returns:
            StateGraph: 생성된 그래프
        """
        # 그래프 빌더 생성
        builder = StateGraph(MainState)


        # 노드 추가
        builder.add_node("classify_service", self.classify_service)
        builder.add_node("hitl_confirmation", self.hitl_confirmation)
        builder.add_node("execute_teacher", self.execute_teacher)
        builder.add_node("execute_farmer", self.execute_farmer)
        builder.add_node("finalize_response", self.finalize_response)
        
        # 엣지 추가
        builder.add_edge(START, "classify_service")
        builder.add_conditional_edges(
            "classify_service",
            self.should_use_hitl,
            {
                "hitl_confirmation": "hitl_confirmation",
                "execute_teacher": "execute_teacher",
                "execute_farmer": "execute_farmer"
            }
        )
        builder.add_edge("hitl_confirmation", "execute_teacher")  # HITL 후 기본적으로 Teacher로
        builder.add_edge("execute_teacher", "finalize_response")
        builder.add_edge("execute_farmer", "finalize_response")
        builder.add_edge("finalize_response", END)
        
        return builder
    
    def run(self, user_query: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        오케스트레이터 실행
        
        Args:
            user_query: 사용자 질문
            config: 실행 설정
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        # 초기 상태 구성
        initial_state = {
            "user_query": user_query,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "service_classification": "",
            "classification_confidence": 0.0,
            "classification_reason": "",
            "llm_response": "",
            "hitl_required": False,
            "final_response": "",
            "session_data": {}
        }
        
        # checkpointer 설정
        if config is None:
            config = {}
        config.update({
            "thread_id": f"{self.user_id}_{self.chat_id}",
            "checkpoint_id": f"main_orchestrator_{int(time.time())}"
        })
        
        try:
            # 그래프 실행
            result = self.app.invoke(initial_state, config=config)
            return dict(result)
            
        except Exception as e:
            print(f"❌ 오케스트레이터 실행 중 오류: {e}")
            return {
                "error": str(e),
                "user_query": user_query,
                "final_response": "오케스트레이터 실행 중 오류가 발생했습니다."
            }
    
    def resume_workflow(self, hitl_response: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        HITL 응답으로 워크플로우 재개
        
        Args:
            hitl_response: HITL 응답
            config: 실행 설정
            
        Returns:
            Dict[str, Any]: 재개 결과
        """
        try:
            result = self.app.invoke(
                {"hitl_response": hitl_response},
                config=config
            )
            return dict(result)
            
        except Exception as e:
            print(f"❌ 워크플로우 재개 중 오류: {e}")
            return {
                "error": str(e),
                "hitl_response": hitl_response
            }

def main():

    """메인 실행 함수"""
    print("🚀 ET-Agent 메인 오케스트레이터 시작")
    
    # 오케스트레이터 생성
    orchestrator = MainOrchestrator(
        user_id="demo_user",
        chat_id="demo_chat"
    )
    
    # 예제 실행
    test_queries = [
        "토마토 재배 방법을 알려줘",
        "소프트웨어 설계 문제 3개 만들어줘",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n=== 테스트 {i}: {query} ===")
        
        result = orchestrator.run(query)
        
        if "error" in result:
            print(f"❌ 오류: {result['error']}")
        else:
            service = result.get("service_classification", "N/A")
            confidence = result.get("classification_confidence", 0.0)
            response = result.get("final_response", "응답 없음")
            
            print(f"분류된 서비스: {service}")
            print(f"신뢰도: {confidence:.2f}")
            print(f"응답: {response[:100]}...")

if __name__ == "__main__":
    main()


