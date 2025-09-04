#!/usr/bin/env python3
#uv run main.py
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
import graphviz
import uuid
from typing import Optional
from langgraph.checkpoint.memory import MemorySaver as LGMemorySaver

from langgraph.graph import StateGraph, END, START
# from langgraph.graph.message import Message  # 사용하지 않는 import 제거
# from langgraph.prebuilt import MemorySaver  # 사용하지 않는 import 제거
from langgraph.types import interrupt

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from teacher.teacher_graph import Teacher, TeacherState
from common.short_term.redis_memory import RedisLangGraphMemory

# 환경 변수 로드
load_dotenv()


# LLM 설정
llm_api_key = os.getenv("OPENAI_API_KEY=REDACTED = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
llm_base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")


# Redis 설정
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6380"))

***REMOVED*** 클라이언트 초기화
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
    service_classification: str        # 서비스 분류 결과 (farmer/teacher/irrelevant)
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
    # 새로운 필드들
    is_relevant: NotRequired[bool]     # 질문 관련성 (농사 자격증 관련인지)
    relevance_check_reason: NotRequired[str]  # 관련성 체크 이유
    service_consistent: NotRequired[bool]     # 서비스 일관성
    is_first_question: NotRequired[bool]      # 첫 번째 질문인지
    consistency_message: NotRequired[str]     # 일관성 체크 메시지
    previous_service: NotRequired[str]        # 이전 서비스
    # 메모리 관련 필드들
    chat_history: NotRequired[List[Dict]]     # 채팅 히스토리
    loaded_memory_data: NotRequired[Dict]     # 로드된 메모리 데이터
    memory_loaded: NotRequired[bool]          # 메모리 로드 성공 여부
    memory_saved: NotRequired[bool]           # 메모리 저장 성공 여부
    # 재개 플래그: 재시작 시 분류/라우팅을 건너뛰고 teacher로 직행
    resume_to_teacher: NotRequired[bool]

class MainOrchestrator:
    """메인 오케스트레이터 클래스"""
    
    def __init__(
        self,
        user_id: str = "cli_user",
        chat_id: Optional[str] = None,
        *,
        redis_host: str = "localhost",
        redis_port: int = 6380,
    ):
        # 1) 식별자(재개에 절대적으로 중요)
        self.user_id = user_id
        self.chat_id = chat_id or "cli_chat"  # 백엔드에서 주면 그걸 쓰고, 아니면 고정/uuid
        self.thread_id = f"teacher:{self.user_id}:{self.chat_id}"
        self.checkpoint_id = "main_orchestrator"

        # 2) 숏텀 메모리(대화/세션) - 기존 Redis 사용 (체크포인터와 역할 다름)
        try:
            self.memory = RedisLangGraphMemory(
                user_id=self.user_id,
                service="main_orchestrator",
                chat_id=self.chat_id,
                redis_host=redis_host,
                redis_port=redis_port,
            )
        except Exception as e:
            print(f"⚠️ Redis 연결 실패: {e}")
            self.memory = None

        # 3) LangGraph 체크포인터(중단/재개 스냅샷용)
        self.checkpointer = LGMemorySaver()

        # 4) Teacher 서브그래프 초기화
        self.teacher = Teacher(
            user_id=self.user_id,
            service="teacher",
            chat_id=self.chat_id,
            init_agents=True
        )

        # 5) 그래프 컴파일도 동일 체크포인터
        self.graph = self._create_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
        self.llm = SimpleLLM(client=client, model=llm_model, temperature=llm_temperature)

        # 6) 기본 config (run/resume 공통 사용 권장)
        self.base_config = {
            "configurable": {
                "thread_id": self.thread_id,
            },
            "interrupt_after": [
                "teacher_app.await_output_mode",
                "teacher_app.await_form_answers",
            ],
        }
    
    def load_memory_data(self, state: MainState) -> MainState:
        """
        숏텀 메모리에서 데이터를 로드하는 노드
        """
        # 재개 시 최소 상태로 들어오는 경우를 대비하여 기본 필드 보정
        new_state = {**state}
        user_id = new_state.get("user_id", self.user_id)
        chat_id = new_state.get("chat_id", self.chat_id)
        new_state["user_id"] = user_id
        new_state["chat_id"] = chat_id
        if not new_state.get("user_query"):
            if getattr(self, "_last_user_query", ""):
                new_state["user_query"] = self._last_user_query
        
        try:
            print(f"📥 메모리 데이터 로드 중... user_id={user_id}, chat_id={chat_id}")
            
            # 채팅 히스토리 로드
            chat_history = self.memory.get_chat_history()
            
            # 세션 데이터 로드
            session_key = f"session_{user_id}_{chat_id}"
            session_data = self.memory.get(session_key) or {}
            
            # 메모리에서 기존 상태 데이터 로드 (있는 경우)
            memory_keys = self.memory.keys(f"{user_id}_{chat_id}_*")
            loaded_data = {}
            
            for key in memory_keys:
                if "session" not in key:  # 세션 데이터는 별도로 처리
                    data = self.memory.get(key)
                    if data:
                        # 키에서 데이터 타입 추출 (예: user_chat_shared, user_chat_generation 등)
                        key_parts = key.split('_')
                        if len(key_parts) >= 3:
                            data_type = '_'.join(key_parts[2:])
                            loaded_data[data_type] = data
            
            print(f"✅ 메모리 로드 완료: 히스토리={len(chat_history)}개, 세션={bool(session_data)}, 데이터={len(loaded_data)}개")
            
            # 최신 질의 캐시
            if new_state.get("user_query"):
                self._last_user_query = new_state["user_query"]

            return {
                **new_state,
                "session_data": session_data,
                "chat_history": chat_history,
                "loaded_memory_data": loaded_data,
                "memory_loaded": True
            }
            
        except Exception as e:
            print(f"❌ 메모리 로드 중 오류: {e}")
            return {
                **new_state,
                "session_data": {},
                "chat_history": [],
                "loaded_memory_data": {},
                "memory_loaded": False,
                "error_message": f"메모리 로드 오류: {str(e)}"
            }
    
    def save_memory_data(self, state: MainState) -> MainState:
        """
        상태를 숏텀 메모리에 저장하는 노드
        """
        user_id = state.get("user_id", self.user_id)
        chat_id = state.get("chat_id", self.chat_id)
        
        try:
            print(f"💾 메모리 데이터 저장 중... user_id={user_id}, chat_id={chat_id}")
            teacher_state = state.get("teacher_state")
            # teacher_state가 없으면 top-level에서 추출 (서브그래프를 raw로 등록한 경우 대비)
            if not teacher_state:
                candidate_keys = ("shared", "artifacts", "final_response", "routing", "score")
                teacher_state = {k: state.get(k) for k in candidate_keys if k in state}

            if teacher_state:
                shared = teacher_state.get("shared", {})
                artifacts = teacher_state.get("artifacts", {})
            
            # 현재 대화를 채팅 히스토리에 추가
            current_interaction = {
                "timestamp": time.time(),
                "user_query": state.get("user_query", self._last_user_query),
                "service_classification": state.get("service_classification", ""),
                "final_response": state.get("final_response", ""),
                "is_relevant": state.get("is_relevant", True),
                "service_consistent": state.get("service_consistent", True)
            }
            
            self.memory.add_to_chat_history(current_interaction)
            
            # 세션 데이터 저장
            session_data = state.get("session_data", {})
            if session_data:
                session_key = f"session_{user_id}_{chat_id}"
                self.memory.put(session_key, session_data)
            
            # Teacher 상태가 있는 경우 저장
            teacher_state = state.get("teacher_state")
            if teacher_state:
                # Teacher의 주요 상태들을 개별적으로 저장
                for state_key in ["shared", "generation", "solution", "score", "analysis", "artifacts"]:
                    if state_key in teacher_state and teacher_state[state_key]:
                        memory_key = f"{user_id}_{chat_id}_{state_key}"
                        self.memory.put(memory_key, teacher_state[state_key])
            
            # Farmer 상태가 있는 경우 저장
            farmer_state = state.get("farmer_state")
            if farmer_state:
                memory_key = f"{user_id}_{chat_id}_farmer_state"
                self.memory.put(memory_key, farmer_state)
            
            print(f"✅ 메모리 저장 완료")
            
            return {
                **state,
                "memory_saved": True
            }
            
        except Exception as e:
            print(f"❌ 메모리 저장 중 오류: {e}")
            return {
                **state,
                "memory_saved": False,
                "error_message": f"메모리 저장 오류: {str(e)}"
            }
    
    def check_question_relevance(self, state: MainState) -> MainState:
        """
        질문이 농사 자격증과 관련이 있는지 확인하는 노드
        """
        user_query = state.get("user_query") or self._last_user_query
        
        try:
            # LLM을 사용한 관련성 체크
            relevance_prompt = f"""
            사용자의 질문이 농사 또는 자격증(정보처리기사)과 관련이 있는지 분석해주세요.

            질문: {user_query}

            다음 중 하나로만 답변해주세요:
            - "relevant": 농사 또는 자격증, 농업 기술, 작물 재배, 농업 관련 질문, 정보처리기사 관련 질문
            - "irrelevant": 농사 또는 자격증과 전혀 무관한 질문

            답변:
            """
            
            response = self.llm.invoke(relevance_prompt)
            relevance = response.content.strip().lower()
            
            is_relevant = "relevant" in relevance
            
            print(f"🔍 질문 관련성 체크: {'관련됨' if is_relevant else '무관함'}")
            
            return {
                **state,
                "is_relevant": is_relevant,
                "relevance_check_reason": relevance
            }
            
        except Exception as e:
            print(f"❌ 관련성 체크 중 오류: {e}")
            return {
                **state,
                "is_relevant": True,  # 오류 시 기본적으로 관련 있다고 가정
                "relevance_check_reason": f"오류로 인한 기본값: {str(e)}"
            }

    def check_service_consistency(self, state: MainState) -> MainState:
        """
        메인에서 로드한 메모리 데이터를 기반으로 서비스 일관성을 체크하는 노드
        """
        user_query = state["user_query"]
        
        try:
            # 메인에서 로드한 데이터 사용
            session_data = state.get("session_data", {})
            chat_history = state.get("chat_history", [])
            
            # 현재 세션에 기존 서비스 분류가 있는지 확인
            previous_service = session_data.get("locked_service")
            
            # 채팅 히스토리에서 최근 서비스 분류 확인
            if not previous_service and chat_history:
                # 가장 최근 대화에서 서비스 확인
                for interaction in reversed(chat_history):
                    if interaction.get("service_classification") in ["teacher", "farmer"]:
                        previous_service = interaction["service_classification"]
                        break
            
            print(f"🔄 서비스 일관성 체크: 이전={previous_service}")
            print(f"📚 채팅 히스토리: {len(chat_history)}개 대화")
            
            # 첫 번째 질문인 경우 또는 기존 서비스가 없는 경우
            if not previous_service:
                return {
                    **state,
                    "service_consistent": True,
                    "is_first_question": True,
                    "consistency_message": ""
                }
            
            # 기존 서비스가 있는 경우, 분류 단계로 넘어가서 비교
            return {
                **state,
                "service_consistent": True,  # 우선 True로 설정, 분류 후 다시 체크
                "is_first_question": False,
                "consistency_message": "",
                "previous_service": previous_service
            }
            
        except Exception as e:
            print(f"❌ 서비스 일관성 체크 중 오류: {e}")
            return {
                **state,
                "service_consistent": True,
                "is_first_question": True,
                "consistency_message": "",
                "error_message": f"일관성 체크 오류: {str(e)}"
            }

    def classify_service(self, state: MainState) -> MainState:
        """
        사용자 입력을 분석하여 서비스를 분류하는 노드
        """
        user_query = state.get("user_query") or self._last_user_query

        # 이전 서비스/첫 질문 유도 플래그를 여기서 계산하여 단일 체크로 간소화
        session_data = state.get("session_data", {})
        chat_history = state.get("chat_history", [])
        previous_service = state.get("previous_service") or session_data.get("locked_service")
        if not previous_service and chat_history:
            for interaction in reversed(chat_history):
                sc = interaction.get("service_classification")
                if sc in ["teacher", "farmer"]:
                    previous_service = sc
                    break
        is_first_question = state.get("is_first_question")
        if is_first_question is None:
            is_first_question = not bool(previous_service)
        
        try:
            # 너무 strict한 서비스 고정은 비활성화하고 항상 재분류
            
            # LLM을 사용한 서비스 분류
            classification_prompt = f"""
            사용자의 질문을 분석하여 어떤 서비스가 필요한지 분류해주세요.

            질문: {user_query}

            다음 중 하나로만 답변해주세요:
            - "farmer": 농업, 재배, 작물, 농사, 농업기술, 작물관리 관련 질문
            - "teacher": 자격증, 시험, 학습, 교육, 문제풀이, 시험준비 관련 질문

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
                "classification_confidence": 1.0,
                "classification_reason": "LLM 분류 결과",
                "is_first_question": is_first_question,
                "previous_service": previous_service,
            }
            
        except Exception as e:
            print(f"❌ 서비스 분류 중 오류: {e}")
            return {
                **state,
                "service_classification": "teacher",  # 기본값
                "classification_confidence": 0.0,
                "classification_reason": f"오류로 인한 기본값: {str(e)}",
                "is_first_question": is_first_question,
                "previous_service": previous_service,
            }
    
    def handle_irrelevant_question(self, state: MainState) -> MainState:
        """
        농사 자격증과 무관한 질문에 대한 안내 응답을 생성하는 노드
        """
        user_query = state.get("user_query") or self._last_user_query
        
        try:
            # LLM을 사용한 친절한 안내 응답 생성
            response_prompt = f"""
            사용자가 농사 또는 정보처리기사 자격증과 관련 없는 질문을 했습니다. 친절하고 도움이 되는 안내 메시지를 작성해주세요.

            사용자 질문: {user_query}

            다음 내용을 포함하여 응답해주세요:
            1. 질문에 대한 간단한 답변 (가능한 경우)
            2. 이 시스템은 농사 또는 정보처리기사 자격증 관련 질문에 특화되어 있다는 안내
            3. 농사 또는 정보처리기사사 자격증 관련 질문을 입력해달라는 요청
            4. 친근하고 도움이 되는 톤

            한국어로 200자 이내로 작성해주세요.
            """
            
            # LLM 모델 설정을 더 관대하게 변경
            client_for_response = OpenAI(base_url=llm_base_url, api_key=llm_api_key)
            response = client_for_response.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": response_prompt}],
                temperature=0.7,
                max_tokens=300
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            print(f"🚫 무관한 질문 처리됨")
            
            return {
                **state,
                "final_response": generated_response,
                "hitl_required": False,
                "service_classification": "irrelevant"
            }
            
        except Exception as e:
            print(f"❌ 무관한 질문 응답 생성 중 오류: {e}")
            fallback_response = f"""
안녕하세요! 

죄송하지만 이 시스템은 농사 자격증(농산업기사, 종자산업기사 등) 관련 질문에 특화되어 있습니다.

농업 기술, 작물 재배, 시험 준비 등 농사 자격증과 관련된 질문을 입력해주시면 더 정확하고 도움이 되는 답변을 드릴 수 있습니다.

감사합니다! 😊
"""
            return {
                **state,
                "final_response": fallback_response,
                "hitl_required": False,
                "service_classification": "irrelevant",
                "error_message": f"응답 생성 오류: {str(e)}"
            }

    def handle_service_inconsistency(self, state: MainState) -> MainState:
        """
        서비스 일관성 문제를 처리하는 노드
        """
        consistency_message = state.get("consistency_message", "")
        
        return {
            **state,
            "final_response": consistency_message,
            "hitl_required": True,
            "hitl_data": {
                "type": "service_consistency",
                "message": consistency_message,
                "options": ["계속", "새 채팅"]
            }
        }

    def update_session_data(self, state: MainState) -> MainState:
        """
        세션 데이터를 업데이트하여 서비스 고정 상태를 저장하는 노드
        """
        service_classification = state.get("service_classification")
        is_first_question = state.get("is_first_question", False)
        
        # 첫 번째 질문이고 유효한 서비스인 경우, 세션에 고정
        if is_first_question and service_classification in ["teacher", "farmer"]:
            try:
                # 메모리에 서비스 고정 상태 저장
                session_update = {
                    "locked_service": service_classification,
                    "first_classification_time": time.time(),
                    "service_locked": True
                }
                
                # 세션 데이터 업데이트
                updated_session_data = {**state.get("session_data", {}), **session_update}
                
                print(f"🔒 서비스 고정됨: {service_classification}")
                
                return {
                    **state,
                    "session_data": updated_session_data
                }
                
            except Exception as e:
                print(f"❌ 세션 데이터 업데이트 중 오류: {e}")
                return state
        
        return state

    def route_after_relevance_check(self, state: MainState) -> str:
        """
        관련성 체크 후 라우팅 결정
        """
        is_relevant = state.get("is_relevant", True)
        if not is_relevant:
            return "handle_irrelevant_question"
        return "classify_service"

    # 간소화로 인해 사용되지 않음 (보존만 함)
    def route_after_consistency_check(self, state: MainState) -> str:
        return "classify_service"

    def check_final_service_consistency(self, state: MainState) -> MainState:
        """
        서비스 분류 후 실제 서비스 불일치를 체크하는 노드
        """
        current_service = state.get("service_classification", "teacher")
        previous_service = state.get("previous_service")
        is_first_question = state.get("is_first_question", False)
        
        # 첫 번째 질문이거나 이전 서비스가 없는 경우 일관성 문제 없음
        if is_first_question or not previous_service:
            return {
                **state,
                "service_consistent": True
            }
        
        # 서비스가 다른 경우
        if previous_service != current_service:
            consistency_message = f"""
            🚨 서비스 변경 감지됨!

            기존 대화에서는 '{previous_service}' 서비스를 사용하고 있었는데,
            현재 질문은 '{current_service}' 서비스가 필요한 것으로 분류되었습니다.

            더 나은 답변을 위해 새로운 채팅을 시작하는 것을 권장합니다.
            현재 대화를 계속하시겠습니까, 아니면 새 채팅을 시작하시겠습니까?

            - 현재 대화 계속: '계속'
            - 새 채팅 시작: '새 채팅'
            """
            return {
                **state,
                "service_consistent": False,
                "consistency_message": consistency_message
            }
        
        # 서비스가 일관된 경우
        return {
            **state,
            "service_consistent": True
        }

    def route_after_classification(self, state: MainState) -> str:
        """
        서비스 분류 후 라우팅 결정
        """
        is_first_question = state.get("is_first_question", False)
        
        # 첫 번째 질문인 경우 세션 데이터 업데이트
        if is_first_question:
            return "update_session_data"
        else:
            # 기존 대화인 경우 최종 일관성 체크
            return "check_final_service_consistency"

    def route_after_final_consistency_check(self, state: MainState) -> str:
        """
        최종 일관성 체크 후 라우팅 결정
        """
        service_consistent = state.get("service_consistent", True)
        
        if not service_consistent:
            return "handle_service_inconsistency"
        else:
            service = state.get("service_classification", "teacher")
            if service == "farmer":
                return "execute_farmer"
            elif service == "teacher":
                return "teacher_app"
            else:
                return "teacher_app"

    def route_after_session_update(self, state: MainState) -> str:
        """
        세션 데이터 업데이트 후 라우팅 결정
        """
        service = state.get("service_classification", "teacher")
        
        if service == "farmer":
            return "execute_farmer"
        elif service == "teacher":
            return "teacher_app"
        else:
            return "teacher_app"

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
        호환성 유지를 위한 래퍼. 현재는 서브그래프 노드 `teacher_app`을 사용합니다.
        """
        return state
    
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
    
    def merge_teacher_result(self, state: MainState) -> MainState:
        """
        Teacher 서비스 실행 결과를 메인 상태에 병합하는 노드
        
        Args:
            state: 메인 상태 (teacher_state 포함)
            
        Returns:
            Teacher 결과가 병합된 메인 상태
        """
        print("🔄 Teacher 결과 병합 중...")
        
        ts = state.get("teacher_state")

        # teacher_state가 없으면 top-level을 teacher 상태로 간주 (raw 서브그래프 대비)
        if not ts and any(k in state for k in ("final_response", "artifacts", "routing", "shared", "score")):
            ts = {k: state.get(k) for k in ("final_response", "artifacts", "routing", "shared", "score") if k in state}

        if not ts:
            return state

        # 복사 대상 키 확장: shared/score 추가
        for key in ("final_response", "artifacts", "routing", "shared", "score"):
            if key in ts:
                state[key] = ts[key]
        return state
        
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
            "last_query": state.get("user_query", self._last_user_query),
            "last_service": state.get("service_classification", ""),
            "last_response": state.get("final_response", ""),
            "timestamp": str(int(time.time()))
        })
        
        return {
            **state,
            "session_data": session_data
        }
    
    def clear_short_term_memory(self):
        """
        숏텀 메모리 초기화
        테스트 실행 전에 호출하여 이전 상태를 정리
        """
        try:
            # Redis 메모리 초기화
            if hasattr(self.memory, 'clear'):
                self.memory.clear(include_questions=True)
                print("🧹 Redis 숏텀 메모리 초기화 완료")
            
            # Teacher 메모리 초기화
            if hasattr(self.teacher, 'memory') and hasattr(self.teacher.memory, 'clear'):
                self.teacher.memory.clear(include_questions=True)
                print("🧹 Teacher 숏텀 메모리 초기화 완료")
                
        except Exception as e:
            print(f"⚠️ 메모리 초기화 중 오류: {e}")
            print("📝 메모리 초기화를 건너뜁니다.")
    
    def _create_graph(self) -> StateGraph:
        """
        LangGraph 워크플로우 생성 - 새로운 분기 로직 포함
        
        Returns:
            StateGraph: 생성된 그래프
        """
        # 그래프 빌더 생성
        builder = StateGraph(MainState)

        # 메모리 관련 노드
        builder.add_node("load_memory_data", self.load_memory_data)
        builder.add_node("save_memory_data", self.save_memory_data)
        
        # 노드 추가 - 새로운 분기 로직
        builder.add_node("check_question_relevance", self.check_question_relevance)
        builder.add_node("handle_irrelevant_question", self.handle_irrelevant_question)
        builder.add_node("handle_service_inconsistency", self.handle_service_inconsistency)
        builder.add_node("classify_service", self.classify_service)
        builder.add_node("check_final_service_consistency", self.check_final_service_consistency)
        builder.add_node("update_session_data", self.update_session_data)
        
        # 기존 노드들
        builder.add_node("hitl_confirmation", self.hitl_confirmation)
        # Teacher 서브그래프를 그대로 노드로 등록하여 interrupt가 버블업되도록 구성
        builder.add_node("teacher_app", self.teacher.graph)
        builder.add_node("merge_teacher_result", self.merge_teacher_result)
        builder.add_node("execute_farmer", self.execute_farmer)
        builder.add_node("finalize_response", self.finalize_response)
        
        # 새로운 워크플로우 엣지 추가
        # 1. 시작 -> 메모리 로드 -> 관련성 체크
        builder.add_edge(START, "load_memory_data")
        builder.add_edge("load_memory_data", "check_question_relevance")
        
        # 2. 관련성 체크 후 분기
        builder.add_conditional_edges(
            "check_question_relevance",
            self.route_after_relevance_check,
            {
                "handle_irrelevant_question": "handle_irrelevant_question",
                # 간소화: 바로 classify_service로 이동
                "classify_service": "classify_service",
                # 재개 시에는 teacher로 바로 진입
                "teacher_app": "teacher_app",
            }
        )
        
        # 3. 무관한 질문 처리 -> 메모리 저장 -> 종료
        builder.add_edge("handle_irrelevant_question", "save_memory_data")
        
        # 4. 서비스 불일치 처리 (HITL 필요) -> 메모리 저장 -> 종료
        builder.add_edge("handle_service_inconsistency", "save_memory_data")
        
        # 5. 서비스 분류 후 분기
        builder.add_conditional_edges(
            "classify_service",
            self.route_after_classification,
            {
                "update_session_data": "update_session_data",
                "check_final_service_consistency": "check_final_service_consistency"
            }
        )
        
        # 5-1. 최종 일관성 체크 후 분기
        builder.add_conditional_edges(
            "check_final_service_consistency",
            self.route_after_final_consistency_check,
            {
                "handle_service_inconsistency": "handle_service_inconsistency",
                "teacher_app": "teacher_app",
                "execute_farmer": "execute_farmer"
            }
        )
        
        # 7. 세션 데이터 업데이트 후 서비스 실행
        builder.add_conditional_edges(
            "update_session_data",
            self.route_after_session_update,
            {
                "teacher_app": "teacher_app",
                "execute_farmer": "execute_farmer"
            }
        )
        
        # 8. 서비스 실행 후 메모리 저장 (wrap 노드 제거, 직접 연결)
        builder.add_edge("teacher_app", "merge_teacher_result")
        builder.add_edge("merge_teacher_result", "save_memory_data")
        builder.add_edge("execute_farmer", "save_memory_data")
        
        # 9. 기존 엣지들 - 메모리 저장 후 완료
        builder.add_edge("hitl_confirmation", "teacher_app")
        builder.add_edge("save_memory_data", "finalize_response")
        builder.add_edge("finalize_response", END)
        
        return builder
    
    # FastAPI용 비동기 메서드들 추가
    async def process_message_async(self, message: str, user_id: str, chat_id: str) -> Dict[str, Any]:
        """FastAPI용 비동기 메시지 처리"""
        try:
            # 새로운 오케스트레이터 인스턴스 생성
            orchestrator = MainOrchestrator(
                user_id=user_id,
                chat_id=chat_id
            )
            
            # 메시지 처리
            result = orchestrator.run(message)
            
            return {
                "response": result.get("final_response", "응답을 생성할 수 없습니다"),
                "service_used": result.get("service_classification", "unknown"),
                "confidence": result.get("classification_confidence", 0.8),
                "artifacts": result.get("artifacts", {}),
                "shared": result.get("shared", {}),
                "score": result.get("score", {})
            }
            
        except Exception as e:
            return {
                "response": f"처리 중 오류가 발생했습니다: {str(e)}",
                "service_used": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def process_teacher_message_async(self, message: str, user_id: str, chat_id: str) -> Dict[str, Any]:
        """Teacher 서비스 전용 비동기 메시지 처리"""
        try:
            # Teacher 서비스로 직접 라우팅
            from teacher.teacher_graph import Teacher
            
            teacher = Teacher(
                user_id=user_id,
                service="teacher",
                chat_id=chat_id,
                init_agents=True
            )
            
            # Teacher 상태 생성
            teacher_state = {
                "user_query": message,
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
            
            # Teacher 실행
            result = teacher.execute(teacher_state)
            
            return {
                "response": result.get("llm_response", "Teacher 서비스 응답을 생성할 수 없습니다"),
                "service_used": "teacher",
                "confidence": 0.9,
                "artifacts": result.get("artifacts", {}),
                "shared": result.get("shared", {}),
                "score": result.get("score", {})
            }
            
        except Exception as e:
            return {
                "response": f"Teacher 서비스 처리 중 오류가 발생했습니다: {str(e)}",
                "service_used": "teacher",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def process_farmer_message_async(self, message: str, user_id: str, chat_id: str) -> Dict[str, Any]:
        """Farmer 서비스 전용 비동기 메시지 처리"""
        try:
            # Farmer 서비스 실행 (기본 응답)
            farmer_response = f"Farmer 서비스: {message}에 대한 농업 관련 답변을 제공합니다."
            
            return {
                "response": farmer_response,
                "service_used": "farmer",
                "confidence": 0.9,
                "artifacts": {},
                "shared": {},
                "score": {}
            }
            
        except Exception as e:
            return {
                "response": f"Farmer 서비스 처리 중 오류가 발생했습니다: {str(e)}",
                "service_used": "farmer",
                "confidence": 0.0,
                "error": str(e)
            }
    
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
        
        # checkpointer 설정 (LangGraph 규격: configurable로 전달)
        if config is None:
            config = {}
        cfg = config.get("configurable", {})
        cfg.update({
            "thread_id": self.thread_id,
        })
        config["configurable"] = cfg
        
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
        # LangGraph Command를 사용해 정확히 중단 지점에서 재개
        from langgraph.types import Command
        try:
            if config is None:
                config = {}
            cfg = (config.get("configurable", {}) or {})
            cfg.update({
                "thread_id": self.thread_id,
                "checkpoint_id": self.checkpoint_id,
            })
            config["configurable"] = cfg
            # 재개 시에도 적절한 인터럽트 포인트를 유지
            config["interrupt_after"] = [
                "teacher_app.await_output_mode",
                "teacher_app.await_form_answers",
            ]
            # Teacher 측 힌트 주입 (있으면 사용되고 곧바로 소모됨)
            try:
                if isinstance(hitl_response, str) and hitl_response.lower() in ("pdf", "form"):
                    setattr(self.teacher, "_pending_output_mode", hitl_response.lower())
                if isinstance(hitl_response, dict) and "user_answer" in hitl_response:
                    setattr(self.teacher, "_pending_form_answers", list(hitl_response["user_answer"]))
                    setattr(self.teacher, "_pending_output_mode", "form")
                if isinstance(hitl_response, dict) and "user_feedback" in hitl_response:
                    setattr(self.teacher, "_pending_user_feedback", str(hitl_response["user_feedback"]))
            except Exception:
                pass
            resume_cmd = Command(resume=hitl_response)
            result = self.app.invoke(resume_cmd, config=config)
            res = dict(result)
            # 보강: teacher_state가 있으면 산출물/중요 키를 항상 노출
            ts = res.get("teacher_state")
            if isinstance(ts, dict):
                # artifacts는 항상 최신으로 덮어써서 PDF 생성 여부가 드러나도록 함
                if "artifacts" in ts:
                    res["artifacts"] = ts["artifacts"]
                # 나머지는 없을 때만 채움
                for key in ("final_response", "routing", "shared", "score"):
                    if key in ts and key not in res:
                        res[key] = ts[key]
            return res
        except Exception as e:
            print(f"❌ 워크플로우 재개 중 오류: {e}")
            return {"error": str(e), "hitl_response": hitl_response}

def visualize_main_graph():
    """메인 오케스트레이터 그래프를 시각화합니다."""
    try:
        orchestrator = MainOrchestrator(
            user_id="demo_user",
            chat_id="demo_chat"
        )
        
        # 컴파일된 그래프에서 get_graph() 호출
        compiled = orchestrator.app
        g = compiled.get_graph()

        dot = graphviz.Digraph(comment="Main Orchestrator Graph", format="png")
        dot.attr(rankdir="TD")

        # 노드 추가 - nodes가 메서드인지 속성인지 확인
        nodes = g.nodes() if callable(g.nodes) else g.nodes
        for n in nodes:
            dot.node(str(n), shape="box")

        # 엣지 추가 - edges가 메서드인지 속성인지 확인
        edges = g.edges() if callable(g.edges) else g.edges
        for e in edges:
            if isinstance(e, (tuple, list)) and len(e) >= 2:
                dot.edge(str(e[0]), str(e[1]))

        path = dot.render("main_orchestrator_graph", cleanup=True)
        print(f"📊 메인 오케스트레이터 그래프 저장됨: {path}")
        return path
    except Exception as e:
        print(f"❌ 메인 그래프 시각화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_teacher_graph():
    """Teacher 에이전트 그래프를 시각화합니다."""
    try:
        from teacher.teacher import Teacher
        
        # 실행 없이 그래프 구조만 빌드하므로 init_agents=False
        teacher = Teacher(user_id="demo", service="teacher", chat_id="viz", init_agents=False)
        compiled = teacher.graph
        g = compiled.get_graph()

        dot = graphviz.Digraph(comment="Teacher Agent Graph", format="png")
        dot.attr(rankdir="TD")

        # 노드 추가 - nodes가 메서드인지 속성인지 확인
        nodes = g.nodes() if callable(g.nodes) else g.nodes
        for n in nodes:
            dot.node(str(n), shape="box")

        # 엣지 추가 - edges가 메서드인지 속성인지 확인
        edges = g.edges() if callable(g.edges) else g.edges
        for e in edges:
            if isinstance(e, (tuple, list)) and len(e) >= 2:
                dot.edge(str(e[0]), str(e[1]))

        path = dot.render("teacher_agent_graph", cleanup=True)
        print(f"📊 Teacher 에이전트 그래프 저장됨: {path}")
        return path
    except Exception as e:
        print(f"❌ Teacher 그래프 시각화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def _get_quiz_from_snapshot(orchestrator, cfg):
    """스냅샷에서 현재 폼의 문항/보기 가져오기"""
    try:
        snap = orchestrator.app.get_state(cfg)
        vals = getattr(snap, "values", {}) or {}
    except Exception:
        vals = {}
    sh = vals.get("shared", {}) or {}
    questions = sh.get("question") or []
    options = sh.get("options") or []  # 2차원 배열(문항별 보기)
    return questions, options

def _ask_answers_interactively(q_count: int, options) -> list[str]:
    """
    정답 입력을 안전하게 받는다.
    - 구분자: 콤마, 공백, 전각 콤마(，) 모두 허용
    - 개수 불일치 시 재입력
    - 숫자 아님/범위 벗어나면 재입력
    """
    while True:
        ans_in = input(f"Bot> 정답을 입력하세요 (문항 {q_count}개, 예: 2,1,3): ").strip()
        # 구분자 정규화
        normalized = ans_in.replace("，", ",").replace(" ", ",")
        parts = [p for p in normalized.split(",") if p]

        # 개수 검증
        if len(parts) != q_count:
            print(f"Bot> ⚠️ 문항 수({q_count})와 입력 개수({len(parts)})가 다릅니다. 다시 입력하세요.")
            continue

        # 숫자/범위 검증
        ok = True
        nums: list[int] = []
        for i, p in enumerate(parts):
            if not p.isdigit():
                print(f"Bot> ⚠️ 모든 정답은 숫자(1~보기수)여야 합니다. 다시 입력하세요.")
                ok = False
                break
            n = int(p)
            # 보기 개수가 있으면 범위 체크 (없으면 1 이상만 체크)
            opt_count = len(options[i]) if i < len(options) and isinstance(options[i], list) else None
            if n < 1 or (opt_count is not None and n > opt_count):
                if opt_count:
                    print(f"Bot> ⚠️ {i+1}번 문항의 정답은 1~{opt_count} 사이여야 합니다. 다시 입력하세요.")
                else:
                    print(f"Bot> ⚠️ {i+1}번 문항의 정답은 1 이상의 숫자여야 합니다. 다시 입력하세요.")
                ok = False
                break
            nums.append(n)
        if not ok:
            continue

        # 정상 입력 → 문자열 리스트로 반환 (['2','1','3'] 형태)
        return [str(n) for n in nums]


def _make_cfg(orchestrator):
    return {
        "configurable": {
            "thread_id": orchestrator.thread_id,
            "checkpoint_id": orchestrator.checkpoint_id,
        },
        "interrupt_after": [
            "teacher_app.await_output_mode",
            "teacher_app.await_form_answers",
        ],
    }

def _get_snapshot(orchestrator, cfg):
    try:
        return orchestrator.app.get_state(cfg)
    except Exception:
        return None

def _get_pending_nodes(orchestrator, cfg):
    snap = _get_snapshot(orchestrator, cfg)
    if snap is None:
        return []
    # 우선 그래프 엔진이 제공하는 helper가 있으면 사용
    try:
        helper = getattr(orchestrator.app, "get_pending_nodes", None)
        if callable(helper):
            nodes = helper(cfg) or []
            return [str(n) for n in nodes]
    except Exception:
        pass
    # fallback: 다양한 형태 지원
    nxt = getattr(snap, "next", None)
    if nxt is None and isinstance(snap, dict):
        nxt = snap.get("next") or snap.get("pending")
    try:
        return [str(n) for n in (nxt or [])]
    except Exception:
        return []

def _get_values(orchestrator, cfg):
    snap = _get_snapshot(orchestrator, cfg)
    if snap is None:
        return {}
    # 1) 객체 속성 형태
    vals = getattr(snap, "values", None)
    if isinstance(vals, dict) and vals:
        return vals
    # 2) 딕셔너리 형태
    if isinstance(snap, dict):
        if isinstance(snap.get("values"), dict):
            return snap.get("values") or {}
        # 일부 실행기에서 상태가 top-level에 직접 실릴 수 있음
        return snap
    return {}

def _print_outputs_if_any(state):
    final_resp = state.get("final_response")
    if final_resp:
        print(f"Bot> {final_resp}")
        return True
    arts = state.get("artifacts") or (state.get("teacher_state") or {}).get("artifacts") or {}
    gen = arts.get("generated_pdfs") or arts.get("pdfs")
    if isinstance(gen, list) and gen:
        print("Bot> PDF 생성 완료:")
        for p in gen:
            print(f"  - {p}")
        return True
    return False

def _in_await_output_mode(orchestrator, cfg):
    # pending 없을 때도 상태값 기반으로 보정
    vals = _get_values(orchestrator, cfg)
    routing = (vals.get("routing") or
               (vals.get("teacher_state") or {}).get("routing") or {})
    stage = routing.get("stage") or routing.get("await") or routing.get("pending")
    # teacher가 stage="await_output_mode" 같은 키를 넣는 경우를 커버
    return (stage == "await_output_mode") or bool(routing.get("await_output_mode"))

def _in_await_form_answers(orchestrator, cfg):
    vals = _get_values(orchestrator, cfg)
    routing = (vals.get("routing") or
               (vals.get("teacher_state") or {}).get("routing") or {})
    stage = routing.get("stage") or routing.get("await") or routing.get("pending")
    ua = (vals.get("shared") or {}).get("user_answer") or \
         ((vals.get("teacher_state") or {}).get("shared") or {}).get("user_answer")
    # 폼이 준비됐는데 user_answer가 아직 없으면 대기 상태로 간주
    return (stage == "await_form_answers") or bool(routing.get("await_form_answers")) or (routing.get("output_mode") == "form" and not ua)

def _get_quiz_from_snapshot(orchestrator, cfg):
    vals = _get_values(orchestrator, cfg)
    sh = vals.get("shared", {}) or (vals.get("teacher_state") or {}).get("shared", {}) or {}
    return sh.get("question") or [], sh.get("options") or []

def _ask_answers_interactively(q_count: int, options) -> list[str]:
    while True:
        ans_in = input(f"Bot> 정답을 입력하세요 (문항 {q_count}개, 예: 2,1,3): ").strip()
        normalized = ans_in.replace("，", ",").replace(" ", ",")
        parts = [p for p in normalized.split(",") if p]
        if len(parts) != q_count:
            print(f"Bot> ⚠️ 문항 수({q_count})와 입력 개수({len(parts)})가 다릅니다. 다시 입력하세요.")
            continue
        ok, nums = True, []
        for i, p in enumerate(parts):
            if not p.isdigit():
                print("Bot> ⚠️ 모든 정답은 숫자(1~보기수)여야 합니다. 다시 입력하세요.")
                ok = False; break
            n = int(p)
            opt_count = len(options[i]) if i < len(options) and isinstance(options[i], list) else None
            if n < 1 or (opt_count is not None and n > opt_count):
                if opt_count:
                    print(f"Bot> ⚠️ {i+1}번 문항의 정답은 1~{opt_count} 사이여야 합니다. 다시 입력하세요.")
                else:
                    print(f"Bot> ⚠️ {i+1}번 문항의 정답은 1 이상의 숫자여야 합니다. 다시 입력하세요.")
                ok = False; break
            nums.append(n)
        if not ok:
            continue
        return [str(n) for n in nums]


def main():
    print("🚀 ET-Agent 메인 오케스트레이터 시작 (대화형 모드)")
    orchestrator = MainOrchestrator(user_id="cli_user", chat_id="cli_chat")
    print("명령: /exit 종료, /clear 메모리 초기화, /new 새 세션")

    while True:
        cfg = _make_cfg(orchestrator)

        # 0) 스냅샷에서 인터럽트 대기 여부를 '두 가지'로 판정 (pending + 상태 보정)
        pending = _get_pending_nodes(orchestrator, cfg)
        print(pending)
        try:
            if pending:
                print(f"[DEBUG] pending nodes: {pending}")
        except Exception:
            pass
        in_await_mode = any(p.startswith("teacher_app.await_output_mode") for p in pending) or _in_await_output_mode(orchestrator, cfg)
        in_await_form = any(p.startswith("teacher_app.await_form_answers") for p in pending) or _in_await_form_answers(orchestrator, cfg)
        print(in_await_mode)
        print(in_await_form)
        # A) 출력 모드 대기 → 무조건 모드 입력 받고 resume
        if in_await_mode:
            # 상태에 이미 모드가 있으면 즉시 그 모드로 resume
            vals = _get_values(orchestrator, cfg)
            routing = (vals.get("routing") or (vals.get("teacher_state") or {}).get("routing") or {})
            decided_mode = (routing or {}).get("output_mode")
            if decided_mode in ("pdf", "form"):
                state = orchestrator.resume_workflow(decided_mode, config=cfg)
                if state.get("error"):
                    print(f"Bot> 오류: {state['error']}")
                else:
                    _print_outputs_if_any(state)
                continue
            while True:
                try:
                    mode_in = input("Bot> 출력 방식을 선택하세요 (pdf|form): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 종료합니다."); return
                if mode_in in ("pdf", "form"):
                    state = orchestrator.resume_workflow(mode_in, config=cfg)
                    if state.get("error"):
                        print(f"Bot> 오류: {state['error']}")
                    else:
                        _print_outputs_if_any(state)
                    break
                print("Bot> 잘못된 입력입니다. 'pdf' 또는 'form'을 입력하세요.")
            continue

        # B) 폼 정답 대기 → 문항/보기 읽어와 검증 입력 받고 resume
        if in_await_form:
            qs, opts = _get_quiz_from_snapshot(orchestrator, cfg)
            if len(qs) <= 0:
                # 혹시 비어 있으면 form으로 한 번 흘려 상태 보강
                state = orchestrator.resume_workflow("form", config=cfg)
                qs, opts = _get_quiz_from_snapshot(orchestrator, cfg)
            if len(qs) <= 0:
                print("Bot> ⚠️ 아직 문항이 준비되지 않았습니다. 다시 시도해 주세요.")
                continue
            answers = _ask_answers_interactively(len(qs), opts)
            state = orchestrator.resume_workflow({"user_answer": answers}, config=cfg)
            if state.get("error"):
                print(f"Bot> 오류: {state['error']}")
            else:
                _print_outputs_if_any(state)
            continue

        # C) 여기까지 왔으면 인터럽트 없음 → 새 입력 받기
        try:
            user_query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 종료합니다."); break
        if not user_query:
            continue
        cmd = user_query.lower()
        if cmd in ("/exit", ":q", "quit", "exit"):
            print("👋 종료합니다."); break
        if cmd == "/clear":
            orchestrator.clear_short_term_memory()
            print("🧹 메모리 초기화 완료"); continue
        if cmd == "/new":
            import uuid
            new_chat = uuid.uuid4().hex
            orchestrator.chat_id = new_chat
            orchestrator.thread_id = f"teacher:{orchestrator.user_id}:{new_chat}"
            print("🔁 새 세션 시작되었습니다."); continue

        # D) 새 실행
        state = orchestrator.run(user_query, config=cfg)
        if state.get("error"):
            print(f"Bot> 오류: {state['error']}"); continue
        _print_outputs_if_any(state)
        # 다음 루프에서 곧바로 pending/상태를 다시 보고 resume 프롬프트를 띄웁니다.


# FastAPI 서버 추가
def start_fastapi_server():
    """FastAPI 서버를 시작하는 함수"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
        
        # FastAPI 앱 생성
        app = FastAPI(
            title="ET-Agent API",
            description="농업 자격증 및 교육 관련 AI 에이전트 API",
            version="1.0.0"
        )
        
        # CORS 미들웨어 설정
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Pydantic 모델
        class ChatRequest(BaseModel):
            message: str
            user_id: str = "api_user"
            chat_id: str = "api_chat"
        
        class ChatResponse(BaseModel):
            response: str
            service_used: str
            confidence: float
            session_id: str
        
        # 전역 오케스트레이터 인스턴스
        orchestrator = None
        
        @app.on_event("startup")
        async def startup_event():
            """서버 시작 시 오케스트레이터 초기화"""
            global orchestrator
            try:
                orchestrator = MainOrchestrator()
                print("✅ 오케스트레이터 초기화 완료")
            except Exception as e:
                print(f"⚠️ 오케스트레이터 초기화 실패: {e}")
                orchestrator = None
        
        @app.get("/")
        async def root():
            return {
                "message": "ET-Agent API",
                "version": "1.0.0",
                "status": "running",
                "docs": "/docs"
            }
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "version": "1.0.0",
                "orchestrator": "ready" if orchestrator else "not_ready"
            }
        
        @app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """채팅 엔드포인트"""
            if not orchestrator:
                raise HTTPException(status_code=503, detail="오케스트레이터가 초기화되지 않았습니다")
            
            try:
                # 오케스트레이터를 사용하여 응답 생성
                response = await orchestrator.process_message_async(
                    request.message, 
                    request.user_id, 
                    request.chat_id
                )
                
                return ChatResponse(
                    response=response.get("response", "응답을 생성할 수 없습니다"),
                    service_used=response.get("service_used", "unknown"),
                    confidence=response.get("confidence", 0.8),
                    session_id=f"{request.user_id}:{request.chat_id}"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")
        
        @app.post("/chat/teacher", response_model=ChatResponse)
        async def chat_teacher(request: ChatRequest):
            """Teacher 서비스 전용 채팅"""
            if not orchestrator:
                raise HTTPException(status_code=503, detail="오케스트레이터가 초기화되지 않았습니다")
            
            try:
                # Teacher 서비스로 직접 라우팅
                response = await orchestrator.process_teacher_message_async(
                    request.message, 
                    request.user_id, 
                    request.chat_id
                )
                
                return ChatResponse(
                    response=response.get("response", "응답을 생성할 수 없습니다"),
                    service_used="teacher",
                    confidence=response.get("confidence", 0.9),
                    session_id=f"{request.user_id}:{request.chat_id}"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")
        
        @app.post("/chat/farmer", response_model=ChatResponse)
        async def chat_farmer(request: ChatRequest):
            """Farmer 서비스 전용 채팅"""
            if not orchestrator:
                raise HTTPException(status_code=503, detail="오케스트레이터가 초기화되지 않았습니다")
            
            try:
                # Farmer 서비스로 직접 라우팅
                response = await orchestrator.process_farmer_message_async(
                    request.message, 
                    request.user_id, 
                    request.chat_id
                )
                
                return ChatResponse(
                    response=response.get("response", "응답을 생성할 수 없습니다"),
                    service_used="farmer",
                    confidence=response.get("confidence", 0.9),
                    session_id=f"{request.user_id}:{request.chat_id}"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")
        
        # LangGraph 호환 엔드포인트 추가
        @app.post("/runs/stream")
        async def langgraph_stream(request: dict):
            """LangGraph 스트림 엔드포인트"""
            if not orchestrator:
                raise HTTPException(status_code=503, detail="오케스트레이터가 초기화되지 않았습니다")
            
            try:
                # 요청에서 메시지 추출
                input_data = request.get("input", {})
                messages = input_data.get("messages", [])
                
                if not messages:
                    raise HTTPException(status_code=400, detail="메시지가 없습니다")
                
                # 마지막 메시지의 내용 추출
                last_message = messages[-1]
                user_message = last_message.get("content", "")
                
                # 오케스트레이터 실행
                result = orchestrator.run(
                    user_query=user_message,
                    config={
                        "configurable": {
                            "thread_id": f"copilotkit:{input_data.get('thread_id', 'default')}",
                        }
                    }
                )
                
                # LangGraph 형식으로 응답 반환
                response_data = {
                    "run_id": f"run_{hash(user_message)}",
                    "status": "success",
                    "output": {
                        "messages": [
                            {
                                "role": "assistant",
                                "content": result.get("final_response", "응답을 생성할 수 없습니다")
                            }
                        ]
                    }
                }
                
                return response_data
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")
        
        @app.get("/assistants")
        async def get_assistants():
            """어시스턴트 목록 반환"""
            return {
                "assistants": [
                    {
                        "assistant_id": "sample_agent",
                        "name": "ET-Agent",
                        "description": "농업 자격증 및 교육 관련 AI 에이전트"
                    }
                ]
            }
        
        # 서버 시작
        print("🚀 ET-Agent FastAPI 서버 시작")
        print("📍 주소: http://localhost:8000")
        print("📚 API 문서: http://localhost:8000/docs")
        print("🔍 헬스 체크: http://localhost:8000/health")
        print("-" * 50)
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ FastAPI 관련 패키지가 설치되지 않았습니다: {e}")
        print("다음 명령어로 설치해주세요: pip install fastapi uvicorn")
        return False
    except Exception as e:
        print(f"❌ FastAPI 서버 시작 실패: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # 명령행 인수 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # FastAPI 서버 모드
        start_fastapi_server()
    else:
        # 기존 콘솔 모드
        main()
