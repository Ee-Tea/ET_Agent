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
import graphviz

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
    
    def load_memory_data(self, state: MainState) -> MainState:
        """
        숏텀 메모리에서 데이터를 로드하는 노드
        """
        user_id = state["user_id"]
        chat_id = state["chat_id"]
        
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
            
            return {
                **state,
                "session_data": session_data,
                "chat_history": chat_history,
                "loaded_memory_data": loaded_data,
                "memory_loaded": True
            }
            
        except Exception as e:
            print(f"❌ 메모리 로드 중 오류: {e}")
            return {
                **state,
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
        user_id = state["user_id"]
        chat_id = state["chat_id"]
        
        try:
            print(f"💾 메모리 데이터 저장 중... user_id={user_id}, chat_id={chat_id}")
            
            # 현재 대화를 채팅 히스토리에 추가
            current_interaction = {
                "timestamp": time.time(),
                "user_query": state["user_query"],
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
        user_query = state["user_query"]
        
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
        user_query = state["user_query"]

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
            # 첫 번째 질문이 아니고 이전 서비스가 있는 경우, 해당 서비스로 고정
            if not is_first_question and previous_service:
                print(f"🔒 서비스 고정됨: {previous_service}")
                return {
                    **state,
                    "service_classification": previous_service,
                    "classification_confidence": 1.0,
                    "classification_reason": f"이전 세션에서 고정된 서비스: {previous_service}",
                    "is_first_question": is_first_question,
                    "previous_service": previous_service,
                }
            
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
        user_query = state["user_query"]
        
        try:
            # LLM을 사용한 친절한 안내 응답 생성
            response_prompt = f"""
사용자가 농사 자격증과 관련 없는 질문을 했습니다. 친절하고 도움이 되는 안내 메시지를 작성해주세요.

사용자 질문: {user_query}

다음 내용을 포함하여 응답해주세요:
1. 질문에 대한 간단한 답변 (가능한 경우)
2. 이 시스템은 농사 자격증(농산업기사, 종자산업기사 등) 관련 질문에 특화되어 있다는 안내
3. 농사 자격증 관련 질문을 입력해달라는 요청
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
        else:
            # 간소화: 바로 서비스 분류로 이동
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
        
        teacher_state = state.get("teacher_state", {})
        
        # Teacher 결과를 메인 상태에 병합
        if teacher_state:
            # Teacher의 최종 응답을 메인 상태에 저장
            if "final_response" in teacher_state:
                state["final_response"] = teacher_state["final_response"]
            
            # Teacher의 아티팩트들을 메인 상태에 저장
            if "artifacts" in teacher_state:
                state["artifacts"] = teacher_state["artifacts"]
            
            # Teacher의 라우팅 정보를 메인 상태에 저장
            if "routing" in teacher_state:
                state["routing"] = teacher_state["routing"]
        
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
            "last_query": state["user_query"],
            "last_service": state["service_classification"],
            "last_response": state["final_response"],
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
        
        # 8. 서비스 실행 후 메모리 저장
        builder.add_edge("teacher_app", "merge_teacher_result")
        builder.add_edge("merge_teacher_result", "save_memory_data")
        builder.add_edge("execute_farmer", "save_memory_data")
        
        # 9. 기존 엣지들 - 메모리 저장 후 완료
        builder.add_edge("hitl_confirmation", "teacher_app")
        builder.add_edge("save_memory_data", "finalize_response")
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
        # LangGraph Command를 사용해 정확히 중단 지점에서 재개
        try:
            from langgraph.checkpoint.memory import Command
        except Exception:
            try:
                from langgraph.types import Command
            except Exception:
                from langgraph import Command  # 최후 시도
        try:
            thread_id = f"{self.user_id}_{self.chat_id}"
            if config is None:
                config = {}
            config.update({
                "thread_id": thread_id,
                "checkpoint_id": f"main_orchestrator"
            })
            resume_cmd = Command(resume=hitl_response)
            result = self.app.invoke(resume_cmd, config=config)
            return dict(result)
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

def main():

    """메인 실행 함수"""
    print("🚀 ET-Agent 메인 오케스트레이터 시작")
    
    # 그래프 시각화 생성
    print("\n📊 그래프 시각화 생성 중...")
    
    print("🔧 메인 그래프 시각화 시도...")
    main_graph_path = visualize_main_graph()
    print(f"메인 그래프 결과: {main_graph_path}")
    
    print("🔧 Teacher 그래프 시각화 시도...")
    teacher_graph_path = visualize_teacher_graph()
    print(f"Teacher 그래프 결과: {teacher_graph_path}")
    
    if main_graph_path:
        print(f"✅ 메인 오케스트레이터 그래프: {main_graph_path}")
    if teacher_graph_path:
        print(f"✅ Teacher 에이전트 그래프: {teacher_graph_path}")
    
    print("📊 그래프 시각화 완료")
    
    # 오케스트레이터 생성
    orchestrator = MainOrchestrator(
        user_id="demo_user",
        chat_id="demo_chat"
    )
    
    # 예제 실행
    test_queries = [
        # "토마토 재배 방법을 알려줘",
        # "소프트웨어 설계 문제 3개 만들어줘",
        """
        문제 6. 데이터베이스 성능에 많은 영향을 주는 DBMS의 구성 요소로, 독립적인 저장 공간을 보유하며
        데이터베이스에 저장된 자료를 더욱 빠르게 조회하기 위해 사용되는 것은?
        1) 인덱스
        2) 테이블
        3) 뷰
        4) 트리거   이거 풀어줘
        """
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n=== 테스트 {i}: {query} ===")
        
        # 테스트 실행 전 숏텀 메모리 초기화
        orchestrator.clear_short_term_memory()
        
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


