import re
import json
import time
import logging
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from typing import Annotated, List, Dict, Optional
import operator
from langsmith import traceable
from dotenv import load_dotenv
import os
import threading
from functools import lru_cache
import traceback
import signal

load_dotenv()

# 전역 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

# 전역 인터럽트 플래그
_interrupt_flag = threading.Event()

# 전역 에이전트 함수 캐시 (지연 로딩용)
_agent_functions_cache = {}

# 병합 함수들을 먼저 정의 (RouterState에서 사용하기 위해)
def merge_dicts(left: dict, right: dict) -> dict:
    """딕셔너리 병합 함수 - LangGraph용"""
    # 타입 검증 및 안전한 처리
    if not isinstance(left, dict):
        left = {} if left is None else {}
    if not isinstance(right, dict):
        right = {} if right is None else {}
    
    if not left:
        return right or {}
    if not right:
        return left or {}
    
    merged = left.copy()
    merged.update(right)
    return merged

def merge_lists_unique(left: list, right: list) -> list:
    """리스트 병합 함수 - 중복 제거 - LangGraph용"""
    # 타입 검증 및 안전한 처리
    if not isinstance(left, list):
        left = [] if left is None else []
    if not isinstance(right, list):
        right = [] if right is None else []
    
    if not left:
        return right or []
    if not right:
        return left or []
    
    # 순서를 유지하면서 중복 제거
    seen = set()
    result = []
    for item in left + right:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class LoggingConfig:
    """로깅 설정 관리 클래스"""
    
    @staticmethod
    def setup_logger(name: str, level: str = None) -> logging.Logger:
        """로거 설정 및 반환 - 부모 로거(root)로만 전송"""
        if level is None:
            level = LOG_LEVEL  # 전역 로그 레벨 사용
            
        logger = logging.getLogger(name)
        
        # 이미 설정된 경우 중복 설정 방지
        if hasattr(logger, '_farmer_configured'):
            return logger
            
        logger.setLevel(getattr(logging, level.upper()))
        
        # 부모 로거(root) 설정 확인 및 설정
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            # 포맷터 설정
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 루트 로거에 핸들러 추가
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            root_logger.setLevel(getattr(logging, level.upper()))
        
        # 부모 로거로 전파하도록 설정 (기본값이지만 명시적으로)
        logger.propagate = True
        
        # 설정 완료 표시
        logger._farmer_configured = True
        
        return logger
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """기존 로거 반환 또는 새로 생성"""
        return LoggingConfig.setup_logger(name)


class FarmerConfig:
    """Farmer 오케스트레이터 설정 클래스"""
    
    # LLM 설정
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.8
    
    
    # 캐시 설정
    LRU_CACHE_SIZE = 128
    
    # 에이전트 설명
    AGENT_DESCRIPTIONS = {
        "작물추천_agent": (
            "사용자의 재배 환경(계절, 토양, 기후 등), 목적, 특정 조건(수확 시기, 맛, 저장성 등)에 맞는 새로운 작물이나 품종을 추천합니다."
            "※ 핵심 키워드: '어떤 작물을 심을까', '무엇을 재배하면 좋을까', '추천해주세요'"
        ),
        "작물재배_agent": (
            "씨앗, 모종 심기부터 작물의 재배 방법, 심는 방법, 이랑을 만드는 방법, 솎음, 영양 관리(시비, 비료, 거름), 병해충 방제, 수확에 이르기까지 특정 작물을 키우는 데 필요한 일상적인 재배 관리 정보를 제공합니다."
            "※ 핵심 키워드: '심는 방법', '키우는 법', '재배 방법', '이랑', '솎음', '거름', '비료', '영양 관리', '병해충', '수확', '어떻게'"
        ),
        "재해_agent": (
            "폭염, 한파, 가뭄, 집중호우, 홍수 등 자연재해 및 이상기후로 인한 피해를 예방하고 대응하는 방법을 안내합니다. 재해 발생 전 대비, 재해 발생 중의 조치, 재해 후 작물 복구 및 피해 최소화 방안을 다룹니다."
            "※ 핵심 키워드: '폭염', '한파', '가뭄', '홍수', '장마', '집중호우', '자연재해', '이상기후', '피해', '대응', '복구'"
        ),
        "판매처_agent": (
            "사용자가 재배하거나 수확한 농산물을 어디에 팔 수 있는지, 판매처 위치 정보와 해당 작물의 시세, 최근 가격 변동을 안내합니다."
            "※ 핵심 키워드: '판매처', '시장', '도매상', '유통', '가격', '시세', '수익', '거래', '실시간 시세', '가격 변동', '팔고 싶어'"
        )
    }
    
    # 에이전트 모듈 매핑
    AGENT_MODULES = {
        "작물추천_agent": "farmer.recommend.crop_recommendation_agent",
        "작물재배_agent": "farmer.cultivation.New_CG_agent", 
        "재해_agent": "farmer.disaster.DisasterAgent_LLM",
        "날씨_agent": "farmer.weather.run_weather_agent_simple",
        "판매처_agent": "farmer.sales.SalesAgent"
    }
    
    @classmethod
    def get_llm_config(cls):
        """LLM 설정 반환"""
        return {
            "model_name": cls.LLM_MODEL,
            "temperature": cls.LLM_TEMPERATURE,
            "api_key": os.getenv("OPENAI_API_KEY")
        }
 
class AgentManager:
    """에이전트 로딩 및 관리 클래스"""
    
    def __init__(self):
        self._cache = _agent_functions_cache
        self.logger = LoggingConfig.get_logger("AgentManager")
    
    def get_agent_function(self, agent_name: str):
        """에이전트 함수를 지연 로딩으로 가져오기 (전역 캐시 사용)"""
        # 해당 에이전트가 아직 로드되지 않은 경우에만 import
        if agent_name not in self._cache:
            self.logger.debug(f"{agent_name} 모듈 로딩 시작")
            
            try:
                if agent_name == "작물추천_agent":
                    from farmer.recommend.crop_recommendation_agent import run as crop_recommend_run
                    self._cache[agent_name] = crop_recommend_run
                elif agent_name == "작물재배_agent":
                    from farmer.cultivation.New_CG_agent import run as crop_cultivation_run
                    self._cache[agent_name] = crop_cultivation_run
                elif agent_name == "재해_agent":
                    from farmer.disaster.DisasterAgent_LLM import run as disaster_run
                    self._cache[agent_name] = disaster_run
                elif agent_name == "날씨_agent":
                    from farmer.weather.run_weather_agent_simple import run as weather_run
                    self._cache[agent_name] = weather_run
                elif agent_name == "판매처_agent":
                    from farmer.sales.SalesAgent import run as market_run
                    self._cache[agent_name] = market_run
                else:
                    self.logger.error(f"알 수 없는 에이전트: {agent_name}")
                    return None
                
                self.logger.info(f"{agent_name} 모듈 로딩 완료")
                
            except Exception as e:
                self.logger.error(f"{agent_name} 모듈 로딩 실패: {e}")
                self._cache[agent_name] = None
                return None
        
        return self._cache.get(agent_name)
    
    def clear_cache(self):
        """에이전트 함수 캐시 정리"""
        self._cache.clear()
        self.logger.info("에이전트 함수 캐시 정리 완료")
    
    def get_loaded_agents(self):
        """현재 로드된 에이전트 목록 반환"""
        return list(self._cache.keys())

class RouterState(dict):
    """LangGraph 상태 클래스 - 적절한 기본값으로 초기화"""
    
    # 사용자 입력 관련
    query: Annotated[List[str], operator.add] = []
    question_parts: Annotated[Dict[str, str], merge_dicts] = {}
    
    # 에이전트 관리 관련
    selected_agents: Annotated[List[str], merge_lists_unique] = []
    execution_order: Annotated[List[str], merge_lists_unique] = []
    
    # 작물 관련
    crop_info: Annotated[List[str], operator.add] = []
    selected_crop: Annotated[List[str], merge_lists_unique] = []
    crop_recommendation_failed: Annotated[List[bool], merge_lists_unique] = []
    
    # 결과 및 출력
    agent_results: Annotated[Dict[str, str], merge_dicts] = {}
    output: Annotated[List[str], operator.add] = []
    
    # 세션 및 메타데이터
    session: Annotated[Dict[str, any], merge_dicts] = {}
    artifacts: Annotated[Dict[str, any], merge_dicts] = {}
    routing: Annotated[Dict[str, any], merge_dicts] = {}
    
    # 에러 및 디버깅
    error_info: Annotated[Dict[str, str], merge_dicts] = {}
    
    # 외부 데이터
    milvus_context: Annotated[str, merge_dicts] = ""
    milvus_data: Annotated[Dict[str, any], merge_dicts] = {}


class ErrorHandler:
    """통합 에러 처리 클래스"""
    
    def __init__(self):
        self.logger = LoggingConfig.get_logger("ErrorHandler")
    
    def handle_error(self, state: RouterState, error: Exception, context: str = "") -> RouterState:
        """에러 처리 및 상태 복구"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": str(time.time())
        }
        
        state["error_info"] = error_info
        self.logger.error(f"에러 발생 [{context}]: {error}")
        self.logger.debug(f"에러 상세: {error_info}")
        
        # 에러 복구 시도
        try:
            # 기본 응답으로 복구
            state["output"] = [f"죄송합니다. {context} 처리 중 오류가 발생했습니다. 다시 시도해주세요."]
            self.logger.info("기본 응답으로 복구 완료")
        except Exception as recovery_error:
            self.logger.critical(f"복구 실패: {recovery_error}")
            state["output"] = ["시스템 오류가 발생했습니다. 관리자에게 문의해주세요."]
        
        return state
    
    @staticmethod
    def safe_execute(func, *args, **kwargs):
        """안전한 함수 실행 래퍼"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 로거를 직접 생성 (static method이므로)
            logger = LoggingConfig.get_logger("ErrorHandler")
            logger.error(f"함수 실행 중 오류: {e}")
            return None


class StateManager:
    """상태 관리 클래스"""
    
    @staticmethod
    def load_state(state: RouterState) -> RouterState:
        """그래프 시작 시 상태 초기화 - supervisor에서 query와 selected_crop만 받음"""
        logger = LoggingConfig.get_logger("StateManager")
        logger.debug("상태 로딩 시작")
        
        # supervisor에서 전달받은 상태들 백업 (query, selected_crop, milvus_data)
        received_query = state.get("query", [])
        received_selected_crop = state.get("selected_crop", [])
        received_milvus_data = state.get("milvus_data", {})
        
        logger.debug(f"Supervisor에서 받은 상태들 - query: {received_query}, selected_crop: {received_selected_crop}")
        logger.debug(f"milvus_data: {'연결됨' if received_milvus_data.get('connection_status') else '연결 안됨'}")
        
        # 세션 정보만 설정 (나머지는 RouterState 기본값 사용)
        state["session"] = {"loaded": True, "new_question": True}
        
        # supervisor에서 받은 상태들 복원
        if received_query:
            state["query"] = received_query if isinstance(received_query, list) else [received_query]
        else:
            state["query"] = []
            
        if received_selected_crop:
            state["selected_crop"] = received_selected_crop if isinstance(received_selected_crop, list) else [received_selected_crop]
        else:
            state["selected_crop"] = []
        
        if received_milvus_data:
            state["milvus_data"] = received_milvus_data
        else:
            state["milvus_data"] = {}
        
        logger.info("상태 로딩 완료")
        logger.debug(f"최종 상태 - query: {state['query']}, selected_crop: {state['selected_crop']}")
        
        return state
    
    @staticmethod
    def persist_state(state: RouterState) -> RouterState:
        """그래프 종료 후 상태 저장"""
        logger = LoggingConfig.get_logger("StateManager")
        logger.debug("상태 저장 완료")
        return state
    
    @staticmethod
    def validate_state(state: RouterState) -> bool:
        """상태 유효성 검증"""
        logger = LoggingConfig.get_logger("StateManager")
        required_fields = ["query", "selected_agents", "agent_results"]
        
        for field in required_fields:
            if field not in state:
                logger.error(f"상태 검증 실패: 필수 필드 '{field}' 누락")
                return False
        
        return True


class WorkflowBuilder:
    """워크플로우 구성 클래스"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
    
    def create_workflow(self):
        """완전한 조건부 분기 워크플로우"""
        workflow = StateGraph(RouterState)
        
        # 노드 추가
        workflow.add_node("load_state", RunnableLambda(StateManager.load_state))
        workflow.add_node("persist_state", RunnableLambda(StateManager.persist_state))
        workflow.add_node("input", self.orchestrator.node_input)
        workflow.add_node("agent_select", self.orchestrator.node_agent_select)
        workflow.add_node("crop_recommend", self.orchestrator.node_crop_recommend)
        
        # 개별 에이전트 노드들 (LangGraph가 자동으로 병렬 처리)
        workflow.add_node("crop_grow_agent", self.orchestrator.node_crop_grow_agent)
        workflow.add_node("disaster_agent", self.orchestrator.node_disaster_agent)
        workflow.add_node("weather_agent", self.orchestrator.node_weather_agent)
        workflow.add_node("sales_agent", self.orchestrator.node_sales_agent)
        
        workflow.add_node("merge_output", self.orchestrator.node_merge_output)
        
        # 기본 엣지 - load_state → input → agent_select
        workflow.add_edge("load_state", "input")
        workflow.add_edge("input", "agent_select")
        
        # agent_select에서 조건부 분기 - 작물추천 우선 처리
        def agent_select_branch_condition(state):
            selected_agents = state.get("selected_agents", [])
            
            # 작물추천_agent가 선택된 경우 - 먼저 작물추천 실행
            if "작물추천_agent" in selected_agents:
                return "crop_recommend"
            # 다른 에이전트만 선택된 경우 - 바로 병렬 처리
            elif len(selected_agents) > 0:
                return "parallel_agents"
            # 아무것도 선택되지 않은 경우 - 작물추천으로 기본 설정
            else:
                return "crop_recommend"

        workflow.add_conditional_edges(
            "agent_select",
            agent_select_branch_condition,
            {
                "crop_recommend": "crop_recommend",
                "parallel_agents": "parallel_agents"
            }
        )
        
        # parallel_agents 노드 추가 (다른 에이전트들만 병렬 실행)
        workflow.add_node("parallel_agents", self.orchestrator.node_parallel_agents)
        
        # parallel_agents에서 각 에이전트로 분기
        workflow.add_edge("parallel_agents", "crop_grow_agent")
        workflow.add_edge("parallel_agents", "disaster_agent")
        workflow.add_edge("parallel_agents", "weather_agent")
        workflow.add_edge("parallel_agents", "sales_agent")
        
        # crop_recommend에서 조건부 분기 - 작물추천 후 병렬 처리
        def crop_recommend_branch_condition(state):
            # 작물추천 실패 시 바로 종료
            failed_list = state.get("crop_recommendation_failed", [])
            if failed_list and True in failed_list:
                return "persist_state"
            
            # 다른 에이전트가 있는지 확인
            other_agents = [agent for agent in state.get("selected_agents", []) if agent != "작물추천_agent"]
            if len(other_agents) > 0:
                # 작물추천 완료 후 나머지 에이전트들을 병렬로 실행
                return "parallel_agents"
            else:
                return "merge_output"
        
        workflow.add_conditional_edges(
            "crop_recommend",
            crop_recommend_branch_condition,
            {
                "parallel_agents": "parallel_agents",
                "merge_output": "merge_output",
                "persist_state": "persist_state"
            }
        )
        
        # LangGraph fan-in: 모든 에이전트 노드에서 병합 노드로
        workflow.add_edge("crop_grow_agent", "merge_output")
        workflow.add_edge("disaster_agent", "merge_output")
        workflow.add_edge("weather_agent", "merge_output")
        workflow.add_edge("sales_agent", "merge_output")

        
        # 병합 노드에서 종료
        workflow.add_edge("merge_output", "persist_state")
        workflow.add_edge("persist_state", END)
        
        workflow.set_entry_point("load_state")
        
        return workflow.compile()


def signal_handler(signum, frame):
    """키보드 인터럽트 시그널 핸들러"""
    _interrupt_flag.set()
    # KeyboardInterrupt 예외 발생으로 정상적인 종료 처리
    raise KeyboardInterrupt()

class Farmer:
    """농업 오케스트레이터 - Supervisor에게 병합 가능한 구조"""
    
    def __init__(self, user_id: str = "default_user", service: str = "farmer", chat_id: str = "default_chat"):
        """Farmer 클래스 초기화 - 지연 로딩 최적화"""
        # 사용자 식별자 설정
        self.user_id = user_id
        self.service = service
        self.chat_id = chat_id
        
        # 설정 클래스 사용
        self.config = FarmerConfig()
        
        # LLM 설정 (필수 - 즉시 로딩)
        llm_config = self.config.get_llm_config()
        self.llm = ChatOpenAI(**llm_config)
        
        # 로거 초기화
        self.logger = LoggingConfig.get_logger("Farmer")
        
        # 관리자 클래스들 초기화
        self.agent_manager = AgentManager()
        self.error_handler = ErrorHandler()
        self.workflow_builder = WorkflowBuilder(self)
        
        # 메모리 시스템 초기화 (가벼운 작업)
        self._init_memory_system()
        
        # 워크플로우 그래프 생성 (지연 로딩으로 변경)
        self._graph = None  # 지연 로딩용
        
        
        # 시그널 핸들러 등록 (메인 스레드에서만)
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal_handler)
        
        self.logger.info("Farmer 초기화 완료 (지연 로딩 모드)")
    
    def _init_memory_system(self):
        """메모리 시스템 초기화 - Main에서 중앙집중식 관리"""
        self.logger.debug("메모리 시스템: Main에서 중앙집중식 관리")
    
    @property
    def graph(self):
        """워크플로우 그래프 지연 로딩"""
        if self._graph is None:
            self.logger.debug("Farmer 워크플로우 그래프 생성 중...")
            self._graph = self.workflow_builder.create_workflow()
            self.logger.info("Farmer 워크플로우 그래프 생성 완료")
        return self._graph
    
    
    def clear_cache(self):
        """에이전트 함수 캐시 정리"""
        self.agent_manager.clear_cache()
    
    def get_loaded_agents(self):
        """현재 로드된 에이전트 목록 반환"""
        return self.agent_manager.get_loaded_agents()
    
    def load_state(self, state: RouterState) -> RouterState:
        """그래프 시작 시 상태 초기화 - StateManager 위임"""
        return StateManager.load_state(state)
    
    def persist_state(self, state: RouterState) -> RouterState:
        """그래프 종료 후 상태 저장 - StateManager 위임"""
        return StateManager.persist_state(state)
    
    def _handle_error(self, state: RouterState, error: Exception, context: str = "") -> RouterState:
        """에러 처리 및 상태 복구 - ErrorHandler 위임"""
        return self.error_handler.handle_error(state, error, context)
    
    def _get_agent_function(self, agent_name: str):
        """에이전트 함수를 지연 로딩으로 가져오기 - AgentManager 위임"""
        return self.agent_manager.get_agent_function(agent_name)
    
    
    def invoke(self, state: dict, config: Optional[Dict] = None) -> dict:
        """
        Supervisor에서 호출되는 메인 실행 함수 - LangGraph 워크플로우를 따라 실행
        
        Args:
            state: Supervisor에서 전달받은 상태 딕셔너리
                  - query: 사용자 질문 (필수)
                  - 기타 필요한 상태 정보들
            config: LangGraph 설정 (선택사항)
        
        Returns:
            dict: 실행 결과
                - output: 최종 응답
                - selected_agents: 선택된 에이전트들
                - agent_results: 각 에이전트별 결과
        """
        if config is None:
            config = {"configurable": {"thread_id": f"{self.user_id}_{self.chat_id}"}}
        
        try:
            # 인터럽트 플래그 초기화 (새로운 요청 시작)
            _interrupt_flag.clear()
            
            # RouterState로 변환하고 Supervisor에서 전달받은 상태들을 반영 (query, selected_crop만)
            router_state = RouterState()
            
            # Supervisor에서 받는 상태들
            router_state["query"] = [state.get("query", "")] if state.get("query") else []
            
            if state.get("selected_crop"):
                router_state["selected_crop"] = [state.get("selected_crop")] if isinstance(state.get("selected_crop"), str) else state.get("selected_crop", [])
            else:
                router_state["selected_crop"] = []
            
            # MilvusDB 연결 정보 전달
            if state.get("milvus_data"):
                router_state["milvus_data"] = state.get("milvus_data")
            else:
                router_state["milvus_data"] = {}
            
            self.logger.info(f"Farmer 실행 시작: {state.get('query', '')}")
            
            # LangGraph 워크플로우를 따라 실행
            result = self.graph.invoke(router_state, config)
            
            # 결과 반환
            return {
                "output": result.get("output", [""])[0] if result.get("output") else "",
                "selected_agents": result.get("selected_agents", []),
                "agent_results": result.get("agent_results", {}),
                "selected_crop": result.get("selected_crop", [""])[0] if result.get("selected_crop") else "",
                "status": "success",
                "error_info": result.get("error_info", {})
            }
            
        except Exception as e:
            self.logger.error(f"Farmer 오케스트레이터 실행 중 오류: {e}")
            traceback.print_exc()
            return {
                "output": f"Farmer 오케스트레이터 실행 중 오류가 발생했습니다: {e}",
                "selected_agents": [],
                "agent_results": {},
                "selected_crop": "",
                "status": "error",
                "error": str(e),
                "error_info": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "context": "main_invoke"
                }
            }

    @lru_cache(maxsize=128)
    def simple_agent_selector(self, user_question):
        """
        사용자 질문을 분석하여 필요한 에이전트를 선택하는 함수
        """
        selection_prompt = f"""
        다음 질문을 분석하여 필요한 에이전트를 선택해주세요.
        
        [에이전트 역할 및 설명]
        1) 작물추천_agent: {self.config.AGENT_DESCRIPTIONS["작물추천_agent"]}
        
        2) 작물재배_agent: {self.config.AGENT_DESCRIPTIONS["작물재배_agent"]}
        
        3) 재해_agent: {self.config.AGENT_DESCRIPTIONS["재해_agent"]}
        
        4) 날씨_agent: **'날씨', '기상' 이라는 키워드가 포함되어 있을 경우에만 선택**
        
        5) 판매처_agent: {self.config.AGENT_DESCRIPTIONS["판매처_agent"]}
        
        질문: "{user_question}"
        
        [응답 규칙]
        - 에이전트가 1개만 필요한 경우: 에이전트명만 선택
        - 에이전트가 2개 이상 필요한 경우: 각 에이전트가 담당할 질문 부분도 함께 분류
        
        다음 JSON 형식으로 답변해주세요:
        
        [1개 에이전트인 경우]
        {{
            "selected_agents": ["에이전트명"],
            "execution_order": ["에이전트명"]
        }}
        
        [2개 이상 에이전트인 경우]
        {{
            "selected_agents": ["에이전트명1", "에이전트명2"],
            "question_parts": {{
                "에이전트명1": "담당할 질문 부분",
                "에이전트명2": "담당할 질문 부분"
            }},
            "execution_order": ["에이전트명1", "에이전트명2"]
        }}
        """
        
        try:
            result = self.llm.invoke(selection_prompt)
            
            # JSON 부분 추출
            json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
            if json_match:
                parsed_result = json.loads(json_match.group())
                selected_agents = parsed_result.get("selected_agents", [])
                
                # 에이전트가 1개인 경우
                if len(selected_agents) == 1:
                    return {
                        "selected_agents": selected_agents,
                        "question_parts": None,  # 질문 분류 없음
                        "execution_order": parsed_result["execution_order"]
                    }
                # 에이전트가 2개 이상인 경우
                elif len(selected_agents) >= 2:
                    # question_parts가 있는지 확인
                    if "question_parts" in parsed_result:
                        return parsed_result
                    else:
                        # question_parts가 없는 경우 기본값 사용
                        self.logger.warning("질문 분류 누락 - 기본값 사용")
                        question_parts = {agent: user_question for agent in selected_agents}
                        return {
                            "selected_agents": selected_agents,
                            "question_parts": question_parts,
                            "execution_order": parsed_result["execution_order"]
                        }
                else:
                    # 에이전트가 0개인 경우 - 작물추천_agent로 기본 설정
                    return {
                        "selected_agents": ["작물추천_agent"],
                        "question_parts": None,
                        "execution_order": ["작물추천_agent"]
                    }
            else:
                # JSON 파싱 실패 시 기본값 - 작물추천_agent로 설정
                return {
                    "selected_agents": ["작물추천_agent"],
                    "question_parts": None,
                    "execution_order": ["작물추천_agent"]
                }
        except Exception as e:
            self.logger.error(f"에이전트 선택 실패: {e}")
            return {
                "selected_agents": ["작물추천_agent"],
                "question_parts": None,
                "execution_order": ["작물추천_agent"]
            }


    @traceable(name="node_input")
    def node_input(self, state: RouterState) -> RouterState:
        """사용자 질문에서 작물명을 추출하는 노드"""
        self.logger.info("질문 분석 및 작물명 추출 시작")
        
        # 현재 상태에서 query 가져오기
        query = state.get("query", [])
        user_input = query[0] if query and len(query) > 0 else ""
        
        if not user_input:
            self.logger.error("질문이 없습니다")
            return state
        
        self.logger.debug(f"분석할 질문: {user_input}")
        
        # 새로운 질문 플래그 설정
        state.setdefault("session", {})
        state["session"]["new_question"] = True
        
        # 질문에서 작물명 추출
        extracted_crop = self.extract_crop_from_question(user_input)
        
        if extracted_crop and extracted_crop.strip():
            self.logger.info(f"추출된 작물명: '{extracted_crop}'")
            self.logger.debug(f"기존 작물 상태 '{state.get('selected_crop', [])}' → '{[extracted_crop.strip()]}' 대체")
            # 기존 selected_crop을 완전히 클리어하고 새 값으로 대체
            state["selected_crop"].clear()  # 기존 리스트 완전 클리어
            state["selected_crop"].append(extracted_crop.strip())  # 새 값만 추가
        else:
            self.logger.debug("질문에서 구체적인 작물명을 찾을 수 없습니다")
            # 기존 selected_crop 유지 (있는 경우) 또는 빈 리스트
            if not state.get("selected_crop"):
                state["selected_crop"] = []
        
        self.logger.debug(f"최종 작물 상태: {state['selected_crop']}")
        self.logger.info("새로운 질문 처리 시작")
        
        return state
    
    def extract_crop_from_question(self, question: str, crop_recommendations: str = None) -> str:
        """
        질문에서 작물명을 추출하거나 작물추천 결과에서 작물을 선택하는 통합 함수
        
        Args:
            question: 사용자 질문
            crop_recommendations: 작물추천 결과 (선택사항)
        
        Returns:
            추출된 작물명
        """
        self.logger.debug("LLM으로 작물명 추출 중...")
        
        if crop_recommendations:
            # 작물추천 결과에서 작물 선택
            extraction_prompt = f"""
            다음은 작물추천 에이전트가 추천한 작물들입니다. 
            사용자의 질문과 상황을 고려하여 상세 분석할 작물 하나를 선택해주세요.
            
            [사용자 질문]
            {question}
            
            [추천 작물 목록]
            {crop_recommendations}
            
            [요구사항]
            - 작물명만 작성 (예: 무, 토마토, 고추, 오이)
            - 설명이나 문장은 절대 포함하지 말 것
            - 한 단어로 된 작물명만
            - 작물을 찾을 수 없으면 "없음"이라고만 답변
            - 작물 추천 결과에 있는 맨 처음 작물을 선택해줘
            
            상세 분석할 작물: """
        else:
            # 질문에서 직접 작물명 추출
            extraction_prompt = f"""
            사용자의 질문에서 구체적인 작물명을 추출해주세요.
            
            [추출 규칙]
            1. 구체적인 작물명만 추출 (예: 토마토, 상추, 무, 배추, 고구마, 감자 등)
            2. 일반적인 용어는 제외 (예: 작물, 채소, 농작물, 식물 등)
            3. 작물명이 없으면 "없음"만 답변
            4. 작물명만 한 단어로 답변 (설명이나 문장 금지)
            
            [예시]
            - "토마토 키우는 방법 알려줘" → "토마토"
            - "알배기배추 가격이 궁금해" → "알배기배추"
            - "감자랑 고구마 중에 뭐가 좋을까?" → "감자"
            - "어떤 작물을 심을까요?" → "없음"
            - "농작물 재배 방법" → "없음"
            
            질문: "{question}"
            
            추출된 작물명:"""
        
        try:
            result = self.llm.invoke(extraction_prompt)
            extracted_crop = result.content.strip()
            
            # "없음"이거나 빈 문자열인 경우 빈 문자열 반환
            if extracted_crop in ["없음", "", "None", "null"]:
                self.logger.debug("추출 실패: 구체적인 작물명 없음")
                return ""
            
            # 간단한 정리 (첫 번째 단어만, 줄바꿈/구두점 제거)
            cleaned_crop = extracted_crop.split('\n')[0].split('.')[0].split(',')[0].strip()
            
            # 특정 작물명 정규화 (파 관련)
            if cleaned_crop in ["경량파", "파"]:
                cleaned_crop = "대파"
                self.logger.debug(f"작물명 정규화: '{extracted_crop}' → '{cleaned_crop}'")
            
            self.logger.info(f"작물명 추출 성공: '{cleaned_crop}'")
            return cleaned_crop
            
        except Exception as e:
            self.logger.error(f"LLM 작물명 추출 실패: {e}")
            return ""

    @traceable(name="node_agent_select")
    def node_agent_select(self, state: RouterState) -> RouterState:
        self.logger.info("에이전트 선택 시작")
        
        # 안전하게 query 가져오기
        query = state.get("query", [])
        query_text = query[0] if query and len(query) > 0 else ""
        
        # 기존 복잡한 로직을 단순화된 함수로 교체
        result = self.simple_agent_selector(query_text)
        
        # 안전하게 상태 업데이트
        state["selected_agents"] = result["selected_agents"] if isinstance(result["selected_agents"], list) else [result["selected_agents"]]
        state["question_parts"] = result.get("question_parts", {}) if result.get("question_parts") is not None else {}
        state["execution_order"] = result["execution_order"] if isinstance(result["execution_order"], list) else [result["execution_order"]]
        
        self.logger.info(f"선택된 에이전트: {state['selected_agents']}")
        
        return state

    @traceable(name="node_crop_recommend")
    def node_crop_recommend(self, state: RouterState) -> RouterState:
        if "작물추천_agent" not in state.get("selected_agents", []):
            return state
        
        self.logger.info("작물추천_agent 실행 시작")
        
        # question_parts가 None인 경우 안전하게 처리
        question_parts = state.get("question_parts", {})
        if not question_parts:
            # 단일 에이전트인 경우 원본 질문 사용
            question_part = state["query"][0] if state["query"] else ""
            self.logger.debug(f"단일 에이전트 - 원본 질문 사용")
        else:
            # 다중 에이전트인 경우 분류된 질문 사용
            question_part = question_parts.get("작물추천_agent", state["query"][0] if state["query"] else "")
            self.logger.debug(f"다중 에이전트 - 분류된 질문 사용")
        
        self.logger.debug(f"담당 질문: {question_part}")
        
        # 작물추천 에이전트 실행
        self.logger.info("작물추천_agent 실행")
        try:
            agent_func = self._get_agent_function("작물추천_agent")
            if agent_func:
                agent_state = {
                    "query": question_part,
                    "milvus_data": state.get("milvus_data", {}),
                }
                agent_result = agent_func(agent_state)
                answer = agent_result.get("agent_answer", "답변 생성 실패")
            else:
                answer = "작물추천_agent 실행 함수가 연결되어 있지 않습니다."
        except Exception as e:
            answer = f"작물추천_agent 실행 중 오류: {e}"
        
        self.logger.debug(f"응답 미리보기: {answer[:100]}...")
        
        # 작물추천 결과에서 하나의 작물 선택
        selected_crop = self.extract_crop_from_question(question_part, answer)
        
        state["crop_info"] = [answer]
        # 기존 selected_crop을 완전히 클리어하고 새 값으로 대체
        state["selected_crop"].clear()  # 기존 리스트 완전 클리어
        state["selected_crop"].append(selected_crop)  # 새 값만 추가
        
        # 작물 추출 실패 시 fallback 플래그 설정
        if not selected_crop or selected_crop.strip() == "":
            self.logger.debug(f"\n작물 추출 실패")
            state["crop_recommendation_failed"] = [True]
            # 최종 출력을 작물추천 답변으로 설정
            state["output"] = [answer]
        else:
            self.logger.debug(f"작물 추출 완료")
            state["crop_recommendation_failed"] = []
        
        return state

    # 각 에이전트별로 개별 노드 생성
    @traceable(name="node_crop_grow_agent")
    def node_crop_grow_agent(self, state: RouterState) -> RouterState:
        """작물재배_agent 전용 노드 - LangGraph 병렬 처리"""
        selected_agents = state.get("selected_agents", [])
        if not selected_agents or "작물재배_agent" not in selected_agents:
            return state
        
        self.logger.info("작물재배_agent 실행")
        
        # 질문 부분 가져오기
        question_parts = state.get("question_parts", {})
        if question_parts and "작물재배_agent" in question_parts:
            question_part = question_parts["작물재배_agent"]
        else:
            question_part = state["query"][0] if state["query"] else ""
        
        
        # 작물재배_agent 전용 작물명 처리
        selected_crop = state.get("selected_crop", [""])[0] if state.get("selected_crop") else ""
        if selected_crop and selected_crop not in question_part:
            question_part = f"{selected_crop} {question_part}"
            self.logger.debug(f"담당 질문: {question_part}")

        # 에이전트 실행
        try:
            agent_func = self._get_agent_function("작물재배_agent")
            if agent_func:
                agent_state = {
                    "query": question_part,
                    "milvus_data": state.get("milvus_data", {}),
                }
                agent_result = agent_func(agent_state)
                answer = agent_result.get("agent_answer", "답변 생성 실패")
            else:
                answer = "작물재배_agent 실행 함수가 연결되어 있지 않습니다."
        except Exception as e:
            answer = f"작물재배_agent 실행 중 오류: {e}"
        
        # 전용 키에 답변 저장
        state["agent_results"]["작물재배_agent"] = answer
        
        self.logger.info("작물재배_agent 완료")
        self.logger.debug(f"응답 미리보기: {answer[:100]}...")
        return state

    @traceable(name="node_disaster_agent")
    def node_disaster_agent(self, state: RouterState) -> RouterState:
        """재해_agent 전용 노드 - LangGraph 병렬 처리"""
        selected_agents = state.get("selected_agents", [])
        if not selected_agents or "재해_agent" not in selected_agents:
            return state
        
        self.logger.info("재해_agent 실행")
        
        # 질문 부분 가져오기
        question_parts = state.get("question_parts", {})
        if question_parts and "재해_agent" in question_parts:
            question_part = question_parts["재해_agent"]
        else:
            question_part = state["query"][0] if state["query"] else ""
        
        
        # 재해_agent 전용 작물명 처리
        selected_crop = state.get("selected_crop", [""])[0] if state.get("selected_crop") else ""
        if selected_crop and selected_crop not in question_part:
            question_part = f"{selected_crop} 재배 중, {question_part}"
            self.logger.debug(f"담당 질문: {question_part}")
        
        # 에이전트 실행
        try:
            agent_func = self._get_agent_function("재해_agent")
            if agent_func:
                agent_state = {
                    "query": question_part,
                    "milvus_data": state.get("milvus_data", {}),
                }
                agent_result = agent_func(agent_state)
                answer = agent_result.get("agent_answer", "답변 생성 실패")
            else:
                answer = "재해_agent 실행 함수가 연결되어 있지 않습니다."
        except Exception as e:
            answer = f"재해_agent 실행 중 오류: {e}"
        
        # 전용 키에 답변 저장
        state["agent_results"]["재해_agent"] = answer
        
        self.logger.info("재해_agent 완료")
        self.logger.debug(f"응답 미리보기: {answer[:100]}...")
        return state

    @traceable(name="node_sales_agent")
    def node_sales_agent(self, state: RouterState) -> RouterState:
        """판매처_agent 전용 노드 - LangGraph 병렬 처리"""
        selected_agents = state.get("selected_agents", [])
        if not selected_agents or "판매처_agent" not in selected_agents:
            return state
        
        self.logger.info("판매처_agent 실행")
        
        # 질문 부분 가져오기
        question_parts = state.get("question_parts", {})
        if question_parts and "판매처_agent" in question_parts:
            question_part = question_parts["판매처_agent"]
        else:
            question_part = state["query"][0] if state["query"] else ""
        
        
        # 판매처_agent 전용 작물명 처리
        selected_crop = state.get("selected_crop", [""])[0] if state.get("selected_crop") else ""
        if selected_crop and selected_crop not in question_part:
            question_part = f"{selected_crop} {question_part}"
            self.logger.debug(f"담당 질문: {question_part}")
        
        # 에이전트 실행
        try:
            agent_func = self._get_agent_function("판매처_agent")
            if agent_func:
                agent_state = {
                    "query": question_part,
                    "milvus_data": state.get("milvus_data", {}),
                }
                agent_result = agent_func(agent_state)
                answer = agent_result.get("agent_answer", "답변 생성 실패")
            else:
                answer = "판매처_agent 실행 함수가 연결되어 있지 않습니다."
        except Exception as e:
            answer = f"판매처_agent 실행 중 오류: {e}"
        
        # 전용 키에 답변 저장
        state["agent_results"]["판매처_agent"] = answer
        
        self.logger.info("판매처_agent 완료")
        self.logger.debug(f"응답 미리보기: {answer[:100]}...")
        return state

    @traceable(name="node_weather_agent")
    def node_weather_agent(self, state: RouterState) -> RouterState:
        """날씨_agent 전용 노드 - LangGraph 병렬 처리"""
        selected_agents = state.get("selected_agents", [])
        if not selected_agents or "날씨_agent" not in selected_agents:
            return state
        
        self.logger.info("날씨_agent 실행")
        
        # 질문 부분 가져오기
        question_parts = state.get("question_parts", {})
        if question_parts and "날씨_agent" in question_parts:
            question_part = question_parts["날씨_agent"]
        else:
            question_part = state["query"][0] if state["query"] else ""
        
        self.logger.debug(f"담당 질문: {question_part}")
        
        # 날씨_agent는 작물명 처리가 필요 없으므로 원본 질문 그대로 사용
        # (날씨는 지역과 시간이 중요하므로)
        
        # 에이전트 실행
        try:
            agent_func = self._get_agent_function("날씨_agent")
            if agent_func:
                agent_state = {
                    "query": question_part,
                    "milvus_data": state.get("milvus_data", {}),
                }
                agent_result = agent_func(agent_state)
                answer = agent_result.get("agent_answer", "답변 생성 실패")
            else:
                answer = "날씨_agent 실행 함수가 연결되어 있지 않습니다."
        except Exception as e:
            answer = f"날씨_agent 실행 중 오류: {e}"
        
        # 전용 키에 답변 저장
        state["agent_results"]["날씨_agent"] = answer
        
        self.logger.info("날씨_agent 완료")
        self.logger.debug(f"응답 미리보기: {answer[:100]}...")
        return state

    @traceable(name="node_parallel_agents")
    def node_parallel_agents(self, state: RouterState) -> RouterState:
        """다른 에이전트들만 병렬 실행하는 노드"""
        selected_agents = state.get("selected_agents", [])
        other_agents = [agent for agent in selected_agents if agent != "작물추천_agent"]
        
        if not other_agents:
            return state
        
        self.logger.info("병렬 에이전트 실행 시작")
        self.logger.debug(f"실행될 에이전트: {other_agents}")
        
        # LangGraph가 자동으로 병렬 처리하므로 상태만 반환
        # 실제 에이전트 실행은 개별 노드들에서 처리됨
        return state

    @traceable(name="node_merge_output")
    def node_merge_output(self, state: RouterState) -> RouterState:
        self.logger.info("최종 응답 병합 시작")
        
        # 각 에이전트 결과 수집
        agent_results = {}
        
        if state.get("crop_info"):
            agent_results["작물추천_agent"] = state["crop_info"][0] if state["crop_info"] else ""

        if state.get("agent_results"):
            agent_results.update(state["agent_results"])
        
        # 실행 요약 출력
        selected_agents = state.get("selected_agents", [])
        
        output = ""
        
        # 에이전트가 하나뿐인 경우 단순 처리
        if len(selected_agents) == 1:
            agent = selected_agents[0]
            if agent in agent_results:
                result = agent_results[agent]
                output = str(result)
                self.logger.info(f"단일 에이전트 응답 완료: {agent}")
            else:
                output = f"{agent} 실행 결과를 찾을 수 없습니다."
                self.logger.error(f"{agent} 응답 없음")
        else:
            # 여러 에이전트가 있는 경우 기존 로직 유지
            # 작물추천 결과가 있으면 먼저 표시
            if state.get("crop_info"):
                output += f"[작물추천 결과]\n{state['crop_info']}\n"
                
                # 선택된 작물 강조 표시
                if state.get("selected_crop"):
                    output += f"\n[상세 분석 작물]\n{state['selected_crop']}\n"

            
            # 다른 에이전트들의 답변 표시
            for agent, answer in agent_results.items():
                if agent != "작물추천_agent":  # 이미 표시됨
                    # 에이전트 결과 추가
                    output += f"[{agent} 결과]\n{str(answer)}\n"
        
        # 최종 출력 처리
        merged_output = str(output).strip()
        
        # 에이전트가 하나뿐인 경우 LLM 요약 생략
        if len(selected_agents) == 1:
            state["output"] = [merged_output]
            return state
        
        # 여러 에이전트가 있는 경우에만 LLM 요약
        self.logger.info("LLM 요약 시작...")
        summary_prompt = (
            f"""
            아래는 여러 농업 agent의 답변입니다.
            사용자는 농작물을 키우는 입장으로 당신에게 조언을 구하고 있습니다.
            사용자에게 최대한 자세하고 상세하게 한국어로 알려주세요.
            작물 추천, 재배 방법, 재해, 판매처, 날씨 순으로 자연스럽게 대화하듯 연결해주세요. (**실행되지 않은 agent는 생략**)
            \n\n
            {merged_output}
            \n\n
            [중요 규칙 - 반드시 지켜주세요]
            - 답변으로 받은 정보는 **변경하거나 빼먹지 말고 출력** (특히 오늘인지 내일인지 등 시간 정보)
            - "정보가 없다"는 것도 빼지 말고 아쉽다는 식으로 없다고 대답
            - 내용 안에 agent 이름을 넣지 말고 대화하는 것처럼 사용자에게 대답
            - 농작물의 **직매장 등을 찾고 있다면** 구매하는 것이 아닌 판매하려 한다는 것을 알아둬
            - **절대 마크다운 형식을 사용하지 마세요**: **, ##, ###, *, -, [], (), ``` 등 모든 마크다운 문법 금지
            - **순수한 텍스트만 사용하세요**: 굵은 글씨, 제목, 리스트, 코드 블록 등 모두 일반 텍스트로 변환
            - 마지막에는 사용자에게 다른 질문을 유도하는 문장 추가
            """
        )
        
        try:
            summary = self.llm.invoke(summary_prompt)
            self.logger.info(f"LLM 요약 완료")
        except Exception as e:
            summary = f"요약 중 오류: {e}"
            self.logger.error(f"LLM 요약 실패: {e}")
        
        # 마크다운 제거 후처리
        raw_content = summary.content.strip() if hasattr(summary, 'content') else summary.strip()
        cleaned_content = self._remove_markdown(raw_content)
        
        state["output"] = [cleaned_content]
        self.logger.debug(f"--- 최종 응답 --- \n{state['output']}")

        return state
    
    def _remove_markdown(self, text: str) -> str:
        """마크다운 형식을 제거하는 함수"""

        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # 코드 블록 제거 (```...```)
        text = re.sub(r'`([^`]+)`', r'\1', text)  # 인라인 코드 제거 (`...`)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # 굵은 글씨 제거 (**...**)
        text = re.sub(r'__([^_]+)__', r'\1', text)  # 굵은 글씨 제거 (__...__)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # 기울임 글씨 제거 (*...*)
        text = re.sub(r'_([^_]+)_', r'\1', text)  # 기울임 글씨 제거 (_..._)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # 제목 제거 (# ## ###)
        text = re.sub(r'^[-*]{3,}$', '', text, flags=re.MULTILINE)  # 수평선 제거 (--- ***)

        return text