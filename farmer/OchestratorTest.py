from sentence_transformers import SentenceTransformer, util
import requests
import re
import json
from langgraph.graph import StateGraph, END

from langchain_core.runnables import RunnableLambda
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Dict, Optional
from tavily import TavilyClient
import operator
from langsmith import traceable
from dotenv import load_dotenv
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import traceback
load_dotenv()

# 병합 함수들을 먼저 정의 (RouterState에서 사용하기 위해)
def merge_dicts(left: dict, right: dict) -> dict:
    """딕셔너리 병합 함수 - LangGraph용"""
    if not left:
        return right or {}
    if not right:
        return left or {}
    merged = left.copy()
    merged.update(right)
    return merged

def merge_lists_unique(left: list, right: list) -> list:
    """리스트 병합 함수 - 중복 제거 - LangGraph용"""
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

# RouterState 클래스를 Farmer 클래스 외부로 이동
class RouterState(dict):
    query: Annotated[List[str], operator.add] = ""
    selected_agents: Annotated[List[str], merge_lists_unique] = []
    question_parts: Annotated[Dict[str, str], merge_dicts] = {}
    execution_order: Annotated[List[str], merge_lists_unique] = []
    crop_info: Annotated[List[str], operator.add] = []
    selected_crop: Annotated[List[str], merge_lists_unique] = []
    agent_results: Annotated[Dict[str, str], merge_dicts] = {}
    output: Annotated[List[str], operator.add] = []
    session: Annotated[Dict[str, any], merge_dicts] = {}
    artifacts: Annotated[Dict[str, any], merge_dicts] = {}
    routing: Annotated[Dict[str, any], merge_dicts] = {}
    error_info: Annotated[Dict[str, str], merge_dicts] = {}

class Farmer:
    """농업 오케스트레이터 - Supervisor에게 병합 가능한 구조"""
    
    def __init__(self, user_id: str = "default_user", service: str = "farmer", chat_id: str = "default_chat"):
        """Farmer 클래스 초기화"""
        # 사용자 식별자 설정
        self.user_id = user_id
        self.service = service
        self.chat_id = chat_id
        
        # LLM 설정
        self.llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct",
                           temperature=0.8,
                           api_key=os.getenv("OPENAI_KEY1"))
        # gpt-4o-mini 사용 시 주석 해제
        # self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.8, api_key=os.getenv("OPENAI_API_KEY"))
        
        # 메모리 시스템 초기화
        self._init_memory_system()
        
        # 에이전트 함수들 로드
        self._load_agent_functions()
        
        # 워크플로우 그래프 생성
        self.graph = self.create_workflow()
        
        # 병렬 처리용 스레드 풀
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    def _init_memory_system(self):
        """메모리 시스템 초기화 - Redis만 사용"""
        # Redis 연결 상태 확인 및 메모리 시스템 초기화
        self.redis_connected = False
        try:
            # Farmer 전용 Redis 메모리 시스템 시도
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from common.short_term.farmer_redis import FarmerRedisMemory
            
            # Redis 포트를 6380으로 설정 (Docker 컨테이너 포트)
            os.environ['REDIS_PORT'] = '6380'
            self.memory = FarmerRedisMemory(
                user_id=self.user_id, 
                service=self.service, 
                chat_id=self.chat_id,
                redis_port=6380
            )
            
            # Redis 연결 테스트
            if self.memory.connected:
                test_state = {"test": "connection"}
                self.memory.save(test_state, test_state)
                self.memory.load(test_state)
                
                self.redis_connected = True
                print("✅ Farmer Redis 메모리 시스템 초기화 완료 - 대화 맥락 유지 가능")
                
                # 메모리 통계 출력
                stats = self.memory.get_memory_stats()
                print(f"📊 메모리 상태: {stats}")
            else:
                raise Exception("Redis 연결 실패")
                
        except Exception as e:
            print(f"⚠️ Redis 연결 실패: {e}")
            print("📝 메모리 시스템 없음 - 단일 질문 처리 모드")
            self.redis_connected = False
            self.memory = None
    
    def load_state(self, state: RouterState) -> RouterState:
        """그래프 시작 시 메모리에서 상태를 불러와 state에 병합"""
        # Redis 연결 상태에 따른 처리
        if self.redis_connected:
            print("[상태 초기화] Farmer Redis 메모리 시스템 - 대화 맥락 유지")
        else:
            print("[상태 초기화] Redis 미연결 - 단일 질문 처리 모드 (문맥 무시)")
            # Redis 미연결 시 항상 깨끗한 상태로 초기화
            return self._initialize_clean_state(state)
        
        session_info = state.get("session", {})
        if session_info.get("loaded"):
            return state
        
        try:
            # 메모리 시스템에서 상태 불러오기 (자동으로 맥락 포함)
            loaded = self.memory.load(state)
            loaded.setdefault("session", {})
            loaded["session"]["loaded"] = True
            loaded["session"]["redis_connected"] = self.redis_connected
            
            print("[메모리 로드] Farmer Redis 메모리에서 상태 불러오기 완료")
            
            # 대화 맥락 정보 출력
            context = self.memory.get_conversation_context(limit=3)
            if context:
                print(f"[대화 맥락] {context[:100]}...")
            
            # 작물 추천 맥락 정보 출력
            crop_context = self.memory.get_crop_recommendation_context(limit=2)
            if crop_context:
                print(f"[작물 추천 맥락] {crop_context[:100]}...")
            
            return loaded
        except Exception as e:
            print(f"⚠️ 메모리 로드 실패: {e}")
            return self._initialize_clean_state(state)
    
    def _initialize_clean_state(self, state: RouterState) -> RouterState:
        """완전히 깨끗한 상태로 초기화 (Redis 미연결 시)"""
        # 모든 상태 초기화
        state.clear()
        state["session"] = {"loaded": True, "new_question": True, "redis_connected": False}
        state["query"] = []
        state["selected_agents"] = []
        state["question_parts"] = {}
        state["execution_order"] = []
        state["crop_info"] = []
        state["selected_crop"] = []
        state["agent_results"] = {}
        state["output"] = []
        state["routing"] = {}
        state["error_info"] = {}
        return state
    
    def persist_state(self, state: RouterState) -> RouterState:
        """그래프 리프 종료 후 메모리에 반영"""
        # Redis 미연결 시 상태 저장 건너뛰기
        if not self.redis_connected:
            print("[메모리 저장] Redis 미연결 - 상태 저장 건너뛰기 (단일 질문 처리 모드)")
            return state
        
        try:
            # 메모리 시스템에 상태 저장 (맥락 포함)
            self.memory.save(state, state)
            print("[메모리 저장] Farmer Redis 메모리에 상태 저장 완료")
            
        except Exception as e:
            print(f"⚠️ Redis 상태 저장 실패: {e}")
        return state
    
    def resume_workflow(self, resume_data: str, config: Optional[Dict] = None) -> RouterState:
        """워크플로우 재개 기능 - Redis 미연결 시 사용 불가"""
        if not self.redis_connected:
            print("❌ Redis 미연결 상태 - 워크플로우 재개 불가")
            raise Exception("Redis 미연결 상태에서는 워크플로우 재개가 불가능합니다.")
        
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        try:
            print(f"🔄 워크플로우 재개 중... resume_data: {resume_data}")
            
            # LangGraph 버전에 따른 Command import 시도
            try:
                from langgraph.checkpoint.memory import Command
            except ImportError:
                try:
                    from langgraph import Command
                except ImportError:
                    try:
                        from langgraph.types import Command
                    except ImportError:
                        print("❌ Command를 import할 수 없습니다. LangGraph 버전을 확인해주세요.")
                        raise ImportError("Command import 실패")
            
            # Command(resume)을 사용하여 중단된 지점부터 재개
            resume_command = Command(resume=resume_data)
            print(f"📤 Command(resume) 전송: {resume_command}")
            
            # 그래프로 재개
            result = self.graph.invoke(resume_command, config)
            print("✅ 워크플로우 재개 완료")
            return result
        except Exception as e:
            print(f"❌ 워크플로우 재개 실패: {e}")
            print(f"🔍 오류 상세: {type(e).__name__}: {str(e)}")
            raise
    
    def _handle_error(self, state: RouterState, error: Exception, context: str = "") -> RouterState:
        """에러 처리 및 상태 복구"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": str(os.time.time() if hasattr(os, 'time') else __import__('time').time())
        }
        
        state["error_info"] = error_info
        print(f"❌ 에러 발생 [{context}]: {error}")
        print(f"🔍 에러 상세: {error_info}")
        
        # 에러 복구 시도
        try:
            # 기본 응답으로 복구
            state["output"] = [f"죄송합니다. {context} 처리 중 오류가 발생했습니다. 다시 시도해주세요."]
            print("🔄 기본 응답으로 복구 완료")
        except Exception as recovery_error:
            print(f"❌ 복구 실패: {recovery_error}")
            state["output"] = ["시스템 오류가 발생했습니다. 관리자에게 문의해주세요."]
        
        return state
    
    def clear_memory(self):
        """메모리 전체 삭제"""
        if self.redis_connected and self.memory and hasattr(self.memory, 'clear_memory'):
            self.memory.clear_memory()
            print("✅ Farmer 메모리 삭제 완료")
        else:
            print("⚠️ Redis 미연결 상태 - 메모리 삭제 불가")
    
    def get_memory_stats(self):
        """메모리 사용 통계 조회"""
        if self.redis_connected and self.memory and hasattr(self.memory, 'get_memory_stats'):
            return self.memory.get_memory_stats()
        else:
            return {"connected": False, "type": "None"}
    
    def get_conversation_context(self, limit: int = 5):
        """대화 맥락 조회"""
        if self.redis_connected and self.memory and hasattr(self.memory, 'get_conversation_context'):
            return self.memory.get_conversation_context(limit)
        else:
            return ""
    
    def get_crop_recommendation_context(self, limit: int = 3):
        """작물 추천 맥락 조회"""
        if self.redis_connected and self.memory and hasattr(self.memory, 'get_crop_recommendation_context'):
            return self.memory.get_crop_recommendation_context(limit)
        else:
            return ""

    def _load_agent_functions(self):
        """에이전트 함수들을 로드"""
        from 작물추천.crop_recommendation_agent import run as crop_recommend_run
        from 재배방법.CG_agent_edit import run as crop_cultivation_run
        from 재해대응.DisasterAgent import run as disaster_run
        from WeatherAgent import run as weather_run
        from sales.SalesAgent import run as market_run
        
        self.agent_functions = {
            "작물추천_agent": crop_recommend_run,
            "작물재배_agent": crop_cultivation_run,
            "재해_agent": disaster_run,
            "날씨_agent": weather_run,
            "판매처_agent": market_run
        }
    
    def invoke(self, state: dict, config: Optional[Dict] = None) -> dict:
        """
        Supervisor에서 호출되는 메인 실행 함수
        
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
            # RouterState로 변환
            router_state = RouterState()
            router_state["query"] = [state.get("query", "")]
            
            # 체크포인터와 함께 그래프 실행
            result = self.graph.invoke(router_state, config)
            
            # 결과 반환
            return {
                "output": result.get("output", [""])[0] if result.get("output") else "",
                "selected_agents": result.get("selected_agents", []),
                "agent_results": result.get("agent_results", {}),
                "selected_crop": result.get("selected_crop", [""])[0] if result.get("selected_crop") else "",
                "crop_info": result.get("crop_info", [""])[0] if result.get("crop_info") else "",
                "status": "success",
                "error_info": result.get("error_info", {})
            }
            
        except Exception as e:
            print(f"❌ Farmer 오케스트레이터 실행 중 오류: {e}")
            traceback.print_exc()
            return {
                "output": f"Farmer 오케스트레이터 실행 중 오류가 발생했습니다: {e}",
                "selected_agents": [],
                "agent_results": {},
                "selected_crop": "",
                "crop_info": "",
                "status": "error",
                "error": str(e),
                "error_info": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "context": "main_invoke"
                }
            }

    # 에이전트 설명 정의
    agent_descriptions = {
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
        "날씨_agent": (
            "현재 날씨, 단기예보, 중기예보, 기상특보 등 실시간 기상 정보를 제공합니다. 기상청 API를 통해 정확한 날씨 데이터를 바탕으로 농업에 필요한 기상 정보를 안내합니다."
            "※ 핵심 키워드: '날씨', '기온', '강수', '예보', '하늘상태', '바람', '습도', '오늘 날씨', '내일 날씨'"
        ),
        "판매처_agent": (
            "사용자가 재배하거나 수확한 농산물을 어디에 팔 수 있는지, 판매처 위치 정보와 해당 작물의 실시간 시세, 최근 가격 변동을 안내합니다."
            "※ 핵심 키워드: '판매처', '시장', '도매상', '유통', '가격', '시세', '수익', '거래', '실시간 시세', '가격 변동', '팔고 싶어'"
        )
    }

    @lru_cache(maxsize=128)
    def simple_agent_selector(self, user_question):
        """
        사용자 질문을 분석하여 필요한 에이전트를 선택하는 함수
        메모리 시스템이 자동으로 대화 맥락을 관리합니다.
        """
        # 메모리 시스템 상태 확인
        if self.redis_connected:
            print("[에이전트 선택] Farmer Redis 메모리 시스템 - 대화 맥락 자동 관리")
        else:
            print("[에이전트 선택] 메모리 시스템 없음 - 단일 질문 처리")
        selection_prompt = f"""
        다음 질문을 분석하여 필요한 에이전트를 선택해주세요.
        
        [에이전트 역할 및 설명]
        1) 작물추천_agent: {self.agent_descriptions["작물추천_agent"]}
        
        2) 작물재배_agent: {self.agent_descriptions["작물재배_agent"]}
        
        3) 재해_agent: {self.agent_descriptions["재해_agent"]}
        
        4) 날씨_agent: {self.agent_descriptions["날씨_agent"]}
        
        5) 판매처_agent: {self.agent_descriptions["판매처_agent"]}
        
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
                        print(f"[⚠️ 질문 분류 누락 - 기본값 사용]")
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
            print(f"에이전트 선택 실패: {e}")
            return {
                "selected_agents": ["작물추천_agent"],
                "question_parts": None,
                "execution_order": ["작물추천_agent"]
            }

    async def execute_agents_parallel(self, agents: List[str], state: RouterState) -> RouterState:
        """에이전트 병렬 실행"""
        print(f"\n=== 🚀 병렬 에이전트 실행 시작: {agents} ===")
        
        async def run_agent(agent_name: str, question_part: str):
            """개별 에이전트 실행"""
            try:
                return agent_name, self.execute_agent_with_boundaries(agent_name, question_part)
            except Exception as e:
                print(f"❌ {agent_name} 실행 실패: {e}")
                return agent_name, f"{agent_name} 실행 중 오류가 발생했습니다: {e}"
        
        # 병렬 실행을 위한 태스크 생성
        tasks = []
        question_parts = state.get("question_parts", {})
        
        for agent in agents:
            if agent == "작물추천_agent":
                continue  # 작물추천은 이미 실행됨
            
            # 질문 부분 가져오기
            if question_parts and agent in question_parts:
                question_part = question_parts[agent]
            else:
                question_part = state["query"][0] if state["query"] else ""
            
            # 작물명 추가 처리
            selected_crop = state.get("selected_crop", [""])[0] if state.get("selected_crop") else ""
            if selected_crop and selected_crop not in question_part:
                if agent == "작물재배_agent":
                    question_part = f"{selected_crop} {question_part}"
                elif agent == "재해_agent":
                    question_part = f"{selected_crop} 재배 중, {question_part}"
                elif agent == "판매처_agent":
                    question_part = f"{selected_crop} {question_part}"
            
            # 태스크 추가
            tasks.append(run_agent(agent, question_part))
        
        # 병렬 실행
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            for result in results:
                if isinstance(result, Exception):
                    print(f"❌ 병렬 실행 중 예외 발생: {result}")
                    continue
                
                agent_name, answer = result
                state["agent_results"][agent_name] = answer
                print(f"✅ {agent_name} 병렬 실행 완료")
        
        except Exception as e:
            print(f"❌ 병렬 실행 실패: {e}")
            state = self._handle_error(state, e, "parallel_agent_execution")
        
        print(f"=== 🎯 병렬 에이전트 실행 완료 ===")
        return state

    def execute_agent_with_boundaries(self, agent_name, question_part):
        agent_func = self.agent_functions.get(agent_name)
        if not agent_func:
            return f"{agent_name} 실행 함수가 연결되어 있지 않습니다."

        agent_prompt = f"{question_part}"

        try:
            agent_state = {"query": agent_prompt}
            agent_result = agent_func(agent_state)
            answer = agent_result.get("agent_answer", "답변 생성 실패")
            return answer

        except Exception as e:
            return f"에이전트 실행 중 오류: {e}"

    def select_single_crop_from_recommendations(self, crop_recommendations):
        """
        작물추천 결과에서 상세 분석할 작물 하나를 선택하는 함수
        """
        print("\n=== 작물 추출 과정 시작 ===")
        
        selection_prompt = f"""
        다음은 작물추천 에이전트가 추천한 작물들입니다. 
        사용자의 질문과 상황을 고려하여 상세 분석할 작물 하나를 선택해주세요.
        
        [추천 작물 목록]
        {crop_recommendations}
        
        [요구사항]
        - 작물명만 작성 (예: 무, 토마토, 고추, 오이)
        - 설명이나 문장은 절대 포함하지 말 것
        - 한 단어로 된 작물명만
        - 작물을 찾을 수 없으면 "없음"이라고만 답변
        - 작물 추천 결과에 있는 맨 처음 작물을 선택해줘
        
        상세 분석할 작물: """
        
        try:
            print("[1단계] LLM에게 작물 추출 요청...")
            selected_crop = self.llm.invoke(selection_prompt).content.strip()
            print(f"[LLM 원본 응답] {selected_crop}")
            
            # "없음"이거나 빈 문자열인 경우 공백 반환
            if selected_crop in ["없음", "", "None", "null"]:
                print(f"[⚠️ 작물을 찾을 수 없음 - 공백 반환]")
                return ""
            
            # 간단한 정리만 수행 (clean_crop_name 함수 사용 안함)
            cleaned_crop = selected_crop.split('\n')[0].split('.')[0].split(',')[0].strip()
            print(f"[정리된 작물명] {cleaned_crop}")
            
            print(f"[✅ 최종 추출된 작물] {cleaned_crop}")
            return cleaned_crop
            
        except Exception as e:
            print(f"[❌ LLM 호출 오류 - 공백 반환] {e}")
            return ""

    @traceable(name="node_input")
    def node_input(self, state: RouterState) -> RouterState:
        while True:
            user_input = input("\n사용자 입력: ").strip()
            
            # 빈 입력인 경우 다시 요청
            if not user_input:
                print("❌ 입력이 비어 있습니다. 다시 입력해주세요.")
                continue
                
            # 유효한 입력인 경우 루프 종료
            break

        # Redis 미연결 시 새로운 질문 플래그 설정 (문맥 무시)
        if not self.redis_connected:
            state.setdefault("session", {})
            state["session"]["new_question"] = True
            print(f"[Redis 미연결] 단일 질문 처리 모드 - 문맥 무시")
        else:
            # Redis 연결 시 기존 로직 유지
            state.setdefault("session", {})
            state["session"]["new_question"] = True
        
        # 새로운 질문 저장
        state["query"] = [user_input]
        print(f"\n[질문] {user_input}")
        print(f"[새로운 질문 처리 시작]")
        
        return state

    @traceable(name="node_agent_select")
    def node_agent_select(self, state: RouterState) -> RouterState:
        # 메모리 시스템 상태 표시
        memory_type = "Farmer Redis" if self.redis_connected else "None"
        print(f"\n[에이전트 선택] 메모리 시스템: {memory_type}")
        
        # 기존 복잡한 로직을 단순화된 함수로 교체
        result = self.simple_agent_selector(state["query"][0] if state["query"] else "")
        # 기존 selected_agents 덮어쓰기 (중복 방지)
        state["selected_agents"] = result["selected_agents"] if isinstance(result["selected_agents"], list) else [result["selected_agents"]]
        state["question_parts"] = result.get("question_parts", {}) if result.get("question_parts") is not None else {}
        state["execution_order"] = result["execution_order"] if isinstance(result["execution_order"], list) else [result["execution_order"]]
        
        print("\n[선택된 에이전트]")
        for agent in state["selected_agents"]:
            print(f"- {agent}")
        
        return state

    @traceable(name="node_crop_recommend")
    def node_crop_recommend(self, state: RouterState) -> RouterState:
        if "작물추천_agent" not in state.get("selected_agents", []):
            return state
        
        print("\n=== 작물추천_agent 실행 ===")
        
        # question_parts가 None인 경우 안전하게 처리
        question_parts = state.get("question_parts")
        if question_parts is None:
            # 단일 에이전트인 경우 원본 질문 사용
            question_part = state["query"][0] if state["query"] else ""
            print(f"[�� 단일 에이전트 - 원본 질문 사용] {question_part}")
        else:
            # 다중 에이전트인 경우 분류된 질문 사용
            question_part = question_parts.get("작물추천_agent", state["query"][0] if state["query"] else "")
            print(f"[📝 다중 에이전트 - 분류된 질문 사용] {question_part}")
        
        print(f"담당 질문: {question_part}")
        
        # 명확한 경계가 설정된 프롬프트로 실행
        answer = self.execute_agent_with_boundaries("작물추천_agent", question_part)
        
        print(f"\n[작물추천_agent 원본 응답]\n{answer}")
        
        # 작물추천 결과에서 하나의 작물 선택
        selected_crop = self.select_single_crop_from_recommendations(answer)
        
        state["crop_info"] = [answer]
        state["selected_crop"] = [selected_crop]  # 선택된 단일 작물 저장
        
        print(f"\n[추출된 작물] {selected_crop}")
        print(f"[작물 추출 완료]")
        
        return state

    # 각 에이전트별로 개별 노드 생성
    @traceable(name="node_crop_grow_agent")
    def node_crop_grow_agent(self, state: RouterState) -> RouterState:
        """작물재배_agent 전용 노드"""
        if "작물재배_agent" not in state.get("selected_agents", []):
            return state
        
        print(f"\n=== 🚀 작물재배_agent 병렬 실행 ===")
        
        # 질문 부분 가져오기
        question_parts = state.get("question_parts", {})
        if question_parts and "작물재배_agent" in question_parts:
            question_part = question_parts["작물재배_agent"]
        else:
            question_part = state["query"][0] if state["query"] else ""
        
        print(f"[📝 담당 질문] {question_part}")
        
        # 작물재배_agent 전용 작물명 처리
        selected_crop = state.get("selected_crop", [""])[0] if state.get("selected_crop") else ""
        if selected_crop and selected_crop not in question_part:
            question_part = f"{selected_crop} {question_part}"
            print(f"[🔄 수정된 질문 ] {question_part}")

        # 에이전트 실행
        answer = self.execute_agent_with_boundaries("작물재배_agent", question_part)
        
        # 전용 키에 답변 저장
        state["agent_results"]["작물재배_agent"] = answer
        
        print(f"[✅ 작물재배_agent 병렬 실행 완료]")
        print(f"[📤 응답 원본] {answer[:200]}...")
        return state

    @traceable(name="node_disaster_agent")
    def node_disaster_agent(self, state: RouterState) -> RouterState:
        """재해_agent 전용 노드"""
        if "재해_agent" not in state.get("selected_agents", []):
            return state
        
        print(f"\n=== 🚀 재해_agent 병렬 실행 ===")
        
        # 질문 부분 가져오기
        question_parts = state.get("question_parts", {})
        if question_parts and "재해_agent" in question_parts:
            question_part = question_parts["재해_agent"]
        else:
            question_part = state["query"][0] if state["query"] else ""
        
        print(f"[📝 담당 질문] {question_part}")
        
        # 재해_agent 전용 작물명 처리
        selected_crop = state.get("selected_crop", [""])[0] if state.get("selected_crop") else ""
        if selected_crop and selected_crop not in question_part:
            question_part = f"{selected_crop} 재배 중, {question_part}"
            print(f"[🔄 수정된 질문 ] {question_part}")
        
        # 에이전트 실행
        answer = self.execute_agent_with_boundaries("재해_agent", question_part)
        
        # 전용 키에 답변 저장
        state["agent_results"]["재해_agent"] = answer
        
        print(f"[✅ 재해_agent 병렬 실행 완료]")
        print(f"[📤 응답 원본] {answer[:200]}...")
        return state

    @traceable(name="node_sales_agent")
    def node_sales_agent(self, state: RouterState) -> RouterState:
        """판매처_agent 전용 노드"""
        if "판매처_agent" not in state.get("selected_agents", []):
            return state
        
        print(f"\n=== �� 판매처_agent 병렬 실행 ===")
        
        # 질문 부분 가져오기
        question_parts = state.get("question_parts", {})
        if question_parts and "판매처_agent" in question_parts:
            question_part = question_parts["판매처_agent"]
        else:
            question_part = state["query"][0] if state["query"] else ""
        
        print(f"[📝 담당 질문] {question_part}")
        
        # 판매처_agent 전용 작물명 처리
        selected_crop = state.get("selected_crop", [""])[0] if state.get("selected_crop") else ""
        if selected_crop and selected_crop not in question_part:
            question_part = f"{selected_crop} {question_part}"
            print(f"[🔄 수정된 질문 ] {question_part}")
        
        # 에이전트 실행
        answer = self.execute_agent_with_boundaries("판매처_agent", question_part)
        
        # 전용 키에 답변 저장
        state["agent_results"]["판매처_agent"] = answer
        
        print(f"[✅ 판매처_agent 병렬 실행 완료]")
        print(f"[📤 응답 원본] {answer[:200]}...")
        return state

    @traceable(name="node_weather_agent")
    def node_weather_agent(self, state: RouterState) -> RouterState:
        """날씨_agent 전용 노드"""
        if "날씨_agent" not in state.get("selected_agents", []):
            return state
        
        print(f"\n=== 🌤️ 날씨_agent 병렬 실행 ===")
        
        # 질문 부분 가져오기
        question_parts = state.get("question_parts", {})
        if question_parts and "날씨_agent" in question_parts:
            question_part = question_parts["날씨_agent"]
        else:
            question_part = state["query"][0] if state["query"] else ""
        
        print(f"[📝 담당 질문] {question_part}")
        
        # 날씨_agent는 작물명 처리가 필요 없으므로 원본 질문 그대로 사용
        # (날씨는 지역과 시간이 중요하므로)
        
        # 에이전트 실행
        answer = self.execute_agent_with_boundaries("날씨_agent", question_part)
        
        # 전용 키에 답변 저장
        state["agent_results"]["날씨_agent"] = answer
        
        print(f"[✅ 날씨_agent 병렬 실행 완료]")
        print(f"[📤 응답 원본] {answer[:200]}...")
        return state
        
    # 병렬 처리 노드 (개선된 비동기 처리)
    @traceable(name="node_parallel_agents")
    def node_parallel_agents(self, state: RouterState) -> RouterState:
        """병렬 에이전트 실행을 조정하는 노드"""
        selected_agents = state.get("execution_order", [])
        
        # 작물추천_agent만 있는 경우
        if len(selected_agents) == 1 and "작물추천_agent" in selected_agents:
            print(f"\n=== 🎯 작물추천_agent만 선택됨 - 병렬 처리 건너뜀 ===")
            return state
        
        # 여러 에이전트가 있는 경우 병렬 처리 실행
        print(f"\n=== 🚀 병렬 에이전트 실행 시작 ===")
        agents_to_run = [agent for agent in selected_agents if agent != "작물추천_agent"]
        print(f"[📋 실행될 에이전트] {agents_to_run}")
        
        try:
            # 비동기 병렬 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result_state = loop.run_until_complete(
                self.execute_agents_parallel(agents_to_run, state)
            )
            loop.close()
            return result_state
        except Exception as e:
            print(f"❌ 병렬 처리 노드 실행 실패: {e}")
            return self._handle_error(state, e, "node_parallel_agents")

    @traceable(name="node_merge_output")
    def node_merge_output(self, state: RouterState) -> RouterState:
        print("\n=== 최종 응답 병합 시작 ===")
        
        # 각 에이전트 결과 수집
        agent_results = {}
        
        if state.get("crop_info"):
            agent_results["작물추천_agent"] = state["crop_info"][0] if state["crop_info"] else ""

        if state.get("agent_results"):
            agent_results.update(state["agent_results"])
        
        # 실행 요약 출력
        selected_agents = state.get("selected_agents", [])
        print(f"[ 실행 요약]")
        print(f"  - 선택된 에이전트: {selected_agents}")
        print(f"  - 선택된 작물: {state.get('selected_crop', [''])[0] if state.get('selected_crop') else ''}")
        print(f"  - 실행된 에이전트: {list(agent_results.keys())}")
        
        output = ""
        
        # 에이전트가 하나뿐인 경우 단순 처리
        if len(selected_agents) == 1:
            agent = selected_agents[0]
            if agent in agent_results:
                output = agent_results[agent]
                print(f"[✅ 단일 에이전트 응답 완료] {agent}")
            else:
                output = f"{agent} 실행 결과를 찾을 수 없습니다."
                print(f"[❌ {agent} 응답 없음]")
        else:
            # 여러 에이전트가 있는 경우 기존 로직 유지
            # 작물추천 결과가 있으면 먼저 표시
            if state.get("crop_info"):
                output += f"[작물추천 결과]\n{state['crop_info']}\n"
                
                # 선택된 작물 강조 표시
                if state.get("selected_crop"):
                    output += f"\n[상세 분석 작물]\n{state['selected_crop']}\n"
                    print(f"[ 상세 분석 작물] {state['selected_crop']}")
            
            # 다른 에이전트들의 답변 표시
            for agent, answer in agent_results.items():
                if agent != "작물추천_agent":  # 이미 표시됨
                    # 에이전트 결과 추가
                    output += f"[{agent} 결과]\n{answer}\n"
            
            # 다른 작물 정보 안내 추가
            if state.get("crop_info") and state.get("selected_crop"):
                output += f"\n[추가 정보 안내]\n"
                output += f"다른 추천 작물에 대한 상세 정보가 궁금하시다면, "
                output += f"'{state['selected_crop'][0] if state['selected_crop'] else ''} 대신 [작물명]에 대해 알려주세요'와 같이 질문해주세요.\n"
        
        merged_output = output.strip()
        
        # 에이전트가 하나뿐인 경우 LLM 요약 생략
        if len(selected_agents) == 1:
            state["output"] = [merged_output]
            print("\n=== 🎯 최종 응답(단일 에이전트) ===")
            print("=" * 50)
            print(state["output"][0] if state["output"] else "")
            print("=" * 50)
            return state
        
        # 여러 에이전트가 있는 경우에만 LLM 요약
        print("\n[🤖 LLM 요약 시작...]")
        summary_prompt = (
            """
            아래는 여러 농업 에이전트의 답변입니다. 답변 외의 정보는 제외해줘.
            사용자에게 최대한 자세하고 상세하게 한국어로 알려주세요.
            우선 순위는 작물 추천_agent, 재배 방법_agent, 재해_agent, 판매처_agent 순으로 최대 2800자 이내로 정리해줘.
            내용 안에 agent 이름을 넣지 말고 대화하는 것처럼 사용자에게 대답해줘.
            마지막에는 사용자에게 다른 질문을 유도하는 문장을 넣어줘.
            \n\n"""
            f"{merged_output}\n\n"
        )
        
        try:
            summary = self.llm.invoke(summary_prompt)
            print(f"[✅ LLM 요약 완료] {len(summary.content)}자")
        except Exception as e:
            summary = f"요약 중 오류: {e}"
            print(f"[❌ LLM 요약 실패] {e}")
        
        state["output"] = [summary.content.strip() if hasattr(summary, 'content') else summary.strip()]
        
        # 최종 요약된 응답만 출력 (중복 제거)
        print("\n=== 🎯 최종 응답(요약) ===")
        print(f"[📊 요약 길이] {len(state['output'][0]) if state['output'] else 0}자")
        print("=" * 50)
        print(state["output"][0] if state["output"] else "")
        print("=" * 50)
        
        return state

    # 워크플로우 그래프
    def create_workflow(self):
        """완전한 조건부 분기 워크플로우"""
        workflow = StateGraph(RouterState)
        
        # 노드 추가
        workflow.add_node("load_state", RunnableLambda(self.load_state))
        workflow.add_node("persist_state", RunnableLambda(self.persist_state))
        workflow.add_node("input", self.node_input)
        workflow.add_node("agent_select", self.node_agent_select)
        workflow.add_node("crop_recommend", self.node_crop_recommend)
        workflow.add_node("parallel_execution", self.node_parallel_agents)
        workflow.add_node("crop_grow_agent", self.node_crop_grow_agent)
        workflow.add_node("disaster_agent", self.node_disaster_agent)
        workflow.add_node("weather_agent", self.node_weather_agent)
        workflow.add_node("sales_agent", self.node_sales_agent)

        workflow.add_node("merge_output", self.node_merge_output)
        
        # 기본 엣지 (메모리 시스템 포함)
        workflow.add_edge("load_state", "input")
        workflow.add_edge("input", "agent_select")
        
        # agent_select에서 조건부 분기 (etc 제거)
        def agent_select_branch_condition(state):
            selected_agents = state.get("selected_agents", [])
            
            # 작물추천_agent가 선택된 경우
            if "작물추천_agent" in selected_agents:
                return "crop_recommend"
            # 단일 에이전트가 선택된 경우 (작물추천_agent 제외)
            elif len(selected_agents) == 1:
                agent = selected_agents[0]
                if agent == "작물재배_agent":
                    return "crop_grow_agent"
                elif agent == "재해_agent":
                    return "disaster_agent"
                elif agent == "날씨_agent":
                    return "weather_agent"
                elif agent == "판매처_agent":
                    return "sales_agent"

            # 여러 에이전트가 선택된 경우
            elif len([agent for agent in selected_agents if agent != "작물추천_agent"]) > 0:
                return "parallel_execution"
            # 아무것도 선택되지 않은 경우 - 작물추천으로 기본 설정
            else:
                return "crop_recommend"

        workflow.add_conditional_edges(
            "agent_select",
            agent_select_branch_condition,
            {
                "crop_recommend": "crop_recommend",
                "crop_grow_agent": "crop_grow_agent",
                "disaster_agent": "disaster_agent", 
                "weather_agent": "weather_agent",
                "sales_agent": "sales_agent",
                "parallel_execution": "parallel_execution",

            }
        )
        
        # crop_recommend에서 조건부 분기
        workflow.add_conditional_edges(
            "crop_recommend",
            lambda state: "parallel_execution" if len([agent for agent in state.get("selected_agents", []) if agent != "작물추천_agent"]) > 0 else "merge_output",
            {
                "parallel_execution": "parallel_execution",
                "merge_output": "merge_output"
            }
        )
        
        # 병렬 에이전트 실행
        workflow.add_edge("parallel_execution", "crop_grow_agent")
        workflow.add_edge("parallel_execution", "disaster_agent")
        workflow.add_edge("parallel_execution", "weather_agent")
        workflow.add_edge("parallel_execution", "sales_agent")
        
        # 모든 에이전트 노드에서 병합 노드로
        workflow.add_edge("crop_grow_agent", "merge_output")
        workflow.add_edge("disaster_agent", "merge_output")
        workflow.add_edge("weather_agent", "merge_output")
        workflow.add_edge("sales_agent", "merge_output")

        
        # 병합 노드에서 메모리 저장 후 종료
        workflow.add_edge("merge_output", "persist_state")
        workflow.add_edge("persist_state", END)
        
        workflow.set_entry_point("load_state")
        
        return workflow.compile()

    def run_orchestrator_langgraph(self):
        graph = self.create_workflow()

        while True:
            try:
                state = RouterState()
                config = {"configurable": {"thread_id": f"{self.user_id}_{self.chat_id}"}}
                result = graph.invoke(state, config)
                
            except KeyboardInterrupt:
                print("\n\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류가 발생했습니다: {e}")
                traceback.print_exc()
                continue

    
    def run_standalone(self):
        """독립 실행용 함수 (기존 방식과 호환)"""
        while True:
            try:
                state = RouterState()
                config = {"configurable": {"thread_id": f"{self.user_id}_{self.chat_id}"}}
                result = self.graph.invoke(state, config)
                
            except KeyboardInterrupt:
                print("\n\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류가 발생했습니다: {e}")
                traceback.print_exc()
                continue


# 기존 실행 방식과의 호환성을 위한 전역 함수들
def run_orchestrator_langgraph():
    """기존 실행 방식과의 호환성을 위한 함수"""
    farmer = Farmer(user_id="default_user", service="farmer", chat_id="default_chat")
    # graph = farmer.create_workflow()
    # try:
    #     graph_image_path = "ochestrator_workflow.png"
    #     with open(graph_image_path, "wb") as f:
    #         f.write(graph.get_graph().draw_mermaid_png())
    #     print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    # except Exception as e:
    #     print(f"그래프 시각화 중 오류 발생: {e}")
    farmer.run_standalone()

if __name__ == "__main__":
    run_orchestrator_langgraph()