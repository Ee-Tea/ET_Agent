from sentence_transformers import SentenceTransformer, util
import requests
import re
import json
from langgraph.graph import StateGraph, END
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Dict
from tavily import TavilyClient
import operator
from langsmith import traceable
from dotenv import load_dotenv
import os
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

class Farmer:
    """농업 오케스트레이터 - Supervisor에게 병합 가능한 구조"""
    
    def __init__(self):
        """Farmer 클래스 초기화"""
        # LLM 설정
        self.llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct",
                           temperature=0.8,
                           api_key=os.getenv("OPENAI_KEY1"))
        # gpt-4o-mini 사용 시 주석 해제
        # self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.8, api_key=os.getenv("OPENAI_API_KEY"))
        
        # 에이전트 함수들 로드
        self._load_agent_functions()
        
        # 워크플로우 그래프 생성
        self.graph = self.create_workflow()
    
    def _load_agent_functions(self):
        """에이전트 함수들을 로드"""
        from 작물추천.crop65pdfllm import run as crop_recommend_run
        from 재배방법.crop_overall import run as crop_cultivation_run
        from 재해대응.verification_search import run as disaster_run
        from sales.SalesAgent import run as market_run
        
        self.agent_functions = {
            "작물추천_agent": crop_recommend_run,
            "작물재배_agent": crop_cultivation_run,
            "재해_agent": disaster_run,
            "판매처_agent": market_run,
            "기타": self.etc_agent_run
        }
    
    def execute(self, state: dict) -> dict:
        """
        Supervisor에서 호출되는 메인 실행 함수
        
        Args:
            state: Supervisor에서 전달받은 상태 딕셔너리
                  - query: 사용자 질문 (필수)
                  - 기타 필요한 상태 정보들
        
        Returns:
            dict: 실행 결과
                - output: 최종 응답
                - selected_agents: 선택된 에이전트들
                - agent_results: 각 에이전트별 결과
        """
        try:
            # RouterState로 변환
            router_state = RouterState()
            router_state["query"] = [state.get("query", "")]
            
            # 그래프 실행
            result = self.graph.invoke(router_state)
            
            # 결과 반환
            return {
                "output": result.get("output", [""])[0] if result.get("output") else "",
                "selected_agents": result.get("selected_agents", []),
                "agent_results": result.get("agent_results", {}),
                "selected_crop": result.get("selected_crop", [""])[0] if result.get("selected_crop") else "",
                "crop_info": result.get("crop_info", [""])[0] if result.get("crop_info") else "",
                "status": "success"
            }
            
        except Exception as e:
            return {
                "output": f"Farmer 오케스트레이터 실행 중 오류가 발생했습니다: {e}",
                "selected_agents": [],
                "agent_results": {},
                "selected_crop": "",
                "crop_info": "",
                "status": "error",
                "error": str(e)
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
        "판매처_agent": (
            "사용자가 재배하거나 수확한 농산물을 어디에 팔 수 있는지, 판매처 위치 정보와 해당 작물의 실시간 시세, 최근 가격 변동을 안내합니다."
            "※ 핵심 키워드: '판매처', '시장', '도매상', '유통', '가격', '시세', '수익', '거래', '실시간 시세', '가격 변동', '팔고 싶어'"
        ),
        "기타": "농업과 전혀 관련 없는 질문일 경우 선택합니다."
    }

    def simple_agent_selector(self, user_question):
        """
        사용자 질문을 분석하여 필요한 에이전트를 선택하는 함수
        """
        selection_prompt = f"""
        다음 질문을 분석하여 필요한 에이전트를 선택해주세요.
        
        [에이전트 역할 및 설명]
        1) 작물추천_agent: {self.agent_descriptions["작물추천_agent"]}
        
        2) 작물재배_agent: {self.agent_descriptions["작물재배_agent"]}
        
        3) 재해_agent: {self.agent_descriptions["재해_agent"]}
        
        4) 판매처_agent: {self.agent_descriptions["판매처_agent"]}
        
        5) 기타: {self.agent_descriptions["기타"]}
        
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
                    # 에이전트가 0개인 경우
                    return {
                        "selected_agents": ["기타"],
                        "question_parts": None,
                        "execution_order": ["기타"]
                    }
            else:
                # JSON 파싱 실패 시 기본값
                return {
                    "selected_agents": ["기타"],
                    "question_parts": None,
                    "execution_order": ["기타"]
                }
        except Exception as e:
            print(f"에이전트 선택 실패: {e}")
            return {
                "selected_agents": ["기타"],
                "question_parts": None,
                "execution_order": ["기타"]
            }

    def execute_agent_with_boundaries(self, agent_name, question_part):
        agent_func = self.agent_functions.get(agent_name)
        if not agent_func:
            return f"{agent_name} 실행 함수가 연결되어 있지 않습니다."

        agent_prompt = f"질문: {question_part}"

        try:
            agent_state = {"query": agent_prompt}
            agent_result = agent_func(agent_state)
            answer = agent_result.get("pred_answer", "답변 생성 실패")
            return answer

        except Exception as e:
            return f"에이전트 실행 중 오류: {e}"

    def web_search_with_tavily(self, query: str, api_key: str = None):
        """
        Tavily를 이용한 웹 검색
        """
        try:
            # API 키 설정
            if not api_key:
                api_key = os.getenv("TAVILY_API_KEY")
            
            if not api_key:
                return "Tavily API 키가 설정되지 않았습니다."
            
            # Tavily 클라이언트 생성
            client = TavilyClient(api_key=api_key)
            
            # 웹 검색 실행
            search_result = client.search(
                query=query,
                search_depth="basic",
                max_results=5
            )
            
            # 결과 정리
            if search_result and 'results' in search_result:
                formatted_results = "=== 웹 검색 결과 ===\n\n"
                for i, result in enumerate(search_result['results'][:5], 1):
                    formatted_results += f"{i}. {result.get('title', '제목 없음')}\n"
                    formatted_results += f"   URL: {result.get('url', 'URL 없음')}\n"
                    formatted_results += f"   내용: {result.get('content', '내용 없음')[:200]}...\n\n"
                return formatted_results
            else:
                return "웹 검색 결과를 찾을 수 없습니다."
                
        except ImportError:
            return "Tavily 라이브러리가 설치되지 않았습니다. 'pip install tavily-python'으로 설치해주세요."
        except Exception as e:
            return f"웹 검색 중 오류가 발생했습니다: {e}"

    def etc_agent_run(self, state: dict) -> dict:
        """
        기타 에이전트 - 웹 검색을 통한 답변
        """
        query = state.get("query", "")
        
        # 웹 검색 실행
        print(f"[기타_agent] 웹 검색 시작: {query}")
        web_result = self.web_search_with_tavily(query)
        
        # 결과 정리
        if "오류" in web_result or "설정되지 않음" in web_result:
            final_answer = f"질문: {query}\n\n{web_result}"
        else:
            final_answer = f"질문: {query}\n\n{web_result}\n\n※ 위 정보는 웹 검색을 통해 제공되었습니다."
        
        return {
            "pred_answer": final_answer,
            "source": "web_search"
        }




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

        # 모든 상태 초기화
        state["crop_info"] = []
        state["selected_crop"] = []
        state["selected_agents"] = []
        state["question_parts"] = {}
        state["execution_order"] = []
        state["agent_results"] = {} # 에이전트별 결과 딕셔너리 초기화
        state["output"] = []

        # 유효한 입력인 경우 상태에 저장하고 다음 단계로 (리스트로 저장)
        state["query"] = [user_input]
        print(f"\n[질문] {user_input}")
        
        return state

    @traceable(name="node_agent_select")
    def node_agent_select(self, state: RouterState) -> RouterState:
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

    @traceable(name="node_etc")
    def node_etc(self, state: RouterState) -> RouterState:
        """기타 에이전트 전용 노드"""
        if "기타" not in state.get("selected_agents", []):
            return state
        
        print(f"\n=== �� 기타_agent 웹검색 실행 ===")
        
        # 원본 질문 사용
        question_part = state["query"][0] if state["query"] else ""
        print(f"[📝 담당 질문] {question_part}")
        
        # 에이전트 실행
        answer = self.execute_agent_with_boundaries("기타", question_part)
        
        # 전용 키에 답변 저장
        state["agent_results"]["기타"] = answer
        
        print(f"[✅ 기타_agent 웹검색 실행 완료]")
        print(f"[📤 응답 원본] {answer[:200]}...")
        return state

    # 병렬 처리 노드 (기존 로직 단순화)
    @traceable(name="node_parallel_agents")
    def node_parallel_agents(self, state: RouterState) -> RouterState:
        """병렬 에이전트 실행을 조정하는 노드"""
        selected_agents = state.get("execution_order", [])
        
        # 작물추천_agent만 있는 경우
        if len(selected_agents) == 1 and "작물추천_agent" in selected_agents:
            print(f"\n=== 🎯 작물추천_agent만 선택됨 - 병렬 처리 건너뜀 ===")
            return state
        
        # 여러 에이전트가 있는 경우 병렬 처리 준비
        print(f"\n=== 🚀 병렬 에이전트 실행 준비 완료 ===")
        print(f"[📋 실행될 에이전트] {[agent for agent in selected_agents if agent != '작물추천_agent']}")
        
        return state

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
        workflow.add_node("input", self.node_input)
        workflow.add_node("agent_select", self.node_agent_select)
        workflow.add_node("crop_recommend", self.node_crop_recommend)
        workflow.add_node("parallel_execution", self.node_parallel_agents)
        workflow.add_node("crop_grow_agent", self.node_crop_grow_agent)
        workflow.add_node("disaster_agent", self.node_disaster_agent)
        workflow.add_node("sales_agent", self.node_sales_agent)
        workflow.add_node("etc", self.node_etc)
        workflow.add_node("merge_output", self.node_merge_output)
        
        # 기본 엣지
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
                elif agent == "판매처_agent":
                    return "sales_agent"
                elif agent == "기타":
                    return "etc"
            # 여러 에이전트가 선택된 경우
            elif len([agent for agent in selected_agents if agent != "작물추천_agent"]) > 0:
                return "parallel_execution"
            # 아무것도 선택되지 않은 경우
            else:
                return "etc"

        workflow.add_conditional_edges(
            "agent_select",
            agent_select_branch_condition,
            {
                "crop_recommend": "crop_recommend",
                "crop_grow_agent": "crop_grow_agent",
                "disaster_agent": "disaster_agent", 
                "sales_agent": "sales_agent",
                "parallel_execution": "parallel_execution",
                "etc": "etc"
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
        workflow.add_edge("parallel_execution", "sales_agent")
        
        # 모든 에이전트 노드에서 병합 노드로
        workflow.add_edge("crop_grow_agent", "merge_output")
        workflow.add_edge("disaster_agent", "merge_output")
        workflow.add_edge("sales_agent", "merge_output")
        workflow.add_edge("etc", "merge_output")
        
        # 병합 노드에서 다시 입력으로
        workflow.add_edge("merge_output", END)
        
        workflow.set_entry_point("input")
        
        return workflow.compile()

    def run_orchestrator_langgraph(self):
        graph = self.create_workflow()
        # try:
        #     graph_image_path = "ochestrator_workflow.png"
        #     with open(graph_image_path, "wb") as f:
        #         f.write(graph.get_graph().draw_mermaid_png())
        #     print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
        # except Exception as e:
        #     print(f"그래프 시각화 중 오류 발생: {e}")

        while True:
            try:
                state = RouterState()
                result = graph.invoke(state)
                
            except KeyboardInterrupt:
                print("\n\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류가 발생했습니다: {e}")
                continue

    
    def run_standalone(self):
        """독립 실행용 함수 (기존 방식과 호환)"""
        while True:
            try:
                state = RouterState()
                result = self.graph.invoke(state)
                
            except KeyboardInterrupt:
                print("\n\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류가 발생했습니다: {e}")
                continue


# 기존 실행 방식과의 호환성을 위한 전역 함수들
def run_orchestrator_langgraph():
    """기존 실행 방식과의 호환성을 위한 함수"""
    farmer = Farmer()
    farmer.run_standalone()

if __name__ == "__main__":
    run_orchestrator_langgraph()