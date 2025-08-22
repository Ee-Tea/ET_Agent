import streamlit as st
import requests
import re
import json
import warnings
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import os
from groq import Groq

# --- 1. 환경 설정 및 라이브러리 로드 (Streamlit UI 로직보다 먼저 실행) ---
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# --- 2. Streamlit 페이지 설정 및 UI 초기화 (반드시 모든 st. 명령보다 상단에 위치) ---
st.set_page_config(page_title="AI 농업 전문가", layout="centered")
st.title("AI 농업 챗봇 🧑‍🌾")
st.markdown("작물 재배, 재해 대응, 판매처 등 농업 관련 궁금증을 해결해 드립니다.")

# --- 3. 에이전트 모듈 로드 및 상태 메시지 표시 ---
agent_modules_loaded = False
try:
    from 작물추천.crop65pdfllm import run as crop_recommend_run
    from 재배방법.crop_overall import run as crop_cultivation_run
    from 재해대응.verification_search import run as disaster_run
    from sales.SalesAgent import run as market_run
    agent_modules_loaded = True
    st.info("✅ 에이전트 모듈이 성공적으로 로드되었습니다.")
except ImportError:
    st.warning("🚨 에이전트 모듈을 찾을 수 없습니다. 데모용 더미 함수를 사용합니다.")
    def crop_recommend_run(state): return {"pred_answer": "작물추천 에이전트 (더미)가 추천 정보를 생성했습니다."}
    def crop_cultivation_run(state): return {"pred_answer": "작물재배 에이전트 (더미)가 재배 방법을 생성했습니다."}
    def disaster_run(state): return {"pred_answer": "재해대응 에이전트 (더미)가 재해 대응 방법을 생성했습니다."}
    def market_run(state): return {"pred_answer": "판매처 에이전트 (더미)가 판매 정보를 생성했습니다."}
    agent_modules_loaded = False

# --- 4. 기존 프로젝트의 핵심 함수 및 클래스 정의 ---

agent_descriptions = {
    "작물추천_agent": (
        "사용자의 재배 환경(계절, 토양, 기후 등), 목적, 특정 조건(수확 시기, 맛, 저장성 등)에 맞는 새로운 작물이나 품종을 추천합니다."
    ),
    "작물재배_agent": (
        "씨앗, 모종 심기부터 작물의 재배 방법, 심는 방법, 이랑을 만드는 방법, 솎음, 영양 관리(시비, 비료, 거름), 병해충 방제, 수확에 이르기까지 특정 작물을 키우는 데 필요한 일상적인 재배 관리 정보를 제공합니다."
    ),
    "재해_agent": (
        "폭염, 한파, 가뭄, 집중호우, 홍수 등 자연재해 및 이상기후로 인한 피해를 예방하고 대응하는 방법을 안내합니다. 재해 발생 전 대비, 재해 발생 중의 조치, 재해 후 작물 복구 및 피해 최소화 방안을 다룹니다."
    ),
    "판매처_agent": (
        "사용자가 재배하거나 수확한 농산물을 어디에 팔 수 있는지, 판매처 위치 정보와 해당 작물의 실시간 시세, 최근 가격 변동을 안내합니다."
    ),
    "기타": "농업과 전혀 관련 없는 질문일 경우 선택합니다."
}

class GroqLLM:
    def __init__(self, model="llama3-8b-8192", api_key=None):
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = Groq()
        self.model = model

    def invoke(self, prompt: str):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_completion_tokens=2048,
            top_p=0.8,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content.strip()

llm = GroqLLM(api_key=os.getenv("OPENAI_KEY1"))

def simple_agent_selector(user_question, llm):
    selection_prompt = f"""
    당신은 사용자 질문에 가장 적합한 농업 전문 에이전트를 선택하는 라우터입니다.
    다음 질문을 분석하여 필요한 에이전트를 JSON 형식으로 선택해주세요.

    [에이전트 역할 및 설명]
    1) 작물추천_agent: {agent_descriptions["작물추천_agent"]}
    2) 작물재배_agent: {agent_descriptions["작물재배_agent"]}
    3) 재해_agent: {agent_descriptions["재해_agent"]}
    4) 판매처_agent: {agent_descriptions["판매처_agent"]}
    5) 기타: 농업과 전혀 관련 없는 질문일 경우.

    [응답 규칙]
    - 질문이 농업과 관련 있다면 절대 '기타'를 선택하지 마세요.
    - 가장 적합한 에이전트명을 선택합니다.
    - 답변은 반드시 아래의 JSON 형식만 포함해야 합니다. 다른 설명은 절대 추가하지 마세요.
    
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
    
    질문: "{user_question}"
    """
    
    try:
        result = llm.invoke(selection_prompt)
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            parsed_result = json.loads(json_match.group())
            selected_agents = parsed_result.get("selected_agents", [])
            
            if "기타" in selected_agents and any(keyword in user_question for keyword in ["키우기", "재배", "추천", "심을", "좋은"]):
                return {"selected_agents": ["작물추천_agent"], "question_parts": None, "execution_order": ["작물추천_agent"]}

            if not selected_agents:
                return {"selected_agents": ["기타"], "question_parts": None, "execution_order": ["기타"]}
            
            if len(selected_agents) == 1:
                return {
                    "selected_agents": selected_agents,
                    "question_parts": None,
                    "execution_order": parsed_result.get("execution_order", [])
                }
            elif len(selected_agents) >= 2:
                if "question_parts" in parsed_result:
                    return parsed_result
                else:
                    question_parts = {agent: user_question for agent in selected_agents}
                    return {
                        "selected_agents": selected_agents,
                        "question_parts": question_parts,
                        "execution_order": parsed_result.get("execution_order", [])
                    }
        else:
            if any(keyword in user_question for keyword in ["키우기", "재배", "추천", "심을", "좋은"]):
                return {"selected_agents": ["작물추천_agent"], "question_parts": None, "execution_order": ["작물추천_agent"]}
            else:
                return {"selected_agents": ["기타"], "question_parts": None, "execution_order": ["기타"]}
    except Exception as e:
        return {"selected_agents": ["기타"], "question_parts": None, "execution_order": ["기타"]}

def execute_agent_with_boundaries(agent_name, question_part, llm):
    agent_functions = {
        "작물추천_agent": crop_recommend_run, "작물재배_agent": crop_cultivation_run,
        "재해_agent": disaster_run, "판매처_agent": market_run, "기타": etc_agent_run
    }
    agent_func = agent_functions.get(agent_name)
    if not agent_func:
        return f"{agent_name} 실행 함수가 연결되어 있지 않습니다."
    
    try:
        agent_state = {"query": question_part}
        agent_result = agent_func(agent_state)
        return agent_result.get("pred_answer", "답변 생성 실패")
    except Exception as e:
        return f"에이전트 실행 중 오류: {e}"

def web_search_with_tavily(query: str, api_key: str = None):
    try:
        from tavily import TavilyClient
        
        if not api_key:
            api_key = os.getenv("TAVILY_API_KEY")
        
        if not api_key:
            return "Tavily API 키가 설정되지 않았습니다."
        
        client = TavilyClient(api_key=api_key)
        
        search_result = client.search(
            query=query,
            search_depth="basic",
            max_results=5
        )
        
        if search_result and 'results' in search_result:
            return search_result['results']
        else:
            return []
                        
    except ImportError:
        return "Tavily 라이브러리가 설치되지 않았습니다. 'pip install tavily-python'으로 설치해주세요."
    except Exception as e:
        return f"웹 검색 중 오류가 발생했습니다: {e}"

def etc_agent_run(state: dict) -> dict:
    query = state.get("query", "")
    st.info("웹 검색을 실행 중입니다... 🌐")
    search_results = web_search_with_tavily(query)
    
    if not search_results:
        final_answer = "웹 검색 결과를 찾을 수 없습니다."
    else:
        # 웹 검색 결과를 텍스트로 변환
        search_results_str = "\n\n".join([
            f"제목: {res.get('title', '없음')}\nURL: {res.get('url', '없음')}\n내용: {res.get('content', '없음')}" 
            for res in search_results
        ])
        
        # LLM을 사용하여 답변 생성 프롬프트 구성
        summary_prompt = f"""
        당신은 검색 전문가입니다.
        아래 검색 결과들을 활용하여 사용자의 질문에 가장 정확하고 완전한 답변을 제공해 주세요.

        답변 규칙
        1. **친절하고 자연스럽게**: 친근하고 명확한 문체로 작성해 주세요.
        2. **정보의 출처 명시**: 검색 결과에 제시된 정보만을 사용하세요. 만약 질문에 대한 답변이 검색 결과에 없다면, '검색 결과에 해당 정보가 없습니다.'라고 명확하게 말해야 합니다.
        3. **핵심 요약 및 정리**: 여러 검색 결과에서 중복되는 핵심 내용들을 종합하여 간결하게 요약해 주세요.
        4. **구체적이고 상세하게**: 답변은 가능한 한 구체적인 정보(예: 날짜, 숫자, 기관명 등)를 포함하여 작성해 주세요.
        5. **한글로만 답변**: 모든 답변은 한글로만 제공해야 합니다.
        6. **검색 결과 요약 후 출처 명시**: 답변 마지막에 '※ 위 정보는 웹 검색을 통해 제공되었습니다.' 문구를 추가해 주세요.

        질문: {query}
        검색 결과:
        {search_results_str[:4000]}
        
        답변:
        """
        
        try:
            final_answer = llm.invoke(summary_prompt)
        except Exception as e:
            final_answer = f"웹 검색 결과를 요약하는 도중 오류가 발생했습니다: {e}"
            
    return {"pred_answer": final_answer, "source": "web_search"}

def select_single_crop_from_recommendations(crop_recommendations, llm):
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
    
    상세 분석할 작물: """
    
    try:
        selected_crop = llm.invoke(selection_prompt).strip()
        
        if selected_crop in ["없음", "", "None", "null"]:
            return ""
        
        cleaned_crop = selected_crop.split('\n')[0].split('.')[0].split(',')[0].strip()
        
        return cleaned_crop
        
    except Exception as e:
        return ""

# --- 5. LangGraph 노드 및 그래프 구조 ---

class RouterState(dict):
    query: str = ""
    selected_agents: list = []
    question_parts: dict = {}
    execution_order: list = []
    crop_info: str = ""
    selected_crop: str = ""
    agent_answers: dict = {}
    output: str = ""

def node_agent_select(state: RouterState) -> RouterState:
    result = simple_agent_selector(state["query"], llm)
    state["selected_agents"] = result["selected_agents"]
    state["question_parts"] = result.get("question_parts", {})
    state["execution_order"] = result["execution_order"]
    state["agent_answers"] = {}
    return state

def node_crop_recommend(state: RouterState) -> RouterState:
    if "작물추천_agent" not in state.get("selected_agents", []):
        return state
    
    question_parts = state.get("question_parts")
    question_part = question_parts.get("작물추천_agent", state["query"]) if question_parts else state["query"]
    answer = execute_agent_with_boundaries("작물추천_agent", question_part, llm)
    selected_crop = select_single_crop_from_recommendations(answer, llm)
    
    state["crop_info"] = answer
    state["selected_crop"] = selected_crop
    state["agent_answers"]["작물추천_agent"] = answer
    return state

def node_parallel_agents(state: RouterState) -> RouterState:
    existing_answers = state.get("agent_answers", {})
    answers = {}
    selected_agents = state.get("execution_order", [])
    selected_crop = state.get("selected_crop", "")
    question_parts = state.get("question_parts")

    for agent in selected_agents:
        if agent == "작물추천_agent":
            continue
        
        if question_parts:
            question_part = question_parts.get(agent, state["query"])
        else:
            question_part = state["query"]

        if selected_crop and selected_crop not in ["I don't know", "None", ""] and selected_crop not in question_part:
            question_part = f"{selected_crop} {question_part}"
        
        answer = execute_agent_with_boundaries(agent, question_part, llm)
        answers[agent] = answer
    
    state["agent_answers"] = {**existing_answers, **answers}
    return state

def node_merge_output(state: RouterState) -> RouterState:
    selected_agents = state.get("selected_agents", [])
    output = ""
    
    if len(selected_agents) == 1:
        agent = selected_agents[0]
        output = state["agent_answers"].get(agent, f"{agent} 실행 결과를 찾을 수 없습니다.")
    else:
        if "작물추천_agent" in state.get("agent_answers", {}) and state.get("crop_info"):
            output += f"**[작물추천 결과]**\n{state['crop_info']}\n"
            if state.get("selected_crop"): output += f"\n**[상세 분석 작물]**\n{state['selected_crop']}\n"
        
        for agent in state.get("execution_order", []):
            if agent != "작물추천_agent":
                answer = state["agent_answers"].get(agent, f"{agent} 실행 결과를 찾을 수 없습니다.")
                output += f"\n**[{agent}]**\n{answer}\n"
        
        if state.get("crop_info") and state.get("selected_crop"):
            output += f"\n---\n\n다른 추천 작물에 대한 정보가 궁금하시면, '{state['selected_crop']} 대신 [다른 작물명]에 대해 알려주세요'와 같이 질문해주세요."

    merged_output = output.strip()
    
    if len(selected_agents) > 1 and "기타" not in selected_agents:
        summary_prompt = f"아래는 여러 농업 에이전트의 답변입니다. 사용자에게 명확하고 자세하게 정리해서 한국어로만 알려주세요.\n\n{merged_output}\n\n"
        try:
            summary = llm.invoke(summary_prompt)
        except Exception as e:
            summary = f"요약 중 오류: {e}"
        state["output"] = summary.strip()
    else:
        state["output"] = merged_output
    return state

def judge_branch(state: RouterState) -> str:
    if "작물추천_agent" in state.get("selected_agents", []):
        return "crop_recommend"
    else:
        return "parallel_agents"

# 그래프 구조 정의
graph = StateGraph(RouterState)
graph.add_node("agent_select", node_agent_select)
graph.add_node("crop_recommend", node_crop_recommend)
graph.add_node("parallel_agents", node_parallel_agents)
graph.add_node("merge_output", node_merge_output)
graph.add_conditional_edges("agent_select", judge_branch, {"crop_recommend": "crop_recommend", "parallel_agents": "parallel_agents"})
graph.add_edge("crop_recommend", "parallel_agents")
graph.add_edge("parallel_agents", "merge_output")
graph.set_entry_point("agent_select")
app = graph.compile()

# --- 6. Streamlit 챗봇 UI 로직 ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 처리
if user_query := st.chat_input("궁금한 점을 입력해주세요."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성 중입니다... 🌱"):
            graph_state = RouterState(query=user_query)
            final_state = app.invoke(graph_state)
            final_answer = final_state.get('output', '답변을 생성하지 못했습니다.')
            
            st.write(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})