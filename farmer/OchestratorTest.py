from sentence_transformers import SentenceTransformer, util
import requests
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 2) 임베딩 모델 로드 (jhgan/ko-sroberta-multitask)
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# 3) 에이전트 설명 정의 (임베딩 라우팅용)
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

# 4) LLM
from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()

class GroqLLM:
    def __init__(self, model="openai/gpt-oss-20b", api_key=None):
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = Groq()
        self.model = model

    def invoke(self, prompt: str):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8,
            max_completion_tokens=2048,
            top_p=0.8,
            reasoning_effort="medium",
            stream=True,
            stop=None
        )
        result = ""
        for chunk in completion:
            result += chunk.choices[0].delta.content or ""
        return result.strip()

# 사용 예시 (api_key는 실제 발급받은 키로 입력)
llm = GroqLLM(api_key = os.getenv("OPENAI_KEY2"))

def embedding_router(text, model, agent_descriptions, threshold=0.5):
    q_vec = model.encode(text, convert_to_tensor=True)
    candidates = []
    
    # 유사도 점수와 에이전트 목록을 함께 저장
    for agent, desc in agent_descriptions.items():
        d_vec = model.encode(desc, convert_to_tensor=True)
        sim = util.cos_sim(q_vec, d_vec).item()
        print(f"[임베딩] {agent} 유사도: {sim:.4f}")
        
        # 임계값 이상의 후보만 리스트에 추가
        if sim >= threshold:
            candidates.append({"agent": agent, "score": sim})

    if not candidates:
        print("[임베딩] 임계값(0.5)을 넘는 후보 없음. LLM 라우팅으로 전환.")
        return None
        
    # 점수가 높은 순으로 정렬
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    print(f"[임베딩] 최종 선택 후보: {candidates}")
    return candidates

def llm_router(text, llm, embedding_hints=None):
    hint_agents_str = ""
    if embedding_hints:
        hint_agents = [hint['agent'] for hint in embedding_hints]
        hint_agents_str = ", ".join(hint_agents)
    
    prompt = f"""
        너는 농업 상담 AI 오케스트레이터야. 
        아래 선택지 중 사용자 질문에 가장 적합한 에이전트를 하나만 골라.

        [규칙]
        - 임베딩 라우터가 추천한 후보가 있으면 참조해서 그 목록 안에서 선택.
        - 후보가 있어도 모두 관련 없으면 '5) 기타' 선택.
        - 농업과 무관한 질문은 무조건 '5) 기타' 선택.
        - 추천 목록이 없으면 전체 선택지 중에서 선택.
        - 여러 에이전트가 필요할 거 같은 경우 여러 에이전트 선택.
        - 대답할 땐 선택지 번호와 agent명만 간단히 작성

        질문: "{text}"
        임베딩 라우터 추천 후보: {hint_agents_str}

        [에이전트 설명]
        1) 작물추천_agent: 재배 환경에 맞는 새로운 작물/품종 추천
        2) 작물재배_agent: 이미 결정된 작물의 재배/관리 정보 제공
        3) 재해_agent: 기후 재해(폭염, 가뭄 등) 및 이상 기후 대비/관리 정보 제공
        4) 판매처_agent: 농산물 판매처, 가격, 실시간 시세, 최근 가격 변동 안내
        5) 기타: 농업과 무관하거나 후보 중 적합한 에이전트가 없을 때

        [예시]
        질문: "장마철 집중호우에 대비해 감자밭에서는 어떤 관리가 필요할까요?"
        정답: 3) 재해_agent

        질문: "여름에 키우기 좋은 작물 추천과 해당 작물을 키울 때 고려해야 할 점에 대해서 알고 싶어요."
        정답: 1) 작물추천_agent, 2) 작물재배_agent
    """
    result = llm.invoke(prompt)
    print(f"[LLM 라우터] 선택된 agent: {result}")
    return result

def build_agent_prompt(agent, user_question):
    if agent == "작물추천_agent":
        return f"너는 작물추천_agent야. 아래 질문에서 작물 추천 관련 내용만 답변해.\n질문: {user_question}"
    elif agent == "작물재배_agent":
        return f"너는 작물재배_agent야. 아래 질문에서 재배/관리 정보만 답변해.\n질문: {user_question}"
    elif agent == "판매처_agent":
        return f"너는 판매처_agent야. 아래 질문에서 판매처, 가격, 시세 관련 정보만 답변해.\n질문: {user_question}"
    elif agent == "재해_agent":
        return f"너는 재해_agent야. 아래 질문에서 기후 재해/이상기후 관련 정보만 답변해.\n질문: {user_question}"
    else:
        return f"너는 기타_agent야. 아래 질문에서 농업과 무관한 내용만 답변해.\n질문: {user_question}"

def split_agents(user_question, llm, embedding_model, agent_descriptions):
    # 1. 어떤 agent가 실행될지 결정
    embedding_hints = embedding_router(user_question, embedding_model, agent_descriptions)
    agent_selection = llm_router(user_question, llm, embedding_hints)
    
    # 2. agent명 추출 (복수 가능)
    agent_names = []
    for part in agent_selection.split(","):
        if ")" in part:
            agent_names.append(part.split(")")[1].strip())
        else:
            agent_names.append(part.strip())
    agent_names = [name for name in agent_names if name in agent_descriptions]

    # 질문 분리 대신 agent 목록만 반환
    return {
        "selected_agents": agent_names,
        "original_question": user_question
    }

# from agents.crop_recommend_agent import run as crop_recommend_run
# from agents.crop_cultivation_agent import run as crop_cultivation_run
# from agents.disaster_agent import run as disaster_run
from sales.SalesAgent import run as market_run
# from agents.etc_agent import run as etc_run

agent_functions = {
#     "작물추천_agent": crop_recommend_run,
#     "작물재배_agent": crop_cultivation_run,
#     "재해_agent": disaster_run,
    "판매처_agent": market_run,
#     "기타": etc_run
}

def hybrid_router(text, model, agent_desc, llm):
    print("=== 임베딩 라우팅 ===")
    embedding_hints = embedding_router(text, model, agent_desc)
    print("=== LLM 라우팅 ===")
    result = llm_router(text, llm, embedding_hints)
    
    # agent명 추출 (예: "2) 작물재배_agent" → "작물재배_agent")
    agent_name = result.split(")")[1].strip() if ")" in result else result.strip()
    agent_func = agent_functions.get(agent_name)
    if agent_func is None:
        return f"{result}\n해당 agent({agent_name})에 대한 실행 함수가 없습니다."
    
    state = {"query": text}
    try:
        agent_output_state = agent_func(state)
        answer = agent_output_state.get("pred_answer", "답변 생성 실패")
    except Exception as e:
        answer = f"에이전트 실행 중 오류: {e}"
    return f"{result}\n{answer}"

def get_user_input():
    """사용자 입력을 받아 검증하는 함수"""
    while True:
        user_input = input("\n사용자 입력 ('종료' 입력 시 종료): ").strip()
        if not user_input:
            print("입력이 비어 있습니다. 다시 입력해주세요.")
            continue
        return user_input

def main():
    print("=== 하이브리드 라우터 데모 ===")
    while True:
        user_input = get_user_input()
        if user_input == "종료":
            print("종료합니다.")
            break
        selected_agent = hybrid_router(user_input, embedding_model, agent_descriptions, llm)
        print(f"선택된 에이전트: {selected_agent}")

from langgraph.graph import StateGraph, END

class RouterState(dict):
    query: str = ""
    selected_agents: list = []
    split_questions: dict = {}
    crop_info: str = ""
    agent_answers: dict = {}
    output: str = ""

def node_input(state: RouterState) -> RouterState:
    user_input = input("\n사용자 입력 ('종료' 입력 시 종료): ").strip()
    if user_input == "종료":
        state["exit"] = True
        return state
    state["query"] = user_input
    return state

def node_agent_select(state: RouterState) -> RouterState:
    result = split_agents(state["query"], llm, embedding_model, agent_descriptions)
    state["selected_agents"] = result["selected_agents"]
    print("\n[선택된 에이전트]")
    for agent in state["selected_agents"]:
        print(f"- {agent}")
    return state

def node_crop_recommend(state: RouterState) -> RouterState:
    print("작물추천_agent 실행")
    crop_func = agent_functions.get("작물추천_agent")
    if crop_func:
        agent_prompt = build_agent_prompt("작물추천_agent", state["query"])
        crop_state = {"query": agent_prompt}
        crop_result = crop_func(crop_state)
        crop_info = crop_result.get("pred_answer", "")
        state["crop_info"] = crop_info
        print(f"[작물추천_agent 응답]\n{crop_info}")
        # 작물추천_agent도 answers에 추가
        if "agent_answers" not in state:
            state["agent_answers"] = {}
        state["agent_answers"]["작물추천_agent"] = crop_info
    else:
        state["crop_info"] = ""
        print("[작물추천_agent] 실행 함수가 연결되어 있지 않습니다.")
        if "agent_answers" not in state:
            state["agent_answers"] = {}
        state["agent_answers"]["작물추천_agent"] = "작물추천_agent 실행 함수가 연결되어 있지 않습니다."
    return state

def node_parallel_agents(state: RouterState) -> RouterState:
    answers = {}
    for agent in state["selected_agents"]:
        if agent == "작물추천_agent":
            continue
        agent_func = agent_functions.get(agent)
        print(f"{agent} 실행")
        if agent_func:
            agent_prompt = build_agent_prompt(agent, state["query"])
            # 작물추천 결과가 있으면 context에 포함
            if state.get("crop_info"):
                agent_prompt += f"\n추천 작물 정보: {state['crop_info']}"
            agent_state = {"query": agent_prompt}
            agent_result = agent_func(agent_state)
            answers[agent] = agent_result.get("pred_answer", "")
        else:
            answers[agent] = f"{agent} 실행 함수가 연결되어 있지 않습니다."
    state["agent_answers"] = answers
    return state

def node_merge_output(state: RouterState) -> RouterState:
    output = ""
    if state.get("crop_info"):
        output += f"[작물추천 결과]\n{state['crop_info']}\n"
    for agent, answer in state.get("agent_answers", {}).items():
        output += f"[{agent} 결과]\n{answer}\n"
    state["output"] = output.strip()
    print("\n=== 최종 응답 ===\n" + state["output"])
    return state

def judge_branch(state: RouterState) -> str:
    # 작물추천_agent가 선택된 경우 분기
    if "작물추천_agent" in state.get("selected_agents", []):
        return "crop_recommend"
    else:
        return "parallel_agents"

graph = StateGraph(RouterState)
graph.add_node("input", node_input)
graph.add_node("agent_select", node_agent_select)
graph.add_node("crop_recommend", node_crop_recommend)
graph.add_node("parallel_agents", node_parallel_agents)
graph.add_node("merge_output", node_merge_output)

graph.add_edge("input", "agent_select")
graph.add_conditional_edges("agent_select", judge_branch)
graph.add_edge("crop_recommend", "parallel_agents")
graph.add_edge("parallel_agents", "merge_output")
graph.add_edge("merge_output", END)
graph.set_entry_point("input")

def run_orchestrator_langgraph():
    app = graph.compile()
    while True:
        state = RouterState()
        app.invoke(state)
        if state.get("exit"):
            print("종료합니다.")
            break

if __name__ == "__main__":
    run_orchestrator_langgraph()
