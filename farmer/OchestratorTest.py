from sentence_transformers import SentenceTransformer, util
import requests
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
llm = GroqLLM(api_key = os.getenv("OPENAI_KEY1"))

def simple_agent_selector(user_question, llm):
    """
    사용자 질문을 분석하여 필요한 에이전트를 선택하는 함수
    """
    selection_prompt = f"""
    다음 질문을 분석하여 필요한 에이전트를 선택해주세요.
    
    [에이전트 역할]
    1) 작물추천_agent: 재배 환경에 맞는 작물/품종 추천
    2) 작물재배_agent: 구체적인 재배/관리 방법
    3) 재해_agent: 기후 재해 예방 및 대응
    4) 판매처_agent: 판매처, 가격, 시세 정보
    5) 기타: 농업과 무관한 질문
    
    [판매처 에이전트 세부 분류]
    - 시세만: "가격", "시세", "얼마" 등의 키워드만 포함
    - 판매처만: "판매처", "어디서", "장소" 등의 키워드만 포함  
    - 둘 다: 시세와 판매처 관련 키워드가 모두 포함
    
    질문: "{user_question}"
    
    다음 JSON 형식으로 답변해주세요:
    {{
        "selected_agents": ["에이전트명1"],
        "question_parts": {{
            "에이전트명1": "해당 에이전트가 답변할 구체적인 질문"
        }},
        "execution_order": ["에이전트명1"]
    }}
    """
    
    try:
        result = llm.invoke(selection_prompt)
        import json
        import re
        
        # JSON 부분 추출
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # JSON 파싱 실패 시 기본값
            return {
                "selected_agents": ["기타"],
                "question_parts": {"기타": user_question},
                "execution_order": ["기타"]
            }
    except Exception as e:
        print(f"에이전트 선택 실패: {e}")
        return {
            "selected_agents": ["기타"],
            "question_parts": {"기타": user_question},
            "execution_order": ["기타"]
        }

def build_agent_prompt(agent, user_question):
    """
    각 에이전트에게 질문만 전달
    """
    prompt = f"질문: {user_question}"
    
    return prompt

def execute_agent_with_boundaries(agent_name, question_part, llm):
    """
    각 에이전트를 명확한 경계 내에서 실행
    """
    agent_func = agent_functions.get(agent_name)
    if not agent_func:
        return f"{agent_name} 실행 함수가 연결되어 있지 않습니다."
    
    # 명확한 역할 제한이 포함된 프롬프트 생성
    agent_prompt = build_agent_prompt(agent_name, question_part)
    
    try:
        agent_state = {"query": agent_prompt}
        agent_result = agent_func(agent_state)
        answer = agent_result.get("pred_answer", "답변 생성 실패")
        
        # 답변에서 다른 에이전트 영역 침범 여부 확인
        boundary_check_prompt = f"""
        다음 답변이 {agent_name}의 역할 범위를 벗어나지 않았는지 확인해주세요.
        
        [에이전트 역할]
        {agent_name}: {agent_descriptions.get(agent_name, '')}
        
        [답변 내용]
        {answer}
        
        [확인 사항]
        1. 답변이 {agent_name}의 전문 영역에만 집중했는가?
        2. 다른 에이전트가 담당해야 할 내용을 포함하지 않았는가?
        3. 웹 검색이 필요한 부분을 명시했는가?
        
        문제가 있다면 수정된 답변을 제공하고, 문제가 없다면 "OK"라고 답변해주세요.
        """
        
        boundary_check = llm.invoke(boundary_check_prompt)
        if "OK" not in boundary_check:
            # 경계를 벗어난 답변 수정
            answer = boundary_check
        
        return answer
        
    except Exception as e:
        return f"에이전트 실행 중 오류: {e}"

def web_search_with_tavily(query: str, api_key: str = None):
    """
    Tavily를 이용한 웹 검색
    """
    try:
        from tavily import TavilyClient
        
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

def etc_agent_run(state: dict) -> dict:
    """
    기타 에이전트 - 웹 검색을 통한 답변
    """
    query = state.get("query", "")
    
    # 웹 검색 실행
    print(f"[기타_agent] 웹 검색 시작: {query}")
    web_result = web_search_with_tavily(query)
    
    # 결과 정리
    if "오류" in web_result or "설정되지 않음" in web_result:
        final_answer = f"질문: {query}\n\n{web_result}"
    else:
        final_answer = f"질문: {query}\n\n{web_result}\n\n※ 위 정보는 웹 검색을 통해 제공되었습니다."
    
    return {
        "pred_answer": final_answer,
        "source": "web_search"
    }

from 작물추천.crop65pdfllm import run as crop_recommend_run
from 재배방법.crop_overall import run as crop_cultivation_run
from 재해대응.verification_search import run as disaster_run
from sales.SalesAgent import run as market_run

agent_functions = {
    "작물추천_agent": crop_recommend_run,
    "작물재배_agent": crop_cultivation_run,
    "재해_agent": disaster_run,
    "판매처_agent": market_run,
    "기타": etc_agent_run
}

def hybrid_router(text, model, agent_desc, llm):
    print("=== 개선된 라우팅 시스템 ===")
    
    # 1. 질문을 각 에이전트가 담당할 부분으로 분리
    question_analysis = split_question_by_agents(text, llm, model, agent_desc)
    
    print(f"[질문 분석 결과]")
    print(f"선택된 에이전트: {question_analysis['selected_agents']}")
    print(f"질문 분리: {question_analysis['question_parts']}")
    if question_analysis['web_search_needed']:
        print(f"웹 검색 필요: {question_analysis['web_search_needed']}")
    
    # 2. 각 에이전트를 순서대로 실행
    all_answers = {}
    
    for agent in question_analysis['execution_order']:
        if agent in question_analysis['question_parts']:
            question_part = question_analysis['question_parts'][agent]
            print(f"\n=== {agent} 실행 ===")
            print(f"담당 질문: {question_part}")
            
            answer = execute_agent_with_boundaries(agent, question_part, llm)
            all_answers[agent] = answer
    
    # 3. 웹 검색이 필요한 부분이 있으면 안내
    web_search_info = ""
    if question_analysis['web_search_needed']:
        web_search_info = f"\n\n[웹 검색 필요]\n"
        for item in question_analysis['web_search_needed']:
            web_search_info += f"- {item}\n"
        web_search_info += "웹 검색 노드를 통해 최신 정보를 확인하겠습니다."
    
    # 4. 최종 응답 구성
    final_response = "=== 에이전트별 답변 ===\n"
    for agent, answer in all_answers.items():
        final_response += f"\n[{agent}]\n{answer}\n"
    
    final_response += web_search_info
    
    return final_response

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

from langgraph.graph import StateGraph

class RouterState(dict):
    query: str = ""
    selected_agents: list = []
    question_parts: dict = {}
    execution_order: list = []
    crop_info: str = ""
    selected_crop: str = ""  # 선택된 단일 작물 추가
    agent_answers: dict = {}
    output: str = ""

def select_single_crop_from_recommendations(crop_recommendations, llm):
    """
    작물추천 결과에서 상세 분석할 작물 하나를 선택하는 함수
    """
    print("\n=== �� 작물 추출 과정 시작 ===")
    print(f"[원본 작물추천 응답]\n{crop_recommendations}")
    
    selection_prompt = f"""
    다음은 작물추천 에이전트가 추천한 작물들입니다. 
    사용자의 질문과 상황을 고려하여 상세 분석할 작물 하나를 선택해주세요.
    
    [추천 작물 목록]
    {crop_recommendations}
    
    [선택 규칙]
    1. 사용자 질문과 가장 관련성이 높은 작물 선택
    2. 계절, 지역, 재배 난이도 등을 고려
    3. 반드시 작물명만 간단하게 답변
    4. 설명이나 다른 문장은 포함하지 말 것
    
    [응답 형식]
    상세 분석할 작물명만 작성하세요. 예시: 무, 토마토, 고추, 오이
    
    상세 분석할 작물: """
    
    try:
        print("[1단계] LLM에게 작물 선택 요청...")
        selected_crop = llm.invoke(selection_prompt).strip()
        print(f"[LLM 원본 응답] {selected_crop}")
        
        # 응답 정리 및 검증
        print("[2단계] 작물명 정리 및 검증...")
        cleaned_crop = clean_crop_name(selected_crop)
        print(f"[정리된 작물명] {cleaned_crop}")
        
        # 검증: 작물명이 너무 길거나 설명이 포함된 경우 재시도
        if len(cleaned_crop) > 10 or "에 대해" in cleaned_crop or "관련" in cleaned_crop:
            print(f"[경고] 첫 번째 시도 결과가 부적절함: '{cleaned_crop}'")
            print("[3단계] 재시도 시작...")
            return retry_crop_selection(crop_recommendations, llm)
        
        print(f"[✅ 최종 선택된 작물] {cleaned_crop}")
        return cleaned_crop
        
    except Exception as e:
        print(f"[❌ 오류] 작물 선택 중 오류: {e}")
        return fallback_crop_selection(crop_recommendations)

def clean_crop_name(crop_text):
    """
    작물명 텍스트를 정리하는 함수
    """
    print(f"[정리 전] {crop_text}")
    
    # 줄바꿈, 마침표, 쉼표 등으로 구분
    crop_text = crop_text.split('\n')[0].split('.')[0].split(',')[0].strip()
    
    # 괄호나 특수문자 제거
    import re
    crop_text = re.sub(r'[\(\)\[\]\{\}]', '', crop_text)
    
    # 숫자나 단위 제거 (예: "무 1kg" -> "무")
    crop_text = re.sub(r'\s*\d+.*$', '', crop_text)
    
    final_result = crop_text.strip()
    print(f"[최종 정리 결과] {final_result}")
    return final_result

def retry_crop_selection(crop_recommendations, llm):
    """
    첫 번째 시도가 실패했을 때 재시도하는 함수
    """
    retry_prompt = f"""
    위의 작물 추천 결과에서 가장 적합한 작물명 하나만 정확히 추출해주세요.
    
    [작물 추천 내용]
    {crop_recommendations}
    
    [요구사항]
    - 작물명만 작성 (예: 무, 토마토, 고추)
    - 설명이나 문장은 절대 포함하지 말 것
    - 한 단어로 된 작물명만
    
    작물명: """
    
    try:
        retry_result = llm.invoke(retry_prompt).strip()
        return clean_crop_name(retry_result)
    except Exception as e:
        print(f"재시도 실패: {e}")
        return fallback_crop_selection(crop_recommendations)

def fallback_crop_selection(crop_recommendations):
    """
    모든 시도가 실패했을 때 사용하는 대체 방법
    """
    print("\n=== �� 대체 방법: 패턴 매칭 시작 ===")
    
    # 텍스트에서 작물명 패턴 찾기
    import re
    
    # 일반적인 작물명 패턴 (한글 + 영문)
    crop_patterns = [
        r'([가-힣]+무)',      # 무, 봄무, 가을무 등
        r'([가-힣]+토마토)',   # 토마토, 방울토마토 등
        r'([가-힣]*고추)',     # 고추, 풋고추, 빨간고추 등
        r'([가-힣]*오이)',     # 오이, 가시오이 등
        r'([가-힣]*상추)',     # 상추, 적상추 등
        r'([가-힣]*배추)',     # 배추, 김장배추 등
        r'([가-힣]*양파)',     # 양파, 적양파 등
        r'([가-힣]*마늘)',     # 마늘, 단마늘 등
        r'([가-힣]*감자)',     # 감자, 새감자 등
        r'([가-힣]*고구마)',   # 고구마, 밤고구마 등
    ]
    
    print("[패턴 매칭 시도...]")
    for i, pattern in enumerate(crop_patterns):
        matches = re.findall(pattern, crop_recommendations)
        if matches:
            # 가장 긴 매치를 선택 (더 구체적인 작물명)
            selected = max(matches, key=len)
            print(f"[✅ 패턴 {i+1} 매치 성공] {selected}")
            return selected
    
    # 패턴 매치가 없는 경우, 첫 번째 한글 단어 반환
    print("[패턴 매칭 실패, 한글 단어 추출 시도...]")
    korean_words = re.findall(r'[가-힣]+', crop_recommendations)
    if korean_words:
        fallback = korean_words[0]
        print(f"[✅ 첫 번째 한글 단어] {fallback}")
        return fallback
    
    # 최후의 수단
    print("[❌ 모든 방법 실패]")
    return None

def node_input(state: RouterState) -> RouterState:
    user_input = input("\n사용자 입력: ").strip()
    
    # 유효한 입력인 경우 상태에 저장하고 다음 단계로
    state["query"] = user_input
    print(f"\n[사용자 입력] {user_input}")
    
    return state

def node_agent_select(state: RouterState) -> RouterState:
    # 기존 복잡한 로직을 단순화된 함수로 교체
    result = simple_agent_selector(state["query"], llm)
    state["selected_agents"] = result["selected_agents"]
    state["question_parts"] = result["question_parts"]
    state["web_search_needed"] = []  # 웹 검색 필요성 제거
    state["execution_order"] = result["execution_order"]
    
    print("\n[선택된 에이전트]")
    for agent in state["selected_agents"]:
        print(f"- {agent}")
    
    return state

def node_crop_recommend(state: RouterState) -> RouterState:
    if "작물추천_agent" not in state.get("selected_agents", []):
        return state
    
    print("\n=== 작물추천_agent 실행 ===")
    question_part = state["question_parts"].get("작물추천_agent", state["query"])
    print(f"담당 질문: {question_part}")
    
    # 명확한 경계가 설정된 프롬프트로 실행
    answer = execute_agent_with_boundaries("작물추천_agent", question_part, llm)
    
    print(f"\n[작물추천_agent 원본 응답]\n{answer}")
    
    # 작물추천 결과에서 하나의 작물 선택
    selected_crop = select_single_crop_from_recommendations(answer, llm)
    
    state["crop_info"] = answer
    state["selected_crop"] = selected_crop  # 선택된 단일 작물 저장
    
    print(f"\n[선택된 작물] {selected_crop}")
    print(f"[작물 선택 완료]")
    
    # agent_answers에 추가
    if "agent_answers" not in state:
        state["agent_answers"] = {}
    state["agent_answers"]["작물추천_agent"] = answer
    
    return state

def node_parallel_agents(state: RouterState) -> RouterState:
    answers = {}
    
    # 선택된 작물 정보 확인
    selected_crop = state.get("selected_crop", "")
    print(f"\n=== 병렬 에이전트 실행 시작 ===")
    print(f"[📌 선택된 작물] {selected_crop}")
    
    for agent in state.get("execution_order", []):
        if agent == "작물추천_agent":
            continue  # 이미 실행됨
        
        if agent in state.get("question_parts", {}):
            original_question = state["question_parts"][agent]
            print(f"\n--- {agent} 실행 ---")
            print(f"[📝 원본 질문] {original_question}")
            
            # 작물명이 유효하고 질문에 포함되지 않은 경우에만 추가
            if (selected_crop and 
                selected_crop not in ["I don't know", "None", ""] and 
                selected_crop not in original_question):
                
                print(f"[🔄 질문 수정 필요] 작물명 '{selected_crop}'이 질문에 포함되지 않음")
                question_part = f"{selected_crop} {original_question}"
                print(f"[🔧 질문 수정] 작물명 '{selected_crop}' 추가")
            else:
                print(f"[✅ 질문 수정 불필요] 원본 질문 사용")
                question_part = original_question
            
            print(f"[🎯 최종 질문] {question_part}")
            print(f"[📊 질문 길이] {len(question_part)}자")
            
            # 명확한 경계가 설정된 프롬프트로 실행
            print(f"[⚡ {agent} 실행 시작...]")
            answer = execute_agent_with_boundaries(agent, question_part, llm)
            answers[agent] = answer
            
            print(f"[✅ {agent} 실행 완료]")
            print(f"[ 답변 길이] {len(answer)}자")
            print(f"[ 답변 미리보기] {answer[:100]}...")
            
        else:
            answers[agent] = f"{agent}에 대한 구체적인 질문이 정의되지 않았습니다."
            print(f"[⚠️ {agent}] 구체적인 질문이 정의되지 않음")
    
    state["agent_answers"] = answers
    print(f"\n=== 모든 에이전트 실행 완료 ===")
    return state

def node_merge_output(state: RouterState) -> RouterState:
    print("\n=== 최종 응답 병합 시작 ===")
    
    # 실행 요약 출력
    print(f"[ 실행 요약]")
    print(f"  - 선택된 에이전트: {state.get('selected_agents', [])}")
    print(f"  - 선택된 작물: {state.get('selected_crop', '없음')}")
    print(f"  - 실행된 에이전트: {list(state.get('agent_answers', {}).keys())}")
    
    output = ""
    
    # 작물추천 결과가 있으면 먼저 표시
    if state.get("crop_info"):
        output += f"[작물추천 결과]\n{state['crop_info']}\n"
        
        # 선택된 작물 강조 표시
        if state.get("selected_crop"):
            output += f"\n[상세 분석 작물]\n{state['selected_crop']}\n"
            print(f"[ 상세 분석 작물] {state['selected_crop']}")
    
    # 다른 에이전트들의 답변 표시
    for agent, answer in state.get("agent_answers", {}).items():
        if agent != "작물추천_agent":  # 이미 표시됨
            # 선택된 작물과 답변의 일관성 확인
            selected_crop = state.get("selected_crop", "")
            if selected_crop and selected_crop in answer:
                output += f"[{agent} 결과 - {selected_crop} 관련]\n{answer}\n"
                print(f"[✅ {agent}] {selected_crop} 관련 답변 일치")
            else:
                output += f"[{agent} 결과]\n{answer}\n"
                print(f"[⚠️ {agent}] {selected_crop} 관련 답변 불일치")
    
    # 다른 작물 정보 안내 추가
    if state.get("crop_info") and state.get("selected_crop"):
        output += f"\n[추가 정보 안내]\n"
        output += f"다른 추천 작물에 대한 상세 정보가 궁금하시다면, "
        output += f"'{state['selected_crop']} 대신 [작물명]에 대해 알려주세요'와 같이 질문해주세요.\n"
    
    merged_output = output.strip()
    
    # LLM에게 전체 응답을 정리하도록 요청
    print("\n[🤖 LLM 요약 시작...]")
    summary_prompt = (
        "아래는 여러 농업 에이전트의 답변입니다. 사용자가 이해하기 쉽도록 정리해서 알려주세요.\n\n"
        f"{merged_output}\n\n"
        "간결하고 명확하게 정리해주세요."
    )
    
    try:
        summary = llm.invoke(summary_prompt)
        print(f"[✅ LLM 요약 완료] {len(summary)}자")
    except Exception as e:
        summary = f"요약 중 오류: {e}"
        print(f"[❌ LLM 요약 실패] {e}")
    
    state["output"] = summary.strip()
    
    # 최종 요약된 응답만 출력 (중복 제거)
    print("\n=== 🎯 최종 응답(요약) ===")
    print(f"[📊 요약 길이] {len(state['output'])}자")
    print("=" * 50)
    print(state["output"])
    print("=" * 50)
    
    return state

def judge_branch(state: RouterState) -> str:
    # 작물추천_agent가 선택된 경우 분기
    if "작물추천_agent" in state.get("selected_agents", []):
        return "crop_recommend"
    else:
        return "parallel_agents"

# 그래프 구조 정의
graph = StateGraph(RouterState)

# 노드 추가
graph.add_node("input", node_input)
graph.add_node("agent_select", node_agent_select)
graph.add_node("crop_recommend", node_crop_recommend)
graph.add_node("parallel_agents", node_parallel_agents)
graph.add_node("merge_output", node_merge_output)

# 엣지 추가 - 조건부 분기를 명확하게
graph.add_edge("input", "agent_select")
graph.add_conditional_edges(
    "agent_select",
    judge_branch,
    {
        "crop_recommend": "crop_recommend",
        "parallel_agents": "parallel_agents"
    }
)
graph.add_edge("crop_recommend", "parallel_agents")
graph.add_edge("parallel_agents", "merge_output")
graph.add_edge("merge_output", "input")
graph.set_entry_point("input")

def run_orchestrator_langgraph():
    app = graph.compile()
    try:
        graph_image_path = "ochestrator_workflow.png"
        with open(graph_image_path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")
    state = RouterState()
    app.invoke(state)

if __name__ == "__main__":
        run_orchestrator_langgraph()
