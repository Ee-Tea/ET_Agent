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

def build_agent_prompt(agent, user_question, context_info=None):
    """
    각 에이전트에게 명확한 역할과 경계를 제시하는 프롬프트 생성
    """
    base_prompts = {
        "작물추천_agent": f"""너는 작물추천_agent입니다. 다음 규칙을 엄격히 따라주세요:

[역할과 책임]
- 재배 환경(계절, 토양, 기후, 지역)에 맞는 작물/품종을 3-5개 추천
- 각 작물의 기본 특성(수확기, 맛, 저장성, 수익성) 정보 제공
- 추천 작물 중 가장 적합한 1개를 선별하여 상세 정보 제공
- 구체적인 재배 방법은 언급하지 말 것

[추천 형식]
1. [추천 작물 목록] - 3-5개 작물 나열
2. [상세 분석 작물] - 가장 적합한 1개 작물 상세 설명
3. [다른 작물 정보] - "다른 작물에 대한 정보도 궁금하시다면 질문해주세요" 안내

[제한사항]
- 재배 방법, 병해충 방제, 수확 방법 등은 절대 답변하지 말 것
- 다른 에이전트가 담당할 내용이 질문에 포함되어 있으면 "이 부분은 작물재배_agent에게 문의하세요"라고 안내
- 웹 검색이 필요한 경우 "추가 정보가 필요합니다. 웹 검색을 통해 최신 정보를 확인하겠습니다"라고 명시

질문: {user_question}""",

        "작물재배_agent": f"""너는 작물재배_agent입니다. 다음 규칙을 엄격히 따라주세요:

[역할과 책임]
- 이미 결정된 작물의 구체적인 재배/관리 방법만 담당
- 씨앗/모종 심기, 이랑 만들기, 솎음, 시비, 병해충 방제, 수확 방법 등
- 작물별 구체적인 관리 일정과 방법

[제한사항]
- 작물 추천은 하지 말 것
- 가격, 판매처 정보는 언급하지 말 것
- 기후 재해 대응은 기본적인 것만 언급하고, 전문적인 재해 대응은 재해_agent에게 안내
- 웹 검색이 필요한 경우 "추가 정보가 필요합니다. 웹 검색을 통해 최신 정보를 확인하겠습니다"라고 명시

질문: {user_question}""",

        "재해_agent": f"""너는 재해_agent입니다. 다음 규칙을 엄격히 따라주세요:

[역할과 책임]
- 기후 재해(폭염, 한파, 가뭄, 집중호우, 홍수) 예방 및 대응 방법만 담당
- 재해 발생 전/중/후의 구체적인 조치 방법
- 작물별 재해 대응 전략

[제한사항]
- 작물 추천은 하지 말 것
- 일반적인 재배 방법은 언급하지 말 것
- 가격, 판매처 정보는 언급하지 말 것
- 웹 검색이 필요한 경우 "추가 정보가 필요합니다. 웹 검색을 통해 최신 정보를 확인하겠습니다"라고 명시

질문: {user_question}""",

        "판매처_agent": f"""너는 판매처_agent입니다. 다음 규칙을 엄격히 따라주세요:

[역할과 책임]
- 농산물 판매처, 가격, 시세, 유통 정보만 담당
- 실시간 시세, 가격 변동 추이, 거래 방법
- 지역별 주요 판매처와 도매상 정보

[제한사항]
- 작물 추천은 하지 말 것
- 재배 방법은 언급하지 말 것
- 재해 대응은 언급하지 말 것
- 웹 검색이 필요한 경우 "추가 정보가 필요합니다. 웹 검색을 통해 최신 정보를 확인하겠습니다"라고 명시

질문: {user_question}"""
    }
    
    prompt = base_prompts.get(agent, f"너는 {agent}입니다. 질문: {user_question}")
    
    # 컨텍스트 정보가 있으면 추가
    if context_info:
        prompt += f"\n\n[참고 정보]\n{context_info}"
    
    return prompt

def split_question_by_agents(user_question, llm, embedding_model, agent_descriptions):
    """
    복합 질문을 각 에이전트가 담당할 부분으로 명확히 분리
    """
    # 1. 질문 분석하여 필요한 에이전트와 각각의 역할 파악
    analysis_prompt = f"""
    다음 사용자 질문을 분석하여 각 에이전트가 담당해야 할 부분을 명확히 분리해주세요.

    [에이전트 역할]
    1) 작물추천_agent: 재배 환경에 맞는 작물/품종 추천
    2) 작물재배_agent: 구체적인 재배/관리 방법
    3) 재해_agent: 기후 재해 예방 및 대응
    4) 판매처_agent: 판매처, 가격, 시세 정보

    [분리 규칙]
    - 각 에이전트가 담당할 구체적인 질문 부분을 명시
    - 중복되는 부분이 있으면 명확히 구분
    - 웹 검색이 필요한 부분이 있으면 명시

    질문: "{user_question}"

    다음 JSON 형식으로 답변해주세요:
    {{
        "selected_agents": ["에이전트명1", "에이전트명2"],
        "question_parts": {{
            "에이전트명1": "해당 에이전트가 답변할 구체적인 질문",
            "에이전트명2": "해당 에이전트가 답변할 구체적인 질문"
        }},
        "web_search_needed": ["웹 검색이 필요한 부분1", "웹 검색이 필요한 부분2"],
        "execution_order": ["에이전트명1", "에이전트명2"]
    }}
    """
    
    try:
        analysis_result = llm.invoke(analysis_prompt)
        # JSON 파싱 (간단한 파싱 로직)
        import json
        import re
        
        # JSON 부분 추출
        json_match = re.search(r'\{.*\}', analysis_result, re.DOTALL)
        if json_match:
            parsed_result = json.loads(json_match.group())
            return parsed_result
        else:
            # JSON 파싱 실패 시 기본 라우팅
            return {
                "selected_agents": ["기타"],
                "question_parts": {"기타": user_question},
                "web_search_needed": [],
                "execution_order": ["기타"]
            }
    except Exception as e:
        print(f"질문 분석 실패: {e}")
        # 기본 라우팅으로 fallback
        return {
            "selected_agents": ["기타"],
            "question_parts": {"기타": user_question},
            "web_search_needed": [],
            "execution_order": ["기타"]
        }

def execute_agent_with_boundaries(agent_name, question_part, llm, context_info=None):
    """
    각 에이전트를 명확한 경계 내에서 실행
    """
    agent_func = agent_functions.get(agent_name)
    if not agent_func:
        return f"{agent_name} 실행 함수가 연결되어 있지 않습니다."
    
    # 명확한 역할 제한이 포함된 프롬프트 생성
    agent_prompt = build_agent_prompt(agent_name, question_part, context_info)
    
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

from 작물추천.crop65pdfllm import run as crop_recommend_run
from 재배방법.crop_overall import run as crop_cultivation_run
from 재해대응.verification_search import run as disaster_run
from sales.SalesAgent import run as market_run
# from agents.etc_agent import run as etc_run

agent_functions = {
    "작물추천_agent": crop_recommend_run,
    "작물재배_agent": crop_cultivation_run,
    "재해_agent": disaster_run,
    "판매처_agent": market_run,
#     "기타": etc_run
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
    context_info = ""
    
    for agent in question_analysis['execution_order']:
        if agent in question_analysis['question_parts']:
            question_part = question_analysis['question_parts'][agent]
            print(f"\n=== {agent} 실행 ===")
            print(f"담당 질문: {question_part}")
            
            # 이전 에이전트의 답변이 있으면 컨텍스트로 전달
            if context_info:
                print(f"컨텍스트 정보: {context_info[:100]}...")
            
            answer = execute_agent_with_boundaries(agent, question_part, llm, context_info)
            all_answers[agent] = answer
            
            # 다음 에이전트를 위한 컨텍스트 업데이트
            context_info += f"\n{agent} 답변: {answer[:200]}..."
    
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

from langgraph.graph import StateGraph, END

class RouterState(dict):
    query: str = ""
    selected_agents: list = []
    question_parts: dict = {}
    web_search_needed: list = []
    execution_order: list = []
    crop_info: str = ""
    selected_crop: str = ""  # 선택된 단일 작물 추가
    agent_answers: dict = {}
    output: str = ""

def select_single_crop_from_recommendations(crop_recommendations, llm):
    """
    작물추천 결과에서 상세 분석할 작물 하나를 선택하는 함수
    """
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
        selected_crop = llm.invoke(selection_prompt).strip()
        
        # 응답 정리 및 검증
        cleaned_crop = clean_crop_name(selected_crop)
        
        # 검증: 작물명이 너무 길거나 설명이 포함된 경우 재시도
        if len(cleaned_crop) > 10 or "에 대해" in cleaned_crop or "관련" in cleaned_crop:
            print(f"[경고] 첫 번째 시도 결과가 부적절함: '{cleaned_crop}'")
            return retry_crop_selection(crop_recommendations, llm)
        
        return cleaned_crop
        
    except Exception as e:
        print(f"작물 선택 중 오류: {e}")
        return fallback_crop_selection(crop_recommendations)

def clean_crop_name(crop_text):
    """
    작물명 텍스트를 정리하는 함수
    """
    # 줄바꿈, 마침표, 쉼표 등으로 구분
    crop_text = crop_text.split('\n')[0].split('.')[0].split(',')[0].strip()
    
    # 괄호나 특수문자 제거
    import re
    crop_text = re.sub(r'[\(\)\[\]\{\}]', '', crop_text)
    
    # 숫자나 단위 제거 (예: "무 1kg" -> "무")
    crop_text = re.sub(r'\s*\d+.*$', '', crop_text)
    
    return crop_text.strip()

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
    print("[대체 방법] 텍스트에서 작물명 패턴 찾기")
    
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
    
    for pattern in crop_patterns:
        matches = re.findall(pattern, crop_recommendations)
        if matches:
            # 가장 긴 매치를 선택 (더 구체적인 작물명)
            selected = max(matches, key=len)
            print(f"[대체 방법] 패턴 매치: {selected}")
            return selected
    
    # 패턴 매치가 없는 경우, 첫 번째 한글 단어 반환
    korean_words = re.findall(r'[가-힣]+', crop_recommendations)
    if korean_words:
        fallback = korean_words[0]
        print(f"[대체 방법] 첫 번째 한글 단어: {fallback}")
        return fallback
    
    # 최후의 수단
    print("작물명 찾기 실패")
    return None

def node_input(state: RouterState) -> RouterState:
    user_input = input("\n사용자 입력 ('종료' 입력 시 종료): ").strip()
    if user_input == "종료":
        state["exit"] = True
        return state
    state["query"] = user_input
    return state

def node_agent_select(state: RouterState) -> RouterState:
    result = split_question_by_agents(state["query"], llm, embedding_model, agent_descriptions)
    state["selected_agents"] = result["selected_agents"]
    state["question_parts"] = result["question_parts"]
    state["web_search_needed"] = result["web_search_needed"]
    state["execution_order"] = result["execution_order"]
    
    print("\n[선택된 에이전트]")
    for agent in state["selected_agents"]:
        print(f"- {agent}")
    
    if state["web_search_needed"]:
        print(f"\n[웹 검색 필요]")
        for item in state["web_search_needed"]:
            print(f"- {item}")
    
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
    print(f"\n[병렬 실행] 선택된 작물: {selected_crop}")
    
    for agent in state.get("execution_order", []):
        if agent == "작물추천_agent":
            continue  # 이미 실행됨
        
        if agent in state.get("question_parts", {}):
            question_part = state["question_parts"][agent]
            print(f"\n{agent} 실행 - 원본 질문: {question_part}")
            
            # 작물추천 결과와 선택된 작물을 컨텍스트에 포함
            context_info = ""
            if state.get("crop_info"):
                context_info += f"추천 작물 정보: {state['crop_info']}\n"
            if selected_crop:
                context_info += f"선택된 작물: {selected_crop}\n"
                
                # 질문에 작물명이 명시되지 않은 경우 추가
                if selected_crop not in question_part:
                    # 에이전트별로 적절한 질문 형태로 수정
                    if agent == "작물재배_agent":
                        question_part = f"{selected_crop}의 재배 방법과 관리법에 대해 알려주세요."
                    elif agent == "재해_agent":
                        question_part = f"{selected_crop} 재배 시 발생할 수 있는 기후 재해와 대응 방법을 알려주세요."
                    elif agent == "판매처_agent":
                        question_part = f"{selected_crop}의 판매처와 시세 정보를 알려주세요."
                    else:
                        question_part = f"{selected_crop}에 대한 {question_part}"
            
            print(f"컨텍스트 정보: {context_info}")
            print(f"수정된 질문: {question_part}")
            
            # 명확한 경계가 설정된 프롬프트로 실행
            answer = execute_agent_with_boundaries(agent, question_part, llm, context_info)
            answers[agent] = answer
        else:
            answers[agent] = f"{agent}에 대한 구체적인 질문이 정의되지 않았습니다."
    
    state["agent_answers"] = answers
    return state

def node_merge_output(state: RouterState) -> RouterState:
    output = ""
    
    # 작물추천 결과가 있으면 먼저 표시
    if state.get("crop_info"):
        output += f"[작물추천 결과]\n{state['crop_info']}\n"
        
        # 선택된 작물 강조 표시
        if state.get("selected_crop"):
            output += f"\n[상세 분석 작물]\n{state['selected_crop']}\n"
            print(f"\n[상세 분석 작물 확인] {state['selected_crop']}")
    
    # 다른 에이전트들의 답변 표시
    for agent, answer in state.get("agent_answers", {}).items():
        if agent != "작물추천_agent":  # 이미 표시됨
            # 선택된 작물과 답변의 일관성 확인
            selected_crop = state.get("selected_crop", "")
            if selected_crop and selected_crop in answer:
                output += f"[{agent} 결과 - {selected_crop} 관련]\n{answer}\n"
            else:
                output += f"[{agent} 결과]\n{answer}\n"
    
    # 웹 검색이 필요한 부분이 있으면 안내
    if state.get("web_search_needed"):
        output += f"\n[웹 검색 필요]\n"
        for item in state["web_search_needed"]:
            output += f"- {item}\n"
        output += "웹 검색 노드를 통해 최신 정보를 확인하겠습니다.\n"
    
    # 다른 작물 정보 안내 추가
    if state.get("crop_info") and state.get("selected_crop"):
        output += f"\n[추가 정보 안내]\n"
        output += f"다른 추천 작물에 대한 상세 정보가 궁금하시다면, "
        output += f"'{state['selected_crop']} 대신 [작물명]에 대해 알려주세요'와 같이 질문해주세요.\n"
    
    merged_output = output.strip()
    print("\n=== 최종 응답(병합 전) ===\n" + merged_output)

    # LLM에게 전체 응답을 정리하도록 요청
    summary_prompt = (
        "아래는 여러 농업 에이전트의 답변입니다. 사용자가 이해하기 쉽도록 정리해서 알려주세요.\n\n"
        f"{merged_output}"
    )
    try:
        summary = llm.invoke(summary_prompt)
    except Exception as e:
        summary = f"요약 중 오류: {e}"

    state["output"] = summary.strip()
    print("\n=== 최종 응답(요약) ===\n" + state["output"])
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

def test_improved_routing():
    """개선된 라우팅 시스템 테스트"""
    print("=== 개선된 라우팅 시스템 테스트 ===")
    
    test_questions = [
        "여름에 키우기 좋은 작물 추천과 해당 작물을 키울 때 고려해야 할 점에 대해서 알고 싶어요.",
        "장마철 집중호우에 대비해 감자밭에서는 어떤 관리가 필요할까요?",
        "토마토를 재배하고 있는데, 수확 후 어디에 팔면 좋을지 가격도 함께 알려주세요."
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 테스트 {i} ---")
        print(f"질문: {question}")
        
        # 개선된 라우팅 시스템 테스트
        result = hybrid_router(question, embedding_model, agent_descriptions, llm)
        print(f"\n결과:\n{result}")
        print("-" * 50)

if __name__ == "__main__":
    # 테스트 모드와 일반 모드 선택
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_improved_routing()
    else:
        run_orchestrator_langgraph()
