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
    "판매처_agent": "판매처 위치와 농산물 가격, 유통 경로를 안내합니다.",
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
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="default",
            stream=True,
            stop=None
        )
        result = ""
        for chunk in completion:
            result += chunk.choices[0].delta.content or ""
        return result.strip()

# 사용 예시 (api_key는 실제 발급받은 키로 입력)
llm = GroqLLM(api_key = os.getenv("OPENAI_KEY"))

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
        너는 농업 상담 AI 오케스트레이터야. 사용자 질문의 핵심 의도에 가장 적절한 전문 에이전트를 아래 선택지에서 하나만 골라.
        질문과 가장 관련된 주제, 전문성, 답변 가능성 등을 종합적으로 고려해 정확한 판단을 내려야 해.
        각 에이전트는 서로 다른 전문 영역을 담당하며, 절대 겹치지 않아.

        <핵심 규칙>
        - **가장 중요한 규칙:** 임베딩 라우터가 추천한 후보 에이전트 목록이 있다면, 반드시 그 목록 안에서만 가장 적절한 에이전트를 선택해야 합니다.
        - 만약 후보 목록이 존재하지만 그 어떤 후보도 질문과 관련이 없다고 판단될 때만 '5) 기타'를 선택하세요.
        - 추천 목록이 없다면, 전체 선택지에서 가장 적절한 에이전트를 선택하세요.
        - 농업과 전혀 관련 없는 질문이라면 무조건 '5) 기타'를 선택해야 합니다.

        사용자 질문:
        "{text}"

        에이전트 설명:
        1) 작물추천_agent: 사용자의 재배 환경(계절, 토양, 기후 등), 목적, **특정 조건(수확 시기, 맛, 저장성 등)**에 맞는 **새로운 작물이나 품종을 추천**합니다.
        ※ 핵심 키워드: "어떤 작물을 심을까", "무엇을 재배하면 좋을까", "추천해주세요"

        2) 작물재배_agent: **이미 결정된 작물의 재배 방법, 심는 방법, 이랑, 솎음, 영양 관리(시비, 비료, 거름), 병해충 방제, 수확 등** 관리 정보를 제공합니다.
        ※ 핵심 키워드: "심는 방법", "재배", "키울 때", "이랑", "간격", "솎음", "병해충", "거름", "비료", "수확", "어떻게"

        3) 재해_agent: **폭염, 한파, 가뭄, 홍수 등 기후로 인한 피해와 그 예방, 대비, 관리 방법을 제공**합니다.
        ※ 핵심 키워드: "기온", "폭염", "이상 기상", "가뭄", "자연재해", "호우", "폭우", "대비", "장마", "피해"

        4) 판매처_agent: 농산물을 어디에 팔 수 있는지, 유통 경로, 가격, 판매처 정보 등을 안내합니다.

        5) 기타: 위의 1~4번 에이전트의 전문 분야인 **농업과 전혀 관련 없는 질문**이거나, **후보 목록 중 적합한 에이전트가 없을 때** 선택합니다.

        <임베딩 라우터 추천 후보>
        - {hint_agents_str}

        예시:
        - 질문: "장마철 집중호우에 대비해 감자밭에서는 어떤 관리가 필요할까요?"
          - 분석: 이 질문의 핵심은 '장마철', '집중호우'와 같은 기후 재해에 대한 **대비 관리**이므로, 재해_agent가 가장 적절하다.
          - 정답: 3) 재해_agent

        - 질문: "아스파라거스를 키울 때 고려해야 할 점에 대해서 알고 싶어요."
          - 분석: '키울 때 고려해야 할 점'은 이미 결정된 작물의 전반적인 재배 방법에 해당한다.
          - 정답: 2) 작물재배_agent

        - 질문: "화장실 가고 싶다"
          - 분석: 농업과 전혀 관련 없는 질문이다.
          - 정답: 5) 기타
        
        선택지: 1) 작물추천_agent, 2) 작물재배_agent, 3) 재해_agent, 4) 판매처_agent, 5) 기타

        정답(선택지 번호와 agent명만 간단히):
    """
    return llm.invoke(prompt)

def hybrid_router(text, model, agent_desc, llm):
    # 1. 임베딩 유사도 기반
    print("=== 임베딩 라우팅 ===")
    embedding_hints = embedding_router(text, model, agent_desc)
    
    # 2. LLM에게 최종 판단을 맡기기
    print("=== LLM 라우팅 ===")
    result = llm_router(text, llm, embedding_hints)
    
    return result

def main():
    print("=== 하이브리드 라우터 데모 ===")
    while True:
        user_input = input("\n사용자 입력 ('종료' 입력 시 종료): ")
        if user_input.strip() == "종료":
            print("종료합니다.")
            break
        selected_agent = hybrid_router(user_input, embedding_model, agent_descriptions, llm)
        print(f"선택된 에이전트: {selected_agent}")

if __name__ == "__main__":
    main()