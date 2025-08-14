from openai import OpenAI
from typing import List
import os 
import json
import re
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQAI_API_KEY")
if not groq_api_key:
  raise ValueError("GROQ_API_KEY 환경변수가 설정되지 않았습니다. .env 또는 환경변수에 키를 설정하세요.")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)


def user_intent(user_question: str) -> dict:
    """
    사용자 질문에서 키워드를 추출합니다.
    """
    system_prompt = f"""다음 사용자 질문을 통해 사용자의 의도를 분석하세요:
    사용자 질문 : {user_question}
    질문의 의도 종류는 다음과 같습니다.
     - generate: 시험 문제 생성 - 원하는 자격증 및 시험에 대한 문제를 생성하는 것
     - retrieve: 정보 검색 - 모르는 단어 및 용어에 대한 정보를 검색하는 것
     - analyze: 오답 분석 - 틀린 문제를 정리하고 유형을 분석하여 보완점 및 전략 생성을 추천하는 것
     - solution: 문제 풀이 - 문제에 대한 답과 풀이, 해설을 제공하는 것
     - score: 채점 = 문제 풀이에 대한 채점 및 합격 여부를 판단 하는 것
     - unknown: 알 수 없는 의도
     
    위 5가지 의도 중 하나로만 분류하고, 그 외에 단어나 문장은 포함되지 않게 답변하세요.
    string 형식으로만 출력하세요. 예시:
    "retrieve"
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # 또는 gpt-3.5-turbo 사용 가능
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.2
    )
    result = response.choices[0].message.content
    
    return result.strip().lower()

def get_user_answer(user_question: str) -> str:
    """
    사용자 질문에서 답변을 추출합니다.
    """
    system_prompt = f"""다음 사용자 질문에서 답변을 추출하세요:
    사용자 질문 : {user_question}
    
    답변은 다음과 같은 형식으로 출력하세요. 모든 문제는 객관식이며, 사용자가 입력한 보기의 숫자만을 리스트(list[str]) 형태로 추출해야 합니다.
    "답변 내용"
    
    예시:
    사용자 입력: "정답은 4번, 3번, 1번, 2번, 4번, 3번, 3번, 2번, 1번, 1번, 4번입니다."
    추출할 내용: "[4, 3, 1, 2, 4, 3, 3, 2, 1, 1, 4]"
    추출할 수 있는 내용이 없으면 빈 리스트를 반환하세요. 예시: "[]"
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # 또는 gpt-3.5-turbo 사용 가능
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.2
    )
    
    result = response.choices[0].message.content
    
    return result.strip()