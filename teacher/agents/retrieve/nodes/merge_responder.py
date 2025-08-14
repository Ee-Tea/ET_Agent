from openai import OpenAI
from typing import List
from dotenv import load_dotenv
import os
load_dotenv()

# Grok이 OpenAI 호환 API를 제공한다고 가정
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQAI_API_KEY")
if not groq_api_key:
  raise ValueError("GROQ_API_KEY 환경변수가 설정되지 않았습니다. .env 또는 환경변수에 키를 설정하세요.")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)

def merge_context(wiki,ddg) -> str:
    """
    여러 개의 context chunk를 하나의 문자열로 병합합니다.
    """
    merge_prompt = f"""
    다음 두 개의 검색 결과를 읽고 중복되는 내용을 제거한 후, 간결하게 통합하여 설명해 주세요.

    [Wikipedia 결과]
    {wiki}

    [DuckDuckGo 결과]
    {ddg}

    ==> 중복을 제거한 요약 결과:
    """
    merged_result = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # 실제 사용 모델명 확인 필요
        messages=[{"role": "user", "content": merge_prompt}],
        temperature=0.5,
    )
    return merged_result.choices[0].message.content.strip()

def generate_answer(prompt: str, context: str) -> str:
    # context = "\n\n".join(context_chunks[:10])  # context 최대 5개만 사용
    full_prompt = f"""당신은 중학교 교사입니다. 다음의 데이터를 바탕으로 사용자의 질문에 답변해야 합니다.

    {context}

    위 참고자료를 바탕으로 사용자의 질문에 친절하고 이해하기 쉽게 답변해주세요.
    질문: {prompt}
    답변:"""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # 실제 사용 모델명 확인 필요
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()