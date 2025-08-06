from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv
from guardrails import Guard

guard = Guard.from_rail("fact_check.rail")

load_dotenv()
groq_api_key = os.getenv("GROQAI_API_KEY")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)

def fact_check_with_context(context: str, answer: str) -> dict:
    prompt = f"""
    다음은 사용자의 질문에 대한 LLM 응답입니다. 이 응답이 주어진 문맥(Context)에 기반하여 사실인지 검토하세요.

    # Context:
    {context}

    # Answer:
    {answer}

    아래 형식으로 JSON으로 출력하세요:
    {{
    "verdict": "SUPPORTED" | "REFUTED" | "NOT ENOUGH INFO",
    "confidence": 0~1 사이의 점수 (float),
    "evidence": ["해당 판단을 뒷받침하는 문장들"]
    }}
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # 실제 사용 모델명 확인 필요
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    
    val_response = guard.parse(response.choices[0].message.content.strip())
    return val_response.output