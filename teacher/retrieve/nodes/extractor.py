from openai import OpenAI
from typing import List
import os 
import json
import re
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQAI_API_KEY")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)

def parse_llama_json(result: str) -> dict:
    """
    LLaMA 응답에서 JSON을 추출하여 dict로 반환합니다.
    코드 블록(```json … ```) 감싸짐을 제거합니다.
    """
    cleaned = result.strip()

    # 코드 블록 제거 (```json 또는 ```)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned, flags=re.IGNORECASE).strip()

    # BOM 제거
    cleaned = cleaned.encode("utf-8").decode("utf-8-sig")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("❌ JSON 파싱 실패")
        print("→ 오류:", e)
        print("→ 원본:", repr(cleaned))
        return {}

def extract_query_elements(user_question: str) -> dict:
    """
    사용자 질문에서 키워드를 추출합니다.
    """
    system_prompt = """다음 사용자 질문에서 아래 정보를 추출하세요:
    - keyword: 질문에 언급된 사용자가 정보를 필요로 하는 용어

    JSON 형식으로만 출력하세요. 예시:
    {
    "keyword": "LLM"
    }"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # 또는 gpt-3.5-turbo 사용 가능
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.2
    )
    result = response.choices[0].message.content
    result_dict = parse_llama_json(result)
    keyword = result_dict.get("keyword", "")
    if isinstance(keyword, str):
        return [keyword]
    elif isinstance(keyword, list):
        return keyword
    else:
        return []

def query_rewrite(question: str, keywords: list[str]) -> str:
    """
    질문을 LLM에 맞게 재작성합니다.
    """
    keyword_text = ", ".join(keywords)
    original_question = question
    rewrite_prompt = f"""
    사용자의 원래 질문과 추출된 키워드를 바탕으로, 검색에 적합한 형태의 명확한 질문으로 바꿔주세요.

    - 사용자 질문: {original_question}
    - 추출 키워드: {keyword_text}

    ==> 재작성된 검색 질문:
    """
    rewritten_question = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # 실제 사용 모델명 확인 필요
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0.5,
    )
    return rewritten_question.choices[0].message.content.strip()