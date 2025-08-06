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
    사용자 질문에서 작물명, 병해충명, 용도명을 추출합니다.
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
    # print(type(result))
    # print(result)

    print(result_dict)
    return result_dict

# result = extract_query_elements("어떤 살균제를 써야 하나요?")
# print(result)  