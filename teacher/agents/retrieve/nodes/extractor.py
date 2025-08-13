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
  코드 블록(```json … ```) 감싸짐과 설명 텍스트를 제거합니다.
  """
  cleaned = result.strip()

  # 코드 블록 제거
  if cleaned.startswith("```"):
      cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
      cleaned = re.sub(r"```$", "", cleaned, flags=re.IGNORECASE).strip()

  # BOM 제거
  cleaned = cleaned.encode("utf-8").decode("utf-8-sig")

  # JSON 본문만 추출 (중괄호로 시작하고 끝나는 가장 큰 JSON)
  json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
  if not json_match:
      print("❌ JSON 블록을 찾을 수 없음")
      print("→ 원본:", repr(cleaned))
      return {}

  try:
      return json.loads(json_match.group())
  except json.JSONDecodeError as e:
      print("❌ JSON 파싱 실패")
      print("→ 오류:", e)
      print("→ 원본:", repr(json_match.group()))
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

def query_reinforce(state: dict) -> str:
    """
    검증에 실패한 경우 원 질문을 보완하여 재작성하는 함수.
    """
    original_question = state["retrieval_question"]
    verdict = state.get("fact_check_result", {}).get("verdict", "NOT ENOUGH INFO")
    answer = state.get("answer", "")
    context = state.get("merged_context", "")

    prompt = f"""
    이전 질문은 다음과 같습니다:

    "{original_question}"

    해당 질문에 대한 LLM 응답은 다음과 같았습니다:

    "{answer}"

    하지만 다음 문맥을 기반으로 검토한 결과, 이 응답은 "{verdict}" 판정을 받았습니다:

    "{context}"

    따라서, 이 질문을 더 명확하고 사실 검증이 가능한 형태로 재작성해주세요.
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  ***REMOVED*** 또는 원하는 모델
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    rewritten_question = response.choices[0].message.content.strip()
    print(f"🔁 재작성된 질문 (보강): {rewritten_question}")
    return rewritten_question