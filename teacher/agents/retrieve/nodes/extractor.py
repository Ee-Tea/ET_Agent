import os
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

openai_api_key = os.getenv("OPENAI_API_KEY=REDACTED(openai_api_key,OPENAI_LLM_MODEL,LLM_TEMPERATURE,LLM_MAX_TOKENS)
if not openai_api_key:
  raise ValueError("OPENAI_API_KEY=REDACTED 또는 환경변수에 키를 설정하세요.")

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

    llm = ChatOpenAI(
        api_key=openai_api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        model=OPENAI_LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )
    
    response = llm.invoke([
        HumanMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ])
    
    result = response.content
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
    
    llm = ChatOpenAI(
        api_key=openai_api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        model=OPENAI_LLM_MODEL,
        temperature=0.5,
        max_tokens=LLM_MAX_TOKENS
    )
    
    response = llm.invoke([HumanMessage(content=rewrite_prompt)])
    return response.content.strip()

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

    llm = ChatOpenAI(
        api_key=openai_api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        model=OPENAI_LLM_MODEL,
        temperature=0.3,
        max_tokens=LLM_MAX_TOKENS
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    rewritten_question = response.content.strip()
    print(f"🔁 재작성된 질문 (보강): {rewritten_question}")
    return rewritten_question