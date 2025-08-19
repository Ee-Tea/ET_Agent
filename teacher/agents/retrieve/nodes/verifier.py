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

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
  raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 또는 환경변수에 키를 설정하세요.")
client = ChatOpenAI(
    model=OPENAI_LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=openai_api_key,
)

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

def fact_check(state: dict) -> dict:
    """
    LLM 응답을 사실 검증합니다.
    """
    question = state["retrieval_question"]
    answer = state["answer"]
    context = state["merged_context"]

    prompt = f"""
    다음 질문과 답변을 주어진 문맥을 바탕으로 사실 검증해주세요.

    질문: {question}
    답변: {answer}
    문맥: {context}

    다음 JSON 형식으로만 출력하세요:
    {{
        "verdict": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
        "confidence": 0.0-1.0,
        "reasoning": "판단 근거"
    }}
    """

    # LLM 호출
    response = client.invoke([HumanMessage(content=prompt)])
    raw_output = response.content
    
    # JSON 파싱
    validated = parse_llama_json(raw_output)
    
    if not validated:
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 0.0,
            "reasoning": "LLM 응답을 파싱할 수 없음"
        }
    
    return validated
