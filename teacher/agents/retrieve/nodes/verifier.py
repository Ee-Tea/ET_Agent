import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import re

# 1. 환경 변수 로드
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQAI_API_KEY")
if not groq_api_key:
  raise ValueError("GROQ_API_KEY 환경변수가 설정되지 않았습니다. .env 또는 환경변수에 키를 설정하세요.")
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

def fact_check_with_context(question:str, context: str, answer: str) -> dict:
    """
    LLM으로 답변 검증 → 출력만 Guardrails로 파싱

    Args:
        context (str): LLM이 참고한 컨텍스트
        answer (str): LLM이 생성한 응답

    Returns:
        dict: {"verdict": ..., "confidence": ..., "evidence": [...]}
    """
    prompt = f"""
    다음은 사용자의 질문에 대한 LLM 응답입니다. 
    이 응답이 주어진 문맥(Context)에 기반하여 사실인지 검토하고 그것이 사용자의 질문에 맞는 응답인지 판단하세요.
    
    # Question:
    {question}

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

    try:
        # LLM 호출
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        raw_output = response.choices[0].message.content.strip()

        # Guardrails로 출력 파싱 및 검증
        validated = parse_llama_json(raw_output)
        return validated

    except Exception as e:
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 0.0,
            "evidence": [],
            "error": str(e)
        }
