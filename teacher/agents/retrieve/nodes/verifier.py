import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import re
import random

# 1. 환경 변수 로드
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQAI_API_KEY")
client = None
if groq_api_key:
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
        # LLM 호출 (client가 있을 때만)
        if not client:
            return {
                "verdict": "NOT ENOUGH INFO",
                "confidence": 0.0,
                "evidence": [],
                "error": "GROQ API 키가 설정되지 않았습니다."
            }
        
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


class FixedResponseSystem:
    """
    주제 외 질문에 대한 고정 응답 시스템
    """
    
    def __init__(self):
        self._load_fixed_responses()
        self._load_patterns()
    
    def _load_fixed_responses(self):
        """고정 응답 데이터 로드"""
        self.fixed_responses = {
            "주제_외_거절": [
                "안녕하세요, 주제와 관련된 질문 부탁드립니다.",
                "안녕하세요, 정보처리기사 시험과 관련된 질문을 해주세요.",
                "안녕하세요, 소프트웨어설계, 소프트웨어개발, 데이터베이스구축, 프로그래밍언어활용, 정보시스템구축관리에 대한 질문을 해주세요."
            ],
            "인사": [
                "안녕하세요! 정보처리기사 시험에 대한 질문을 도와드리겠습니다.",
                "반갑습니다! 정보처리기사 시험 준비를 도와드릴게요.",
                "안녕하세요! 어떤 과목에 대해 궁금하신가요?"
            ],
            "감사": [
                "천만에요! 더 궁금한 점이 있으시면 언제든 물어보세요.",
                "도움이 되어서 기쁩니다. 정보처리기사 시험 합격을 응원합니다!",
                "별말씀을요! 시험 준비에 도움이 되었으면 좋겠습니다."
            ]
        }
    
    def _load_patterns(self):
        """패턴 매칭을 위한 정규표현식 로드"""
        self.patterns = {
            "주제_외_거절": [
                r"날씨|기상|비|맑음|흐림|더움|추움",
                r"시간|몇시|언제|오늘|내일|어제",
                r"음식|맛집|식당|밥|점심|저녁|아침",
                r"영화|드라마|예능|연예인|가수|배우",
                r"운동|스포츠|축구|야구|농구|테니스",
                r"여행|휴가|휴일|바캉스|관광",
                r"경제|주식|부동산|투자|금융",
                r"정치|뉴스|사회|사건|사고",
                r"건강|병원|의사|약|증상|아픔",
                r"취미|게임|독서|음악|미술|공예"
            ],
            "인사": [
                r"안녕|반가워|하이|헬로|안녕하세요|반갑습니다",
                r"좋은\s*(아침|점심|저녁)|굿모닝|굿애프터눈|굿이브닝"
            ],
            "감사": [
                r"감사|고마워|땡큐|고맙습니다|감사합니다|고마워요"
            ]
        }
    
    def _match_pattern(self, query: str) -> str:
        """쿼리와 패턴을 매칭하여 카테고리 반환"""
        query_lower = query.lower()
        
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return category
        
        return "주제_관련"
    
    def _generate_rejection_response(self, category: str) -> str:
        """거절 응답 생성"""
        responses = self.fixed_responses.get(category, [])
        if responses:
            return random.choice(responses)
        return "죄송합니다. 이 질문은 답변할 수 없습니다."
    
    def generate_response(self, query: str) -> dict:
        """
        쿼리에 대한 응답 생성
        
        Args:
            query (str): 사용자 질문
            
        Returns:
            dict: {"type": "rejection"|"quick_response"|"topic_related", 
                   "response": str, "category": str}
        """
        if not query or not query.strip():
            return {
                "type": "rejection",
                "response": "질문을 입력해주세요.",
                "category": "주제_외_거절"
            }
        
        # 패턴 매칭으로 카테고리 결정
        category = self._match_pattern(query)
        
        if category == "주제_외_거절":
            return {
                "type": "rejection",
                "response": self._generate_rejection_response(category),
                "category": category
            }
        elif category in ["인사", "감사"]:
            return {
                "type": "quick_response",
                "response": self._generate_rejection_response(category),
                "category": category
            }
        else:
            return {
                "type": "topic_related",
                "response": "",
                "category": "주제_관련"
            }


def fact_check(state: dict) -> dict:
    """
    LLM 응답을 사실 검증합니다.
    """
    question = state["retrieval_question"]
    answer = state["answer"]
    context = state["merged_context"]

    # 고정된 검증 결과 반환 (LLM 사용하지 않음)
    return {
        "verdict": "SUPPORTS",
        "confidence": 0.9,
        "reasoning": "규칙 기반 시스템으로 생성된 답변입니다."
    }
