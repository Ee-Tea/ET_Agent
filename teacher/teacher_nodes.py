from openai import OpenAI
from typing import List
import os 
import json
import re
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
  raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 또는 환경변수에 키를 설정하세요.")

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"), api_key=openai_api_key)


def user_intent(user_question: str) -> dict:
    """
    사용자 질문에서 키워드를 추출합니다.
    """
    system_prompt = f"""다음 사용자 질문을 통해 사용자의 의도를 분석하세요:
    사용자 질문 : {user_question}
    질문의 의도 종류는 다음과 같습니다.
     - generate: 시험 문제 생성 - 새로운 문제를 만들거나 생성하는 것 (예: "문제 만들어줘", "5문제 출제해줘")
     - retrieve: 정보 검색 - 모르는 단어 및 용어에 대한 정보를 검색하는 것
     - analyze: 오답 분석 - 틀린 문제를 정리하고 유형을 분석하여 보완점 및 전략 생성을 추천하는 것
     - solution: 문제 풀이 - 기존 문제에 대한 답과 풀이, 해설을 제공하는 것 (예: "문제 풀어줘", "이거 해설 해줘", "PDF 풀이해줘")
     - score: 채점 - 문제 풀이에 대한 채점 및 합격 여부를 판단 하는 것
     
    중요한 구분:
    - PDF 파일이나 기존 문제를 풀어달라고 하면 "solution"
    - 새로운 문제를 만들어달라고 하면 "generate"
    - "풀어", "풀이", "해설" 등의 키워드가 있으면 대부분 "solution"
    - 사용자가 본인이 입력한 답에 대해 묻는 경우 "score"
     
    위 5가지 의도 중 하나로만 분류하고, 그 외에 단어나 문장은 포함되지 않게 답변하세요.
    string 형식으로만 출력하세요. 예시:
    "solution"
    """

    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=LLM_TEMPERATURE
    )
    result = response.choices[0].message.content
    
    return result.strip().lower()

def get_user_answer(user_question: str) -> list:
    """
    사용자 질문에서 답변을 추출합니다.
    """
    system_prompt = f"""다음 사용자 질문에서 답변을 추출하세요:
    사용자 질문 : {user_question}
    
    답변은 다음과 같은 형식으로 출력하세요. 모든 문제는 객관식이며, 사용자가 입력한 보기의 숫자만을 리스트(list[str]) 형태로 추출해야 합니다.
    "[4, 3, 1, 2, 4, 3, 3, 2, 1, 1, 4]"
    
    예시:
    사용자 입력: "정답은 4번, 3번, 1번, 2번, 4번, 3번, 3번, 2번, 1번, 1번, 4번입니다."
    추출할 내용: "[4, 3, 1, 2, 4, 3, 3, 2, 1, 1, 4]"
    추출할 수 있는 내용이 없으면 빈 리스트를 반환하세요. 예시: "[]"
    """

    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=LLM_TEMPERATURE
    )
    
    result = response.choices[0].message.content.strip()
    
    # LLM 응답을 파싱하여 리스트로 변환
    try:
        # JSON 형태의 리스트 문자열을 파싱
        import json
        if result.startswith('[') and result.endswith(']'):
            parsed_list = json.loads(result)
            if isinstance(parsed_list, list):
                # ★ 항상 문자열로 정규화해서 반환
                return [str(x).strip() for x in parsed_list]
        
        # JSON 파싱이 실패한 경우, 정규표현식으로 숫자 추출
        import re
        numbers = re.findall(r'\d+', result)
        if numbers:
            # ★ 여기서도 문자열 보장
            return [n.strip() for n in numbers]
        
        # 숫자도 없는 경우 빈 리스트 반환
        return []
        
    except (json.JSONDecodeError, ValueError):
        # JSON 파싱 실패 시 정규표현식으로 숫자 추출
        import re
        numbers = re.findall(r'\d+', result)
        return numbers if numbers else []

def parse_generator_input(user_question: str) -> dict:
    """
    사용자 질문에서 과목/문항수/난이도 파싱
    """
    system_prompt = f"""다음 사용자 질문에서 과목/문항수/난이도를 파싱하세요:
    지정된 형식의 응답 외에 문장이나 단어, 특수문자는 작성하지 마세요.
    사용자 질문 : {user_question}
    
    지원하는 모드:
    1. subject_quiz: 단일 과목 문제 생성
    2. partial_exam: 선택된 과목들에 대해 지정된 문제 수만큼 생성
    3. full_exam: 5과목 전체 문제 생성 (각 20문제씩)
    
    과목은 5가지 중 선택: (과목은 무조건 아래 5가지 중 선택을 해야 합니다. 또한, '과목 당', '전과목'을 요청할 경우 모든 과목을 포함해야 합니다.)
    - 소프트웨어설계, 소프트웨어개발, 데이터베이스구축, 프로그래밍언어활용, 정보시스템구축관리
    
    문항 수는 과목별 최대 40문제까지 가능
    난이도는 초급, 중급, 고급 중 하나, 언급 없으면 중급으로 간주
    과목 선정 없이 문제 수 만으로 문제 생성 요청 시 전체 과목 생성. 이 때, 과목 당 문제 수는 전체 문제 수의 1/5로 간주
    다른 변수 없이 단순한 문제 생성 요청은 전체 시험 문제 생성으로 간주
    
    파싱 결과는 다음과 같은 형식으로 출력하세요:
    
    단일 과목의 경우:
    {{
        "mode": "subject_quiz",
        "subject_area": "과목명",
        "target_count": "문항 수",
        "difficulty": "난이도"
    }}
    
    여러 과목 선택의 경우:
    {{
        "mode": "partial_exam",
        "selected_subjects": ["과목1", "과목2", "과목3"],
        "questions_per_subject": "과목당 문항 수",
        "difficulty": "난이도"
    }}
    
    전체 과목의 경우:
    {{
        "mode": "full_exam",
        "difficulty": "난이도"
    }}
    """

    try:
        response = client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=LLM_TEMPERATURE
        )
        
        result = response.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        try:
            import json
            parsed = json.loads(result)
            return parsed
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 기본값 반환
            print(f"⚠️ JSON 파싱 실패: {result}")
            return {
                "mode": "full_exam",
                "difficulty": "중급"
            }
            
    except Exception as e:
        print(f"⚠️ parse_generator_input 오류: {e}")
        # 오류 시 기본값 반환
        return {
            "mode": "full_exam",
            "difficulty": "중급"
        }