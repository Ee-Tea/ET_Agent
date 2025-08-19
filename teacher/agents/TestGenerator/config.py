import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Groq 관련 상수
DEFAULT_MODEL = "moonshotai/kimi-k2-instruct"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_TOKENS = 2048
MAX_GENERATION_ATTEMPTS = 15
MAX_ROUNDS = 10

# 경로 상수
DEFAULT_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), "test")
DEFAULT_WEAKNESS_DIR = os.path.join(os.path.dirname(__file__), "weakness")

# 정보처리기사 과목 정의
SUBJECT_AREAS = {
    "소프트웨어설계": {
        "count": 5,
        "keywords": ["요구사항", "UI 설계", "애플리케이션 설계", "인터페이스", "UML", "객체지향", "디자인패턴", "모듈화", "결합도", "응집도"]
    },
    "소프트웨어개발": {
        "count": 5,
        "keywords": ["자료구조", "스택", "큐", "리스트", "통합구현", "모듈", "패키징", "테스트케이스", "알고리즘", "인터페이스"]
    },
    "데이터베이스구축": {
        "count": 5,
        "keywords": ["SQL", "트리거", "DML", "DDL", "DCL", "정규화", "관계형모델", "E-R모델", "데이터모델링", "무결성"]
    },
    "프로그래밍언어활용": {
        "count": 5,
        "keywords": ["개발환경", "프로그래밍언어", "라이브러리", "운영체제", "네트워크", "데이터타입", "변수", "연산자"]
    },
    "정보시스템구축관리": {
        "count": 5,
        "keywords": ["소프트웨어개발방법론", "프로젝트관리", "보안", "시스템보안", "네트워크보안", "테일러링", "생명주기모델"]
    }
}

# 프롬프트 템플릿 상수
WEAKNESS_ANALYSIS_PROMPT = """당신은 정보처리기사 시험 전문가입니다. 아래 학습자 분석 결과를 바탕으로 취약점을 분석하고 맞춤형 학습이 필요한 핵심 개념들을 추출해주세요.

[학습자 분석 데이터]
{analysis_text}

다음 항목들을 분석해서 JSON 형식으로 출력하세요:

1. weakness_concepts: 학습자가 취약한 핵심 개념들 (구체적인 기술 용어나 개념명으로, 5-10개)
2. subject_focus: 집중해야 할 과목 영역들
3. difficulty_level: 추천 난이도 ("초급", "중급", "고급")
4. question_types: 필요한 문제 유형들
5. learning_priorities: 우선적으로 학습해야 할 순서

정보처리기사 출제 기준에 맞는 구체적이고 실용적인 개념들을 추출하되, 다음과 같은 영역에서 선별하세요:
- 소프트웨어 설계: 요구사항 분석, UML, 디자인패턴, 자료흐름도 등
- 소프트웨어 개발: 자료구조, 알고리즘, 프로그래밍 등  
- 데이터베이스: SQL, 정규화, 트랜잭션 등
- 프로그래밍언어: 언어별 특성, 라이브러리 등
- 정보시스템: 보안, 네트워크, 프로젝트관리 등

출력 예시:
{{
  "weakness_concepts": ["자료흐름도", "미들웨어", "SQL 조인", "정규화", "UML 다이어그램"],
  "subject_focus": ["소프트웨어설계", "데이터베이스구축"],
  "difficulty_level": "중급",
  "question_types": ["개념이해", "응용문제"],
  "learning_priorities": ["자료흐름도 구성요소 이해", "미들웨어 역할과 기능", "SQL 조인 유형별 특징"]
}}

JSON만 출력하세요:"""
