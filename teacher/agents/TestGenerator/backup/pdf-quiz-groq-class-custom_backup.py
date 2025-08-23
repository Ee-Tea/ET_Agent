import os
import json
import re
from typing import List, Dict, Any, TypedDict
from abc import ABC, abstractmethod
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langgraph.graph import StateGraph, END
import time
from datetime import datetime

# .env 파일 로드를 위한 임포트
from dotenv import load_dotenv

***REMOVED*** 관련 임포트
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # openai 패키지가 없으면 None 처리

# 상수 정의
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

def extract_quiz_params(
    user_question: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    groq_api_key: str = None,
    base_url: str = DEFAULT_BASE_URL
) -> dict:
    """사용자 질문에서 save_to_file, filename, difficulty, mode를 LLM을 통해 추출합니다."""
    if OpenAI is None:
        raise ImportError("openai 패키지가 설치되어 있지 않습니다. 'pip install openai'로 설치하세요.")
    
    load_dotenv()
    api_key = groq_api_key or os.getenv("GROQAI_API_KEY") or os.getenv("GROQ_API_KEY=REDACTED not api_key:
        raise ValueError("GROQAI_API_KEY 환경변수 또는 groq_api_key 인자가 필요합니다.")
    
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    prompt = f"""다음 사용자 질문에서 아래 4가지 정보를 추출하여 JSON 형식으로 출력하세요.

- save_to_file: 문제 생성 결과를 파일로 저장할지 여부 (True/False)
- filename: 저장할 파일명 (사용자가 명시하지 않으면 null)
- difficulty: 난이도 ("초급", "중급", "고급" 중 하나, 명시 없으면 "중급")
- mode: "full_exam" 또는 "subject_quiz" 또는 "weakness_quiz" 중 하나

예시: {{"save_to_file": true, "filename": "내문제.json", "difficulty": "고급", "mode": "weakness_quiz"}}

사용자 질문: {user_question}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=temperature
        )
        content = response.choices[0].message.content
        match = re.search(r'\{[\s\S]*\}', content)
        return json.loads(match.group(0)) if match else {}
    except Exception:
        return {}

# .env 파일 로드
load_dotenv()

class BaseAgent(ABC):
    """모든 에이전트가 상속받아야 하는 기본 추상 클래스입니다."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """에이전트의 고유한 이름을 반환합니다."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """에이전트의 역할에 대한 설명을 반환합니다."""
        pass
    
    @abstractmethod
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트의 주된 로직을 실행하는 메서드입니다."""
        pass


class GraphState(TypedDict):
    """그래프 상태 정의"""
    query: str
    documents: List[Document]
    context: str
    quiz_questions: List[Dict[str, Any]]
    quiz_count: int
    difficulty: str
    error: str
    used_sources: List[str]
    generation_attempts: int
    target_quiz_count: int
    subject_area: str
    validated_questions: List[Dict[str, Any]]
    weakness_analysis: Dict[str, Any]  # 취약점 분석 결과
    weakness_concepts: List[str]  # 추출된 취약점 개념들


class InfoProcessingExamAgent(BaseAgent):
    """
    정보처리기사 출제기준에 맞는 25문제 자동 출제 에이전트 (순차 처리 버전)
    LLM 기반 취약점 분석 및 맞춤형 문제 생성
    """
    
    # 전역 상수 사용
    
    def __init__(self, data_folder=DEFAULT_DATA_FOLDER, groq_api_key=None):
        """초기화"""
        self.data_folder = data_folder
        os.makedirs(self.data_folder, exist_ok=True)
        
        ***REMOVED*** API 키 설정
        if groq_api_key:
            os.environ["GROQ_API_KEY=REDACTED
        elif not os.getenv("GROQ_API_KEY=REDACTED ValueError("Groq API 키가 필요합니다.")
        
        # RAG 엔진 초기화
        from rag_engine import RAGEngine
        self.rag_engine = RAGEngine(data_folder=data_folder)
        
        # RAG 엔진 벡터 스토어 초기화
        print("🔨 RAG 엔진 벡터 스토어 초기화 중...")
        if not self.rag_engine.build_vectorstore_from_all_pdfs():
            raise ValueError(f"'{data_folder}' 폴더에서 PDF 파일을 로드할 수 없습니다.")
        print("✅ RAG 엔진 벡터 스토어 초기화 완료")
        
        self.llm = None
        self.workflow = None
        
        self._initialize_models()
        self._build_graph()
            
    @property
    def name(self) -> str:
        """에이전트의 고유한 이름을 반환합니다."""
        return "InfoProcessingExamAgent"

    @property
    def description(self) -> str:
        """에이전트의 역할에 대한 설명을 반환합니다."""
        return "정보처리기사 출제기준에 맞는 25문제를 자동으로 생성하는 에이전트입니다. PDF 문서를 기반으로 5개 과목별로 문제를 생성하며, LLM을 활용하여 학습자의 취약점을 자동 분석하고 맞춤형 문제를 생성합니다."

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트의 주된 로직을 실행하는 메서드입니다.
        
        Args:
            input_data (Dict[str, Any]): 에이전트 실행에 필요한 입력 데이터
                - mode: "full_exam", "subject_quiz", "weakness_quiz"
                - difficulty: "초급", "중급", "고급" (기본값: "중급")
                - subject_area: 특정 과목명 (subject_quiz 모드일 때)
                - target_count: 생성할 문제 수 (subject_quiz, weakness_quiz 모드일 때)
                - save_to_file: JSON 파일 저장 여부 (기본값: False)
                - filename: 저장할 파일명 (선택사항)
                - analysis_file_path: 분석 결과 JSON 파일 경로 (weakness_quiz 모드일 때)
                - raw_analysis_text: 분석 텍스트 (weakness_quiz 모드일 때)
                
        Returns:
            Dict[str, Any]: 에이전트 실행 결과 데이터
                - success: 성공 여부
                - result: 생성된 시험 데이터
                - error: 오류 메시지 (실패시)
                - file_path: 저장된 파일 경로 (저장시)
        """
        try:
            # 입력 데이터 검증 및 기본값 설정
            mode = input_data.get("mode", "full_exam")
            difficulty = input_data.get("difficulty", "중급")
            save_to_file = input_data.get("save_to_file", False)
            filename = input_data.get("filename")
            
            # RAG 엔진 상태 확인
            vectorstore_info = self.rag_engine.get_vectorstore_info()
            if not vectorstore_info.get("is_initialized", False):
                return {
                    "success": False,
                    "error": f"RAG 엔진이 초기화되지 않았습니다."
                }
            
            if mode == "full_exam":
                # 전체 25문제 생성
                result = self._generate_full_exam(difficulty)
            elif mode == "subject_quiz":
                # 특정 과목 문제 생성
                subject_area = input_data.get("subject_area")
                target_count = input_data.get("target_count", 5)
                
                if not subject_area or subject_area not in SUBJECT_AREAS:
                    return {
                        "success": False,
                        "error": f"유효하지 않은 과목명입니다. 가능한 과목: {list(SUBJECT_AREAS.keys())}"
                    }
                
                result = self._generate_subject_quiz(subject_area, target_count, difficulty)
            elif mode == "weakness_quiz":
                # 취약점 기반 문제 생성
                result = self._generate_weakness_quiz(input_data, difficulty)
            else:
                return {
                    "success": False,
                    "error": "유효하지 않은 모드입니다. 'full_exam', 'subject_quiz', 'weakness_quiz' 중 하나를 사용하세요."
                }
            
            # 오류 확인
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"]
                }
            
            response = {
                "success": True,
                "result": result
            }
            
            # 파일 저장 요청 시
            if save_to_file:
                try:
                    file_path = self._save_to_json(result, filename)
                    response["file_path"] = file_path
                except Exception as e:
                    response["save_error"] = str(e)
            
            return response
            
        except Exception as e:
            return {
                "success": False,
                "error": f"에이전트 실행 중 오류 발생: {str(e)}"
            }

    def analyze_weakness_with_llm(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM을 사용하여 분석 데이터에서 취약점을 분석하고 학습 개념을 추출합니다.
        
        Args:
            analysis_data: 분석 결과 데이터
            
        Returns:
            취약점 분석 결과
        """
        try:
            # 분석 데이터를 텍스트로 변환
            analysis_text = self._convert_analysis_to_text(analysis_data)
            
            analysis_prompt = PromptTemplate(
                input_variables=["analysis_text"],
                template=WEAKNESS_ANALYSIS_PROMPT
            )
            
            prompt = analysis_prompt.format(analysis_text=analysis_text)
            
            response = self.llm.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # JSON 추출
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                weakness_analysis = json.loads(match.group(0))
                return weakness_analysis
            else:
                return {"error": "LLM 분석 결과를 파싱할 수 없습니다."}
                
        except Exception as e:
            return {"error": f"취약점 LLM 분석 중 오류: {str(e)}"}

    def _convert_analysis_to_text(self, analysis_data: Dict[str, Any]) -> str:
        """분석 데이터를 텍스트로 변환"""
        try:
            text_parts = []
            
            # overall_assessment에서 정보 추출
            overall = analysis_data.get("analysis", {}).get("overall_assessment", {})
            if overall.get("weaknesses"):
                text_parts.append(f"취약점: {overall['weaknesses']}")
            if overall.get("strengths"):
                text_parts.append(f"강점: {overall['strengths']}")
            
            # detailed_analysis에서 정보 추출
            detailed = analysis_data.get("analysis", {}).get("detailed_analysis", [])
            for item in detailed:
                concept_path = item.get("concept_path", "")
                mistake_type = item.get("mistake_type", "")
                analysis = item.get("analysis", "")
                
                detail_text = f"개념경로: {concept_path}, 실수유형: {mistake_type}, 분석: {analysis}"
                text_parts.append(detail_text)
            
            return "\n\n".join(text_parts)
            
        except Exception:
            # 원본 데이터를 문자열로 변환
            return str(analysis_data)

    def _extract_weakness_concepts_from_analysis(self, analysis_data: Dict[str, Any]) -> List[str]:
        """
        분석 결과 JSON에서 취약점 개념들을 추출합니다.
        
        Args:
            analysis_data: 분석 결과 딕셔너리
            
        Returns:
            취약점 개념 리스트
        """
        try:
            weakness_concepts = []
            
            # detailed_analysis에서 개념 추출
            detailed_analysis = analysis_data.get("analysis", {}).get("detailed_analysis", [])
            for item in detailed_analysis:
                analysis_text = item.get("analysis", "")
                if analysis_text:
                    # 분석 텍스트에서 핵심 개념 추출
                    # 예: "자료 흐름도(DFD)의 구성 요소", "미들웨어(Middleware)의 정의" 등
                    concepts = self._extract_concepts_from_text(analysis_text)
                    weakness_concepts.extend(concepts)
            
            # 중복 제거 및 정리
            unique_concepts = list(set(weakness_concepts))
            
            # 빈 문자열이나 너무 짧은 개념 제거
            filtered_concepts = [concept for concept in unique_concepts if concept and len(concept.strip()) > 1]
            
            # 의미 있는 기술 개념만 필터링
            meaningful_concepts = []
            for concept in filtered_concepts:
                # 기술 용어 사전에 있는 개념들 우선 선택
                if any(tech_term.lower() in concept.lower() for tech_term in [
                    "자료 흐름도", "dfd", "미들웨어", "middleware", "프로세스", "process",
                    "자료 저장소", "data store", "종단점", "terminator", "sql", "정규화",
                    "uml", "다이어그램", "트랜잭션", "보안", "네트워크", "알고리즘",
                    "요구사항", "설계", "개발", "테스트", "구현", "모듈", "인터페이스",
                    "객체지향", "패턴", "데이터베이스", "관계형", "정규화", "무결성"
                ]):
                    meaningful_concepts.append(concept)
                # 3글자 이상의 한글 개념 중 의미 있는 것들
                elif len(concept) >= 3 and all(ord(c) > 127 for c in concept):
                    # 일반적인 조사나 부사 제외
                    if concept not in ["대한", "설명을", "책임자에", "유형에", "특징을", "학생은", "이해하지", "오해했습니다", "원칙과", "충분히", "통한", "위한", "있는", "하는", "되는", "있는", "하는", "되는"]:
                        meaningful_concepts.append(concept)
            
            # 의미 있는 개념이 없으면 기본 기술 개념 반환
            if not meaningful_concepts:
                meaningful_concepts = [
                    "자료 흐름도", "미들웨어", "프로세스", "자료 저장소", "SQL", "정규화",
                    "UML", "다이어그램", "트랜잭션", "보안", "네트워크", "알고리즘"
                ]
            
            return meaningful_concepts[:10]  # 최대 10개까지 반환
            
        except Exception as e:
            print(f"취약점 개념 추출 중 오류: {e}")
            return []

    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """
        텍스트에서 핵심 개념을 추출합니다.
        
        Args:
            text: 분석 텍스트
            
        Returns:
            추출된 개념 리스트
        """
        try:
            concepts = []
            
            # 괄호 안의 영문 용어 추출 (예: "자료 흐름도(DFD)", "미들웨어(Middleware)")
            import re
            bracket_pattern = r'([가-힣\s]+)\(([A-Za-z\s]+)\)'
            bracket_matches = re.findall(bracket_pattern, text)
            for korean, english in bracket_matches:
                concepts.append(korean.strip())
                concepts.append(english.strip())
            
            # 일반적인 기술 용어들 추출
            tech_terms = [
                "자료 흐름도", "DFD", "미들웨어", "Middleware", "프로세스", "Process",
                "자료 저장소", "Data Store", "종단점", "Terminator", "SQL", "정규화",
                "UML", "다이어그램", "트랜잭션", "보안", "네트워크", "알고리즘",
                "요구사항", "설계", "개발", "테스트", "구현", "모듈", "인터페이스",
                "객체지향", "패턴", "데이터베이스", "관계형", "무결성", "인덱스",
                "뷰", "트리거", "저장프로시저", "트랜잭션", "동시성", "데드락"
            ]
            
            for term in tech_terms:
                if term.lower() in text.lower():
                    concepts.append(term)
            
            # 의미 있는 한글 개념 추출 (3글자 이상, 조사/부사 제외)
            meaningful_korean_pattern = r'[가-힣]{3,}'
            korean_matches = re.findall(meaningful_korean_pattern, text)
            
            # 의미 없는 조사/부사 목록
            meaningless_words = {
                "대한", "설명을", "책임자에", "유형에", "특징을", "학생은", "이해하지", 
                "오해했습니다", "원칙과", "충분히", "통한", "위한", "있는", "하는", 
                "되는", "있습니다", "합니다", "됩니다", "입니다", "것입니다", "것을",
                "것이", "것은", "것에", "것으로", "것을", "것이", "것은", "것에"
            }
            
            for match in korean_matches:
                if match not in concepts and match not in meaningless_words:
                    # 기술적 맥락에서 의미 있는 단어인지 확인
                    if any(tech_word in match for tech_word in ["설계", "개발", "테스트", "구현", "분석", "관리", "보안", "네트워크", "데이터", "시스템", "프로그램", "소프트웨어", "하드웨어"]):
                        concepts.append(match)
            
            return concepts
            
        except Exception as e:
            print(f"개념 추출 중 오류: {e}")
            return []

    def load_analysis_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        분석 결과 JSON 파일을 로드합니다.
        
        Args:
            file_path: 분석 결과 JSON 파일 경로
            
        Returns:
            분석 결과 딕셔너리
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"분석 파일 로드 실패: {e}")

    def _generate_weakness_quiz(self, input_data: Dict[str, Any], difficulty: str) -> Dict[str, Any]:
        """
        LLM을 활용하여 취약점을 분석하고 맞춤형 문제를 생성합니다.
        
        Args:
            input_data: 입력 데이터
            difficulty: 난이도
            
        Returns:
            생성 결과
        """
        try:
            target_count = input_data.get("target_count", 10)
            
            # 분석 데이터 로드
            analysis_data = None
            if "analysis_file_path" in input_data:
                analysis_data = self.load_analysis_from_file(input_data["analysis_file_path"])
            elif "raw_analysis_text" in input_data:
                # 텍스트를 구조화된 데이터로 변환
                analysis_data = {"analysis_text": input_data["raw_analysis_text"]}
            else:
                return {"error": "분석 파일 경로(analysis_file_path) 또는 분석 텍스트(raw_analysis_text)가 필요합니다."}
            
            # JSON 파일에서 직접 취약점 개념 추출
            weakness_concepts = self._extract_weakness_concepts_from_analysis(analysis_data)
            
            # LLM을 사용하여 취약점 분석 (백업 방법)
            weakness_analysis = None
            if not weakness_concepts:
                weakness_analysis = self.analyze_weakness_with_llm(analysis_data)
                if "error" in weakness_analysis:
                    return {"error": weakness_analysis["error"]}
                weakness_concepts = weakness_analysis.get("weakness_concepts", [])
            
            # 의미 있는 개념이 없으면 기본 기술 개념 사용
            if not weakness_concepts:
                weakness_concepts = [
                    "자료 흐름도", "미들웨어", "프로세스", "자료 저장소", "SQL", "정규화",
                    "UML", "다이어그램", "트랜잭션", "보안", "네트워크", "알고리즘"
                ]
                print(f"⚠️  추출된 개념이 없어 기본 기술 개념을 사용합니다: {weakness_concepts}")
            
            subject_focus = weakness_analysis.get("subject_focus", []) if weakness_analysis else []
            
            if not weakness_concepts:
                return {"error": "취약점 개념을 추출할 수 없습니다."}
            
            print(f"🧠 추출된 취약점 개념: {weakness_concepts}")
            print(f"📚 집중 과목: {subject_focus}")
            print(f"🎯 개념별 문제 생성 시작...")
            
            # 취약점 개념을 활용한 문제 생성
            all_questions = []
            
            # 취약점 개념별로 문제 생성
            questions_per_concept = max(1, target_count // len(weakness_concepts))
            remaining_questions = target_count
            
            for i, concept in enumerate(weakness_concepts):
                if remaining_questions <= 0:
                    break
                
                # 마지막 개념에서는 남은 문제 수만큼 생성
                current_target = questions_per_concept
                if i == len(weakness_concepts) - 1:
                    current_target = remaining_questions
                else:
                    current_target = min(questions_per_concept, remaining_questions)
                
                print(f"  📝 '{concept}' 개념으로 {current_target}개 문제 생성 중...")
                
                # 개념 기반 문제 생성
                result = self._generate_weakness_focused_questions(
                    weakness_concept=concept,
                    target_count=current_target,
                    difficulty=difficulty,
                    subject_areas=subject_focus
                )
                
                if "questions" in result and result["questions"]:
                    # 중복 문제 제거
                    existing_questions = [q.get('question', '') for q in all_questions]
                    new_questions = []
                    
                    for q in result["questions"]:
                        if q.get('question', '') not in existing_questions:
                            q["weakness_concept"] = concept  # 취약점 개념 태깅
                            new_questions.append(q)
                            existing_questions.append(q.get('question', ''))
                    
                    all_questions.extend(new_questions)
                    remaining_questions -= len(new_questions)
                    print(f"    ✅ '{concept}' 개념으로 {len(new_questions)}개 문제 생성 완료")
                else:
                    print(f"    ❌ '{concept}' 개념으로 문제 생성 실패: {result.get('error', '알 수 없는 오류')}")
            
            final_questions = all_questions[:target_count]
            
            return {
                "quiz_type": "weakness_based_llm",
                "difficulty": difficulty,
                "weakness_analysis": weakness_analysis if "weakness_analysis" in locals() else {"extracted_concepts": weakness_concepts},
                "weakness_concepts": weakness_concepts,
                "requested_count": target_count,
                "quiz_count": len(final_questions),
                "questions": final_questions,
                "generation_summary": {
                    "analyzed_concepts": len(weakness_concepts),
                    "generated_questions": len(final_questions),
                    "success_rate": f"{len(final_questions)/target_count*100:.1f}%",
                    "focus_subjects": subject_focus
                },
                "status": "SUCCESS" if len(final_questions) >= target_count else "PARTIAL"
            }
            
        except Exception as e:
            return {"error": f"취약점 기반 문제 생성 중 오류: {str(e)}"}

    def _generate_weakness_focused_questions(self, weakness_concept: str, target_count: int, difficulty: str, subject_areas: List[str] = None) -> Dict[str, Any]:
        """
        특정 취약점 개념에 집중된 문제를 생성합니다.
        
        Args:
            weakness_concept: 취약점 개념
            target_count: 생성할 문제 수
            difficulty: 난이도
            subject_areas: 집중할 과목 영역들
            
        Returns:
            생성 결과
        """
        try:
            # 개념 기반으로 관련 문서 검색
            search_query = weakness_concept
            if subject_areas:
                search_query += f" {' '.join(subject_areas)}"
            
            # 기본 과목 영역 설정
            default_subject = "종합"
            if subject_areas and len(subject_areas) > 0:
                default_subject = subject_areas[0]
            elif weakness_concept in ["요구사항", "UI 설계", "애플리케이션 설계", "인터페이스", "UML", "객체지향", "디자인패턴", "모듈화", "결합도", "응집도"]:
                default_subject = "소프트웨어설계"
            elif weakness_concept in ["SQL", "트리거", "DML", "DDL", "DCL", "정규화", "관계형모델", "E-R모델", "데이터모델링", "무결성"]:
                default_subject = "데이터베이스구축"
            elif weakness_concept in ["소프트웨어개발방법론", "프로젝트관리", "보안", "시스템보안", "네트워크보안", "테일러링", "생명주기모델"]:
                default_subject = "정보시스템구축관리"
            elif weakness_concept in ["개발환경", "프로그래밍언어", "라이브러리", "운영체제", "네트워크", "데이터타입", "변수", "연산자"]:
                default_subject = "프로그래밍언어활용"
            elif weakness_concept in ["자료구조", "스택", "큐", "리스트", "통합구현", "모듈", "패키징", "테스트케이스", "알고리즘", "인터페이스"]:
                default_subject = "소프트웨어개발"
            
            initial_state = {
                "query": search_query,
                "quiz_count": target_count,
                "target_quiz_count": target_count,
                "difficulty": difficulty,
                "generation_attempts": 0,
                "quiz_questions": [],
                "validated_questions": [],
                "subject_area": default_subject,
                "weakness_concepts": [weakness_concept]
            }
            
            result = self.workflow.invoke(initial_state)
            
            if result.get("error"):
                return {"error": result["error"]}
            
            return {
                "questions": result.get("validated_questions", []),
                "used_sources": result.get("used_sources", [])
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _initialize_models(self):
        """LLM 모델 초기화"""
        try:
            self.llm = ChatGroq(
                model=DEFAULT_MODEL,
                temperature=0.0,
                max_tokens=DEFAULT_MAX_TOKENS,
                timeout=DEFAULT_TIMEOUT,
                max_retries=DEFAULT_MAX_RETRIES
            )
            
            # 연결 테스트
            self.llm.invoke("안녕하세요")
            
        except Exception as e:
            raise ValueError(f"LLM 모델 초기화 중 오류 발생: {e}")

    def _retrieve_documents(self, state: GraphState) -> GraphState:
        """문서 검색 및 출처 분석 노드"""
        try:
            query = state["query"]
            subject_area = state.get("subject_area", "")
            weakness_concepts = state.get("weakness_concepts", [])
            
            print(f"🔍 문서 검색 시작: query='{query}', subject_area='{subject_area}', weakness_concepts={weakness_concepts}")
            
            # RAG 엔진을 사용하여 문서 검색
            result = self.rag_engine.retrieve_documents(
                query=query,
                subject_area=subject_area,
                weakness_concepts=weakness_concepts
            )
            
            print(f"📋 RAG 엔진 결과: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            if isinstance(result, dict):
                print(f"   - documents 키 존재: {'documents' in result}")
                print(f"   - documents 타입: {type(result.get('documents'))}")
                if 'documents' in result:
                    print(f"   - documents 길이: {len(result['documents']) if result['documents'] else 0}")
            
            if "error" in result:
                print(f"❌ RAG 엔진 오류: {result['error']}")
                return {**state, "error": result["error"]}
            
            if "documents" not in result:
                print(f"❌ 'documents' 키가 결과에 없습니다!")
                return {**state, "error": "RAG 엔진에서 'documents' 키를 찾을 수 없습니다."}
            
            print(f"✅ 문서 검색 성공: {len(result['documents'])}개 문서, {len(result.get('used_sources', []))}개 소스")
            
            return {
                **state, 
                "documents": result["documents"], 
                "used_sources": result["used_sources"]
            }
        except Exception as e:
            print(f"❌ 문서 검색 중 예외 발생: {e}")
            return {**state, "error": f"문서 검색 오류: {e}"}

    def _prepare_context(self, state: GraphState) -> GraphState:
        """컨텍스트 준비 노드"""
        documents = state["documents"]
        weakness_concepts = state.get("weakness_concepts", [])
        
        # RAG 엔진을 사용하여 컨텍스트 준비
        context = self.rag_engine.prepare_context(
            documents=documents,
            weakness_concepts=weakness_concepts
        )
        
        return {**state, "context": context}

    def _generate_quiz_incremental(self, state: GraphState) -> GraphState:
        """취약점 집중 문제 생성"""
        try:
            context = state["context"]
            target_quiz_count = state.get("target_quiz_count", 5)
            validated_questions = state.get("validated_questions", [])
            subject_area = state.get("subject_area", "")
            weakness_concepts = state.get("weakness_concepts", [])
            difficulty = state.get("difficulty", "중급")
            needed_count = target_quiz_count - len(validated_questions)
            
            if needed_count <= 0:
                return {**state, "quiz_questions": validated_questions}
            
            if not context.strip():
                return {**state, "error": "검색된 문서 내용이 없습니다."}
            
            generate_count = max(needed_count, 3)
            
            # 취약점 개념 집중 문제 생성 프롬프트
            if weakness_concepts:
                weakness_focus = f"""학습자가 특히 어려워하는 취약점 개념들: {', '.join(weakness_concepts)}
이 개념들에 대해 학습자의 이해도를 높일 수 있는 문제를 집중적으로 출제하세요.

취약점 개념별 문제 생성 지침:
- 개념의 정의나 특징을 묻는 기본 문제
- 개념을 실제 상황에 적용하는 응용 문제  
- 비슷한 개념들과의 차이점을 구분하는 문제
- 개념의 구성요소나 절차를 묻는 문제"""
                
                template_text = """당신은 정보처리기사 출제 전문가입니다. 아래 문서 내용을 바탕으로 학습자의 취약점을 보강할 수 있는 {subject_area} 관련 객관식 문제 {quiz_count}개를 생성하세요.

{weakness_focus}

난이도: {difficulty}

[문서 내용]
{context}

[출제 기준]
1. 정보처리기사 시험 출제 기준에 맞는 4지선다 객관식 문제
2. 취약점 개념에 대한 정확한 이해를 확인할 수 있는 문제
3. 실무에서 활용 가능한 실용적인 문제
4. 명확한 정답과 해설 포함

반드시 JSON 형식으로만 출력하세요:

{{
  "questions": [
    {{
      "question": "구체적이고 명확한 문제 내용",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "answer": "정답 번호(1, 2, 3, 4 중 하나)",
      "explanation": "정답에 대한 상세한 해설과 취약점 개념 설명",
      "weakness_focus": "집중한 취약점 개념명"
    }}
  ]
}}"""
            else:
                weakness_focus = ""
                template_text = """아래 문서 내용을 바탕으로 {subject_area} 과목의 객관식 문제 {quiz_count}개를 반드시 생성하세요.
각 문제는 4지선다, 정답 번호와 간단한 해설을 포함해야 하며, 반드시 JSON만 출력하세요.

[문서]
{context}

[출력 예시]
{{
  "questions": [
    {{
      "question": "문제 내용",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "answer": "정답 번호(예: 1)",
      "explanation": "간단한 해설"
    }}
  ]
}}"""
            
            prompt_template = PromptTemplate(
                input_variables=["context", "quiz_count", "subject_area", "weakness_focus", "difficulty"],
                template=template_text
            )
            
            prompt = prompt_template.format(
                context=context,
                quiz_count=generate_count,
                subject_area=subject_area,
                weakness_focus=weakness_focus,
                difficulty=difficulty
            )
            
            self.llm.temperature = 0.2
            self.llm.max_tokens = 1500
            response = self.llm.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            new_questions = self._parse_quiz_response(response_content, subject_area)
            if not new_questions:
                return {**state, "error": "유효한 문제를 생성하지 못했습니다."}
            
            return {
                **state,
                "quiz_questions": new_questions,
                "validated_questions": validated_questions,
                "generation_attempts": state.get("generation_attempts", 0) + 1
            }
        except Exception as e:
            return {**state, "error": f"문제 생성 중 오류 발생: {e}"}

    def _validate_quiz_incremental(self, state: GraphState) -> GraphState:
        """취약점 집중 문제 검증"""
        subject_area = state.get("subject_area", "")
        weakness_concepts = state.get("weakness_concepts", [])
        
        previously_validated = state.get("validated_questions", [])
        new_questions = state.get("quiz_questions", [])
        context = state.get("context", "")
        target_quiz_count = state.get("target_quiz_count", 5)
        generation_attempts = state.get("generation_attempts", 0)

        if not new_questions:
            return {**state, "validated_questions": previously_validated}

        newly_validated = []
        
        # 취약점 집중 검증 프롬프트
        validation_template = """당신은 정보처리기사 출제 전문가입니다. 아래 제공된 '문서 내용'에 근거하여 '퀴즈 문제'가 취약점 보강에 적합한지 평가해주세요.

[문서 내용]
{context}

[취약점 개념들]
{weakness_concepts}

[평가할 퀴즈 문제]
{question_data}

[평가 기준]
1. 문제가 문서 내용에서 직접적으로 확인 가능한가?
2. 취약점 개념에 대한 이해를 확인할 수 있는가?
3. 정보처리기사 시험 수준에 적합한 난이도인가?
4. 4개 선택지가 명확하고 정답이 유일한가?
5. 해설이 취약점 개념을 잘 설명하고 있는가?

평가 결과를 JSON 형식으로 출력하세요:
{{
  "is_valid": true/false,
  "reason": "평가 이유 설명",
  "weakness_relevance": true/false
}}"""
        
        validation_prompt_template = PromptTemplate(
            input_variables=["context", "question_data", "weakness_concepts"],
            template=validation_template
        )
        
        needed = target_quiz_count - len(previously_validated)
        weakness_concepts_str = ', '.join(weakness_concepts) if weakness_concepts else "일반"
        
        for i, q in enumerate(new_questions):
            if len(newly_validated) >= needed:
                break
                
            try:
                question_str = json.dumps(q, ensure_ascii=False, indent=2)
                prompt = validation_prompt_template.format(
                    context=context[:4000], 
                    question_data=question_str,
                    weakness_concepts=weakness_concepts_str
                )
                
                response = self.llm.invoke(prompt)
                response_str = response.content if hasattr(response, 'content') else str(response)
                
                match = re.search(r'\{.*\}', response_str, re.DOTALL)
                if not match:
                    continue

                validation_result = json.loads(match.group(0))

                # 유효성과 취약점 관련성 모두 확인
                if (validation_result.get("is_valid") is True and 
                    validation_result.get("weakness_relevance", True) is True):
                    newly_validated.append(q)

            except Exception:
                continue
        
        all_validated = previously_validated + newly_validated
        
        need_more_questions = (len(all_validated) < target_quiz_count and 
                              generation_attempts < MAX_GENERATION_ATTEMPTS)
        
        return {
            **state,
            "validated_questions": all_validated,
            "quiz_questions": all_validated,
            "need_more_questions": need_more_questions
        }

    def _check_completion(self, state: GraphState) -> str:
        """문제 생성 완료 여부를 체크하는 조건부 노드"""
        validated_count = len(state.get("validated_questions", []))
        target_count = state.get("target_quiz_count", 5)
        generation_attempts = state.get("generation_attempts", 0)
        
        if validated_count >= target_count:
            return "complete"
        elif generation_attempts < MAX_GENERATION_ATTEMPTS:
            return "generate_more"
        else:
            return "complete"

    def _parse_quiz_response(self, response: str, subject_area: str = "") -> List[Dict[str, Any]]:
        """응답에서 JSON 형식의 문제를 파싱"""
        try:
            json_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1)
            else:
                json_str_match = re.search(r'\{.*\}', response.strip(), re.DOTALL)
                if not json_str_match:
                    return []
                json_str = json_str_match.group(0)

            json_str = json_str.replace('\\u312f', '').replace('\\n', ' ')
            data = json.loads(json_str)
            
            if "questions" not in data or not isinstance(data["questions"], list):
                return []
            
            for question in data["questions"]:
                if "options" in question and isinstance(question["options"], list):
                    numbered_options = []
                    for i, option_text in enumerate(question["options"], 1):
                        cleaned_text = re.sub(r'^\s*\d+\.\s*', '', option_text).strip()
                        numbered_options.append(f"  {i}. {cleaned_text}")
                    question["options"] = numbered_options
                
                if "subject" not in question:
                    question["subject"] = subject_area
            
            return data.get("questions", [])
        except:
            return []

    def _build_graph(self):
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("generate_quiz", self._generate_quiz_incremental)
        workflow.add_node("validate_quiz", self._validate_quiz_incremental)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "prepare_context")
        workflow.add_edge("prepare_context", "generate_quiz")
        workflow.add_edge("generate_quiz", "validate_quiz")
        
        workflow.add_conditional_edges(
            "validate_quiz",
            self._check_completion,
            {
                "generate_more": "generate_quiz",
                "complete": END
            }
        )
        
        self.workflow = workflow.compile()

    def _generate_subject_quiz(self, subject_area: str, target_count: int = 5, difficulty: str = "중급") -> Dict[str, Any]:
        """특정 과목의 문제를 순차로 생성"""
        if subject_area not in SUBJECT_AREAS:
            return {"error": f"유효하지 않은 과목: {subject_area}"}
        
        keywords = SUBJECT_AREAS[subject_area]["keywords"]
        
        all_validated_questions = []
        
        current_round = 0
        
        while len(all_validated_questions) < target_count and current_round < MAX_ROUNDS:
            current_round += 1
            remaining_needed = target_count - len(all_validated_questions)
            
            # 키워드를 2-3개씩 묶어서 순차 처리
            for i in range(0, len(keywords), 2):
                if len(all_validated_questions) >= target_count:
                    break
                    
                combo = " ".join(keywords[i:i+3])
                
                result = self._generate_with_keywords(combo, subject_area, remaining_needed, difficulty)
                
                if "questions" in result and result["questions"]:
                    new_questions = []
                    existing_questions = [q.get('question', '') for q in all_validated_questions]
                    
                    for q in result["questions"]:
                        if q.get('question', '') not in existing_questions:
                            new_questions.append(q)
                            existing_questions.append(q.get('question', ''))
                    
                    all_validated_questions.extend(new_questions)
                
                if len(all_validated_questions) >= target_count:
                    break
            
            if current_round < MAX_ROUNDS and len(all_validated_questions) < target_count:
                time.sleep(2)
        
        final_questions = all_validated_questions[:target_count]
        
        return {
            "subject_area": subject_area,
            "difficulty": difficulty,
            "requested_count": target_count,
            "quiz_count": len(final_questions),
            "questions": final_questions,
            "status": "SUCCESS" if len(final_questions) >= target_count else "PARTIAL"
        }

    def _generate_with_keywords(self, query: str, subject_area: str, needed_count: int, difficulty: str) -> Dict[str, Any]:
        """특정 키워드로 문제 생성"""
        try:
            initial_state = {
                "query": query,
                "quiz_count": needed_count,
                "target_quiz_count": needed_count,
                "difficulty": difficulty,
                "generation_attempts": 0,
                "quiz_questions": [],
                "validated_questions": [],
                "subject_area": subject_area
            }
            
            result = self.workflow.invoke(initial_state)
            
            if result.get("error"):
                return {"error": result["error"]}
            
            return {
                "questions": result.get("validated_questions", []),
                "used_sources": result.get("used_sources", [])
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _generate_full_exam(self, difficulty: str = "중급") -> Dict[str, Any]:
        """정보처리기사 전체 25문제를 순차로 생성"""
        start_time = time.time()
        
        full_exam_result = {
            "exam_title": "정보처리기사 모의고사",
            "total_questions": 0,
            "difficulty": difficulty,
            "subjects": {},
            "all_questions": [],
            "generation_summary": {},
            "failed_subjects": [],
            "model_info": "Groq llama-4-scout-17b-16e-instruct with LLM weakness analysis"
        }
        
        total_generated = 0
        
        for i, (subject_area, subject_info) in enumerate(SUBJECT_AREAS.items(), 1):
            target_count = subject_info["count"]
            
            subject_result = self._generate_subject_quiz(
                subject_area=subject_area,
                target_count=target_count,
                difficulty=difficulty
            )
            
            if "error" in subject_result:
                full_exam_result["failed_subjects"].append({
                    "subject": subject_area,
                    "error": subject_result["error"]
                })
            else:
                questions = subject_result["questions"]
                actual_count = len(questions)
                total_generated += actual_count
                
                full_exam_result["subjects"][subject_area] = {
                    "requested_count": target_count,
                    "actual_count": actual_count,
                    "questions": questions,
                    "status": subject_result.get("status", "UNKNOWN")
                }
                
                full_exam_result["all_questions"].extend(questions)
            
            if i < 5:
                time.sleep(2)
        
        elapsed_time = time.time() - start_time
        
        full_exam_result["total_questions"] = total_generated
        full_exam_result["generation_summary"] = {
            "target_total": 25,
            "actual_total": total_generated,
            "success_rate": f"{total_generated/25*100:.1f}%",
            "successful_subjects": 5 - len(full_exam_result["failed_subjects"]),
            "failed_subjects": len(full_exam_result["failed_subjects"]),
            "completion_status": "COMPLETE" if total_generated >= 25 else "PARTIAL",
            "generation_time": f"{elapsed_time:.1f}초"
        }
        
        return full_exam_result

    def _save_to_json(self, exam_result: Dict[str, Any], filename: str = None) -> str:
        """시험 결과를 JSON 파일로 저장"""
        save_dir = DEFAULT_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if "exam_title" in exam_result:
                filename = f"정보처리기사_25문제_{timestamp}.json"
            elif "quiz_type" in exam_result and exam_result["quiz_type"] == "weakness_based_llm":
                concepts = "_".join([c[:10] for c in exam_result.get("weakness_concepts", ["취약점"])[:3]])
                count = exam_result.get("quiz_count", 0)
                filename = f"LLM취약점맞춤_{concepts}_{count}문제_{timestamp}.json"
            else:
                subject = exam_result.get("subject_area", "문제")
                count = exam_result.get("quiz_count", 0)
                filename = f"{subject}_{count}문제_{timestamp}.json"
        
        if not os.path.isabs(filename):
            filename = os.path.join(save_dir, filename)
        elif not filename.startswith(save_dir):
            filename = os.path.join(save_dir, os.path.basename(filename))
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(exam_result, f, ensure_ascii=False, indent=2)
        
        return filename


# LLM 취약점 분석 전용 편의 함수들

def generate_weakness_quiz_from_analysis_llm(
    agent: InfoProcessingExamAgent,
    analysis_file_path: str,
    target_count: int = 10,
    difficulty: str = "중급",
    save_to_file: bool = True,
    filename: str = None
) -> Dict[str, Any]:
    """
    LLM을 활용하여 분석 결과 파일을 기반으로 취약점 맞춤 문제를 생성하는 편의 함수
    
    Args:
        agent: InfoProcessingExamAgent 인스턴스
        analysis_file_path: 분석 결과 JSON 파일 경로
        target_count: 생성할 문제 수
        difficulty: 난이도
        save_to_file: 파일 저장 여부
        filename: 저장할 파일명
        
    Returns:
        생성 결과
    """
    input_data = {
        "mode": "weakness_quiz",
        "analysis_file_path": analysis_file_path,
        "target_count": target_count,
        "difficulty": difficulty,
        "save_to_file": save_to_file,
        "filename": filename
    }
    
    return agent.execute(input_data)


def generate_weakness_quiz_from_text_llm(
    agent: InfoProcessingExamAgent,
    analysis_text: str,
    target_count: int = 8,
    difficulty: str = "중급",
    save_to_file: bool = True,
    filename: str = None
) -> Dict[str, Any]:
    """
    LLM을 활용하여 분석 텍스트를 기반으로 취약점 맞춤 문제를 생성하는 편의 함수
    
    Args:
        agent: InfoProcessingExamAgent 인스턴스
        analysis_text: 분석 텍스트 내용
        target_count: 생성할 문제 수
        difficulty: 난이도
        save_to_file: 파일 저장 여부
        filename: 저장할 파일명
        
    Returns:
        생성 결과
    """
    input_data = {
        "mode": "weakness_quiz",
        "raw_analysis_text": analysis_text,
        "target_count": target_count,
        "difficulty": difficulty,
        "save_to_file": save_to_file,
        "filename": filename
    }
    
    return agent.execute(input_data)


# 업데이트된 대화형 인터페이스
def interactive_menu_llm():
    """LLM 기반 취약점 분석을 포함한 대화형 메뉴 시스템"""
    try:
        agent = InfoProcessingExamAgent(
            data_folder=DEFAULT_DATA_FOLDER
        )
        
        print(f"\n🧠 {agent.name} 초기화 완료")
        print(f"📖 설명: {agent.description}")
        
        while True:
            print("\n" + "="*70)
            print("  🧠 LLM 기반 정보처리기사 맞춤형 문제 생성 에이전트")
            print("="*70)
            print("1. 전체 25문제 생성")
            print("2. 특정 과목만 문제 생성")
            print("3. 🧠 LLM 취약점 분석 + 맞춤 문제 생성 (파일)")
            print("4. 🧠 LLM 취약점 분석 + 맞춤 문제 생성 (텍스트)")
            print("5. 사용 가능한 PDF 목록 보기")
            print("0. 종료")
            print("-"*70)
            
            choice = input("선택하세요: ").strip()
            
            if choice == "1":
                # 전체 25문제 생성 (기존과 동일)
                difficulty = input("난이도 (초급/중급/고급, 기본값: 중급): ").strip()
                if difficulty not in ["초급", "중급", "고급"]:
                    difficulty = "중급"
                
                save_option = input("JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower()
                save_to_file = save_option == 'y'
                
                filename = None
                if save_to_file:
                    filename_input = input("파일명 (엔터: 자동생성): ").strip()
                    if filename_input:
                        filename = filename_input
                
                input_data = {
                    "mode": "full_exam",
                    "difficulty": difficulty,
                    "save_to_file": save_to_file,
                    "filename": filename
                }
                
                print("\n전체 25문제 생성 중...")
                result = agent.execute(input_data)
                
                if result["success"]:
                    exam_data = result["result"]
                    summary = exam_data.get("generation_summary", {})
                    
                    print(f"\n✅ 생성 완료!")
                    print(f"전체 문제 수: {summary.get('actual_total', 0)}/25문제")
                    print(f"성공률: {summary.get('success_rate', '0%')}")
                    print(f"소요 시간: {summary.get('generation_time', 'N/A')}")
                    
                    if "file_path" in result:
                        print(f"📁 저장 경로: {result['file_path']}")
                else:
                    print(f"❌ 실패: {result['error']}")
            
            elif choice == "2":
                # 특정 과목 문제 생성 (기존과 동일)
                print("\n[정보처리기사 과목 선택]")
                subjects = list(agent.SUBJECT_AREAS.keys())
                for i, subject in enumerate(subjects, 1):
                    count = agent.SUBJECT_AREAS[subject]["count"]
                    print(f"{i}. {subject} ({count}문제)")
                
                try:
                    subject_choice = int(input("과목 번호 선택: "))
                    if 1 <= subject_choice <= len(subjects):
                        selected_subject = subjects[subject_choice - 1]
                        default_count = agent.SUBJECT_AREAS[selected_subject]["count"]
                        
                        count_input = input(f"생성할 문제 수 (기본값: {default_count}): ").strip()
                        target_count = int(count_input) if count_input.isdigit() else default_count
                        
                        difficulty = input("난이도 (초급/중급/고급, 기본값: 중급): ").strip()
                        if difficulty not in ["초급", "중급", "고급"]:
                            difficulty = "중급"
                        
                        save_option = input("JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower()
                        save_to_file = save_option == 'y'
                        
                        filename = None
                        if save_to_file:
                            filename_input = input("파일명 (엔터: 자동생성): ").strip()
                            if filename_input:
                                filename = filename_input
                        
                        input_data = {
                            "mode": "subject_quiz",
                            "subject_area": selected_subject,
                            "target_count": target_count,
                            "difficulty": difficulty,
                            "save_to_file": save_to_file,
                            "filename": filename
                        }
                        
                        print(f"\n{selected_subject} 과목 {target_count}문제 생성 중...")
                        result = agent.execute(input_data)
                        
                        if result["success"]:
                            subject_data = result["result"]
                            print(f"✅ 생성 완료!")
                            print(f"{subject_data['subject_area']}: {subject_data['quiz_count']}/{subject_data['requested_count']}문제")
                            print(f"상태: {subject_data.get('status', 'UNKNOWN')}")
                            
                            if "file_path" in result:
                                print(f"📁 저장 경로: {result['file_path']}")
                        else:
                            print(f"❌ 실패: {result['error']}")
                    else:
                        print("유효하지 않은 과목 번호입니다.")
                except ValueError:
                    print("숫자를 입력해주세요.")
            
            elif choice == "3":
                # LLM 기반 취약점 분석 + 맞춤 문제 생성 (파일)
                print("\n🧠 [LLM 기반 취약점 분석 + 맞춤 문제 생성 - 파일]")
                
                analysis_file_path = input("분석 결과 JSON 파일 경로를 입력하세요: ").strip()
                
                if not os.path.exists(analysis_file_path):
                    print("❌ 파일이 존재하지 않습니다.")
                    continue
                
                try:
                    # 분석 파일 미리보기
                    with open(analysis_file_path, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    
                    print(f"\n📋 분석 파일 로드 완료")
                    
                    count_input = input("생성할 문제 수 (기본값: 10): ").strip()
                    target_count = int(count_input) if count_input.isdigit() else 10
                    
                    difficulty = input("난이도 (초급/중급/고급, 기본값: 중급): ").strip()
                    if difficulty not in ["초급", "중급", "고급"]:
                        difficulty = "중급"
                    
                    save_option = input("JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower()
                    save_to_file = save_option == 'y'
                    
                    filename = None
                    if save_to_file:
                        filename_input = input("파일명 (엔터: 자동생성): ").strip()
                        if filename_input:
                            filename = filename_input
                    
                    print(f"\n🧠 LLM이 취약점을 분석하고 맞춤 문제 {target_count}개를 생성 중...")
                    
                    result = generate_weakness_quiz_from_analysis_llm(
                        agent=agent,
                        analysis_file_path=analysis_file_path,
                        target_count=target_count,
                        difficulty=difficulty,
                        save_to_file=save_to_file,
                        filename=filename
                    )
                    
                    if result["success"]:
                        weakness_data = result["result"]
                        print(f"\n✅ LLM 취약점 분석 및 맞춤 문제 생성 완료!")
                        print(f"🧠 LLM이 분석한 취약점 개념: {weakness_data.get('weakness_concepts', [])}")
                        print(f"📚 집중 추천 과목: {weakness_data.get('weakness_analysis', {}).get('subject_focus', [])}")
                        print(f"🎯 추천 난이도: {weakness_data.get('weakness_analysis', {}).get('difficulty_level', '중급')}")
                        print(f"📊 생성된 문제 수: {weakness_data.get('quiz_count', 0)}/{weakness_data.get('requested_count', 0)}")
                        print(f"📈 성공률: {weakness_data.get('generation_summary', {}).get('success_rate', '0%')}")
                        
                        # 학습 우선순위 표시
                        learning_priorities = weakness_data.get('weakness_analysis', {}).get('learning_priorities', [])
                        if learning_priorities:
                            print(f"📝 추천 학습 순서:")
                            for i, priority in enumerate(learning_priorities[:5], 1):
                                print(f"  {i}. {priority}")
                        
                        # 문제 미리보기
                        questions = weakness_data.get("questions", [])
                        if questions and input("\n생성된 문제를 미리보시겠습니까? (y/n): ").strip().lower() == 'y':
                            for i, q in enumerate(questions[:2], 1):
                                weakness_concept = q.get('weakness_concept', '일반')
                                weakness_focus = q.get('weakness_focus', weakness_concept)
                                print(f"\n[🎯 취약점 집중: {weakness_focus}] [문제 {i}]")
                                print(f"❓ {q.get('question', '')}")
                                for option in q.get('options', []):
                                    print(f"{option}")
                                print(f"✅ 정답: {q.get('answer', '')}")
                                print(f"💡 해설: {q.get('explanation', '')}")
                                if i < 2 and i < len(questions):
                                    input("다음 문제를 보려면 Enter를 누르세요...")
                            
                            if len(questions) > 2:
                                print(f"\n... 외 {len(questions)-2}개 문제가 더 있습니다.")
                        
                        if "file_path" in result:
                            print(f"📁 저장 경로: {result['file_path']}")
                    else:
                        print(f"❌ 실패: {result['error']}")
                        
                except Exception as e:
                    print(f"❌ 분석 파일 처리 중 오류: {e}")
            
            elif choice == "4":
                # LLM 기반 취약점 분석 + 맞춤 문제 생성 (텍스트)
                print("\n🧠 [LLM 기반 취약점 분석 + 맞춤 문제 생성 - 텍스트 입력]")
                
                print("학습자의 취약점이나 분석 내용을 자유롭게 입력하세요.")
                print("예: '자료흐름도 구성요소 이해 부족, SQL 조인 연산 실수 많음, UML 다이어그램 해석 어려움'")
                print("(여러 줄 입력 가능, 완료 후 빈 줄에서 Enter)")
                
                analysis_lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    analysis_lines.append(line)
                
                analysis_text = "\n".join(analysis_lines)
                
                if not analysis_text.strip():
                    print("❌ 분석 내용이 입력되지 않았습니다.")
                    continue
                
                print(f"\n📝 입력된 분석 내용:")
                print(f"{analysis_text[:200]}...")
                
                count_input = input("\n생성할 문제 수 (기본값: 8): ").strip()
                target_count = int(count_input) if count_input.isdigit() else 8
                
                difficulty = input("난이도 (초급/중급/고급, 기본값: 중급): ").strip()
                if difficulty not in ["초급", "중급", "고급"]:
                    difficulty = "중급"
                
                save_option = input("JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower()
                save_to_file = save_option == 'y'
                
                filename = None
                if save_to_file:
                    filename_input = input("파일명 (엔터: 자동생성): ").strip()
                    if filename_input:
                        filename = filename_input
                
                print(f"\n🧠 LLM이 입력 내용을 분석하고 맞춤 문제 {target_count}개를 생성 중...")
                
                result = generate_weakness_quiz_from_text_llm(
                    agent=agent,
                    analysis_text=analysis_text,
                    target_count=target_count,
                    difficulty=difficulty,
                    save_to_file=save_to_file,
                    filename=filename
                )
                
                if result["success"]:
                    weakness_data = result["result"]
                    print(f"\n✅ LLM 텍스트 분석 및 맞춤 문제 생성 완료!")
                    print(f"🧠 LLM이 추출한 취약점: {weakness_data.get('weakness_concepts', [])}")
                    print(f"📚 집중 추천 과목: {weakness_data.get('weakness_analysis', {}).get('subject_focus', [])}")
                    print(f"🎯 LLM 추천 난이도: {weakness_data.get('weakness_analysis', {}).get('difficulty_level', '중급')}")
                    print(f"📊 생성된 문제 수: {weakness_data.get('quiz_count', 0)}/{weakness_data.get('requested_count', 0)}")
                    print(f"📈 성공률: {weakness_data.get('generation_summary', {}).get('success_rate', '0%')}")
                    
                    # 추천 문제 유형 표시
                    question_types = weakness_data.get('weakness_analysis', {}).get('question_types', [])
                    if question_types:
                        print(f"📋 추천 문제 유형: {', '.join(question_types)}")
                    
                    # 문제 미리보기
                    questions = weakness_data.get("questions", [])
                    if questions and input("\n생성된 문제를 미리보시겠습니까? (y/n): ").strip().lower() == 'y':
                        for i, q in enumerate(questions[:2], 1):
                            weakness_concept = q.get('weakness_concept', '일반')
                            weakness_focus = q.get('weakness_focus', weakness_concept)
                            print(f"\n[🎯 취약점 집중: {weakness_focus}] [문제 {i}]")
                            print(f"❓ {q.get('question', '')}")
                            for option in q.get('options', []):
                                print(f"{option}")
                            print(f"✅ 정답: {q.get('answer', '')}")
                            print(f"💡 해설: {q.get('explanation', '')}")
                            if i < 2 and i < len(questions):
                                input("다음 문제를 보려면 Enter를 누르세요...")
                        
                        if len(questions) > 2:
                            print(f"\n... 외 {len(questions)-2}개 문제가 더 있습니다.")
                    
                    if "file_path" in result:
                        print(f"📁 저장 경로: {result['file_path']}")
                else:
                    print(f"❌ 실패: {result['error']}")
            
            elif choice == "5":
                # PDF 파일 목록 보기 (RAG 엔진 사용)
                pdf_files = agent.rag_engine.get_pdf_files()
                if pdf_files:
                    print(f"\n=== '{agent.rag_engine.data_folder}' 폴더의 PDF 파일 목록 ===")
                    for i, file_path in enumerate(pdf_files, 1):
                        filename = os.path.basename(file_path)
                        file_size = os.path.getsize(file_path) / 1024
                        print(f"{i}. {filename} ({file_size:.1f} KB)")
                else:
                    print(f"'{agent.rag_engine.data_folder}' 폴더에 PDF 파일이 없습니다.")
            
            elif choice == "0":
                print("🧠 LLM 기반 에이전트를 종료합니다.")
                break
            
            else:
                print("잘못된 선택입니다. 0~5 중에서 선택해주세요.")
    
    except Exception as e:
        print(f"에이전트 초기화 실패: {e}")


def main():
    """메인 실행 함수"""
    ***REMOVED*** API 키 확인
    if not os.getenv("GROQ_API_KEY=REDACTED("GROQ_API_KEY=REDACTED
    
    # 사용 방법 선택
    print("🧠 LLM 기반 정보처리기사 맞춤형 문제 생성 에이전트")
    print("1. 대화형 인터페이스 사용")
    print("2. LLM 취약점 분석 기능 테스트")
    
    choice = input("선택하세요 (1 또는 2): ").strip()
    
    if choice == "1":
        interactive_menu_llm()
    elif choice == "2":
        # JSON 파일에서 취약점 분석 테스트
        try:
            agent = InfoProcessingExamAgent()
            
            # test_sample 폴더에서 분석 파일 선택
            test_sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_sample")
            
            if not os.path.exists(test_sample_dir):
                print(f"❌ test_sample 폴더를 찾을 수 없습니다: {test_sample_dir}")
                return
            
            # 폴더 내 JSON 파일 목록 가져오기
            json_files = [f for f in os.listdir(test_sample_dir) if f.endswith('.json')]
            
            if not json_files:
                print(f"❌ {test_sample_dir} 폴더에 JSON 파일이 없습니다.")
                return
            
            print(f"\n📁 {test_sample_dir} 폴더의 분석 파일 목록:")
            for i, filename in enumerate(json_files, 1):
                file_path = os.path.join(test_sample_dir, filename)
                file_size = os.path.getsize(file_path) / 1024
                print(f"{i}. {filename} ({file_size:.1f} KB)")
            
            # 사용자가 파일 선택
            while True:
                try:
                    file_choice = input(f"\n분석할 파일 번호를 선택하세요 (1-{len(json_files)}): ").strip()
                    file_index = int(file_choice) - 1
                    
                    if 0 <= file_index < len(json_files):
                        selected_filename = json_files[file_index]
                        analysis_file_path = os.path.join(test_sample_dir, selected_filename)
                        break
                    else:
                        print(f"1-{len(json_files)} 사이의 숫자를 입력해주세요.")
                except ValueError:
                    print("유효한 숫자를 입력해주세요.")
            
            print(f"\n📁 선택된 분석 파일: {selected_filename}")
            print(f"📁 파일 경로: {analysis_file_path}")
            
            analysis_data = agent.load_analysis_from_file(analysis_file_path)
            
            print("🧠 JSON 파일에서 취약점 개념 추출 중...")
            weakness_concepts = agent._extract_weakness_concepts_from_analysis(analysis_data)
            
            if weakness_concepts:
                print(f"✅ 취약점 개념 추출 완료!")
                print(f"🧠 추출된 취약점 개념: {weakness_concepts}")
                
                print("\n🔧 추출된 취약점 기반 맞춤 문제 생성 중...")
                result = agent._generate_weakness_quiz(
                    input_data={
                        "analysis_file_path": analysis_file_path,
                        "target_count": 5
                    },
                    difficulty="중급"
                )
                
                if "error" not in result:
                    print(f"✅ 문제 생성 완료!")
                    print(f"📝 생성된 문제 수: {result.get('quiz_count', 0)}")
                    print(f"🎯 취약점 개념 수: {result.get('generation_summary', {}).get('analyzed_concepts', 0)}")
                    
                    # 결과를 파일로 저장 (자동 번호 증가)
                    base_filename = "weakness_based_quiz"
                    counter = 1
                    
                    # 기존 파일이 있는지 확인하고 다음 번호 찾기
                    while os.path.exists(f"{base_filename}{counter}_result.json"):
                        counter += 1
                    
                    output_file = f"{base_filename}{counter}_result.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"💾 결과가 {output_file}에 저장되었습니다.")
                else:
                    print(f"❌ 문제 생성 실패: {result['error']}")
            else:
                print("❌ 취약점 개념을 추출할 수 없습니다.")
                
        except Exception as e:
            print(f"❌ 취약점 분석 테스트 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()

