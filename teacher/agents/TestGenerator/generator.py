import os
import json
import re
import glob
import time
from typing import List, Dict, Any, TypedDict, Annotated
from collections import Counter
from pathlib import Path
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from common.milvus_helpers import search_milvus_documents_by_subject, create_context_from_documents
from ..base_agent import BaseAgent

# MilvusDB 관련 임포트는 common.milvus_helpers에서 처리

# RAGAS 관련 코드 제거 (LLM 기반 검증 사용)

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

class SubjectState(TypedDict):
    """과목별 독립적인 상태 정의"""
    query: str
    documents: List[Document]
    context: str
    quiz_questions: List[Dict[str, Any]]
    difficulty: str
    error: str
    used_sources: List[str]
    generation_attempts: int
    target_quiz_count: int
    subject_area: str
    validated_questions: List[Dict[str, Any]]
    node_id: int

class GraphState(TypedDict):
    """전체 그래프 상태 (과목별 결과 수집)"""
    # 과목별 독립적인 상태들
    소프트웨어설계: SubjectState
    소프트웨어개발: SubjectState
    데이터베이스구축: SubjectState
    프로그래밍언어활용: SubjectState
    정보시스템구축관리: SubjectState
    
    # 전체 결과
    all_questions: List[Dict[str, Any]]
    total_questions: int
    generation_summary: Dict[str, Any]
    failed_subjects: List[Dict[str, Any]]


class InfoProcessingExamAgent(BaseAgent):
    """
    정보처리기사 출제기준에 맞는 자동 출제 에이전트
    - full_exam: 5과목 × 20문항 = 총 100문항
    - subject_quiz: 특정 과목 최대 40문항
    - 과목별 생성/검증 노드 2개(총 10개)
    - 사용자 지정 병렬 실행
    - 머지 순서 고정
    """

    # 1) 과목/키워드 + full_exam 기본 카운트(20)로 변경
    SUBJECT_AREAS = {
        "소프트웨어설계": {
            "count": 20,
            "keywords": ["요구사항", "UI 설계", "애플리케이션 설계", "인터페이스", "UML", "객체지향", "디자인패턴", "모듈화", "결합도", "응집도"]
        },
        "소프트웨어개발": {
            "count": 20,
            "keywords": ["자료구조", "스택", "큐", "리스트", "통합구현", "모듈", "패키징", "테스트케이스", "알고리즘", "인터페이스"]
        },
        "데이터베이스구축": {
            "count": 20,
            "keywords": ["SQL", "트리거", "DML", "DDL", "DCL", "정규화", "관계형모델", "E-R모델", "데이터모델링", "무결성"]
        },
        "프로그래밍언어활용": {
            "count": 20,
            "keywords": ["개발환경", "프로그래밍언어", "라이브러리", "운영체제", "네트워크", "데이터타입", "변수", "연산자"]
        },
        "정보시스템구축관리": {
            "count": 20,
            "keywords": ["소프트웨어개발방법론", "프로젝트관리", "보안", "시스템보안", "네트워크보안", "테일러링", "생명주기모델"]
        }
    }

    # 4) 최종 머지 순서
    MERGE_ORDER = [
        "소프트웨어설계",
        "소프트웨어개발",
        "데이터베이스구축",
        "프로그래밍언어활용",
        "정보시스템구축관리",
    ]

    def __init__(self, data_folder=None, groq_api_key=None):
        if data_folder is None:
            base_dir = Path(__file__).resolve().parent
            data_folder = base_dir / "data"
        self.data_folder = Path(data_folder)
        os.makedirs(self.data_folder, exist_ok=True)

        if groq_api_key:
            os.environ["OPENAI_API_KEY"] = groq_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API 키가 필요합니다.")

        self.embeddings_model = None
        self.llm = None
        self.workflow = None
        self._current_milvus_data = None  # MilvusDB 연결 정보 저장용
        # MilvusDB는 이제 supervisor에서 관리됨

        self._initialize_models()
        self._build_graph()  # 2) 과목별 2노드(생성/검증) 구축

    # --- Subject helpers: spacing-insensitive matching ---
    def _normalize_subject(self, s: str) -> str:
        try:
            return re.sub(r"\s+", "", (s or "")).strip()
        except Exception:
            return (s or "").strip()

    def _subject_aliases(self, subject: str) -> List[str]:
        base = self._normalize_subject(subject)
        alias_map = {
            "소프트웨어설계": ["소프트웨어설계", "소프트웨어 설계"],
            "소프트웨어개발": ["소프트웨어개발", "소프트웨어 개발"],
            "데이터베이스구축": ["데이터베이스구축", "데이터베이스 구축"],
            "프로그래밍언어활용": ["프로그래밍언어활용", "프로그래밍언어 활용", "프로그래밍 언어 활용"],
            "정보시스템구축관리": ["정보시스템구축관리", "정보시스템 구축관리", "정보시스템 구축 관리"],
        }
        for key, aliases in alias_map.items():
            if base == self._normalize_subject(key):
                return aliases
        # 기본: 입력 그대로와 공백 제거형 둘 다 시도
        uniq = []
        for cand in [subject, base]:
            if cand and cand not in uniq:
                uniq.append(cand)
        return uniq

    @property
    def name(self) -> str:
        return "InfoProcessingExamAgent"

    @property
    def description(self) -> str:
        return "정보처리기사 5과목 기준으로 문제를 생성/검증하여 100문제(또는 과목별 지정 수)를 자동 생성합니다."

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args (확장):
          - mode: "full_exam" | "subject_quiz" | "partial_exam"
          - difficulty: "초급" | "중급" | "고급" (default: "중급")
          - subject_area: subject_quiz 모드에서 필수
          - target_count: subject_quiz 모드에서 요청 문항 수 (최대 40)
          - selected_subjects: partial_exam 모드에서 선택할 과목 리스트
          - questions_per_subject: partial_exam 모드에서 과목당 문제 수
          - parallel_agents: 동시 병렬 실행 개수 (default: 2, 권장: 2~5)
          - save_to_file: bool
          - filename: 저장 파일명
          - milvus_question_path: (선택) JSON 문제은행 경로. 지정 시 해당 JSON을 Milvus에 적재 후 검색에 사용
        """
        try:
            mode = input_data.get("mode", "full_exam")
            difficulty = input_data.get("difficulty", "중급")
            save_to_file = input_data.get("save_to_file", False)
            filename = input_data.get("filename")
            parallel_agents = max(1, int(input_data.get("parallel_agents", 2)))  # 3) 병렬 개수

            # MilvusDB는 supervisor에서 관리됨

            if mode == "full_exam":
                # 1) 5과목 × 20문항 = 총 100문항
                result = self._generate_full_exam(difficulty=difficulty,
                                                  parallel_agents=parallel_agents,
                                                  milvus_data=input_data.get("milvus_data"))
            elif mode == "partial_exam":
                # 선택된 과목들에 대해 지정된 문제 수만큼 생성
                selected_subjects = input_data.get("selected_subjects", [])
                questions_per_subject = input_data.get("questions_per_subject", 10)
                
                if not selected_subjects or not isinstance(selected_subjects, list):
                    return {
                        "success": False,
                        "error": "partial_exam 모드에서는 selected_subjects 리스트가 필요합니다."
                    }
                
                if not all(subj in self.SUBJECT_AREAS for subj in selected_subjects):
                    invalid_subjects = [subj for subj in selected_subjects if subj not in self.SUBJECT_AREAS]
                    return {
                        "success": False,
                        "error": f"유효하지 않은 과목명입니다: {invalid_subjects}. 가능한 과목: {list(self.SUBJECT_AREAS.keys())}"
                    }
                
                result = self._generate_partial_exam(
                    selected_subjects=selected_subjects,
                    questions_per_subject=questions_per_subject,
                    difficulty=difficulty,
                    parallel_agents=parallel_agents,
                    milvus_data=input_data.get("milvus_data")
                )
            elif mode == "subject_quiz":
                subject_area = input_data.get("subject_area")
                if not subject_area or subject_area not in self.SUBJECT_AREAS:
                    return {
                        "success": False,
                        "error": f"유효하지 않은 과목명입니다. 가능한 과목: {list(self.SUBJECT_AREAS.keys())}"
                    }
                # 최대 40개 제한
                target_count = min(int(input_data.get("target_count", 20)), 40)
                result = self._generate_subject_quiz(
                    subject_area=subject_area,
                    target_count=target_count,
                    difficulty=difficulty,
                    milvus_data=input_data.get("milvus_data")
                )
                # subject_quiz는 단일 과목 결과만 리턴
                if "error" in result:
                    return {"success": False, "error": result["error"]}
                response = {"success": True, "result": result}
                if save_to_file:
                    try:
                        file_path = self._save_to_json(result, filename)
                        response["file_path"] = file_path
                    except Exception as e:
                        response["save_error"] = str(e)
                return response
            else:
                return {"success": False, "error": "유효하지 않은 모드입니다. 'full_exam' 또는 'subject_quiz'를 사용하세요."}

            if "error" in result:
                return {"success": False, "error": result["error"]}

            response = {"success": True, "result": result}
            if save_to_file:
                try:
                    file_path = self._save_to_json(result, filename)
                    response["file_path"] = file_path
                except Exception as e:
                    response["save_error"] = str(e)
            return response

        except Exception as e:
            return {"success": False, "error": f"에이전트 실행 중 오류 발생: {str(e)}"}

    def _initialize_models(self):
        try:
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.llm = ChatOpenAI(
                model=OPENAI_LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                timeout=LLM_TIMEOUT,
                max_retries=LLM_MAX_RETRIES,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
            _ = self.llm.invoke("ping")
        except Exception as e:
            raise ValueError(f"모델 초기화 중 오류 발생: {e}")

    # MilvusDB 리트리버는 이제 supervisor에서 관리됨

    # MilvusDB는 이제 supervisor에서 관리됨

    def get_pdf_files(self) -> List[str]:
        # Milvus 사용으로 의미는 적지만, 외부 호환을 위해 남김
        return glob.glob(os.path.join(self.data_folder, "*.pdf"))

    # ---- 공통 노드 구현(그대로 재사용) ----
    def _retrieve_documents(self, state) -> dict:
        try:
            query = state["query"]
            subject_area = state.get("subject_area", "")
            enhanced_query = f"{subject_area} {query}".strip()
            print(f"[DEBUG] _retrieve_documents: query='{query}', subject_area='{subject_area}', enhanced_query='{enhanced_query}'")
            
            documents: List[Document] = []
            
            # 전역 milvus_data 사용
            if hasattr(self, '_current_milvus_data') and self._current_milvus_data and self._current_milvus_data.get("connection_status", False):
                print(f"🔍 MilvusDB에서 과목별 문서 검색 중: {subject_area}")
                
                # 과목명으로 개념 관련 문서 검색
                concept_docs = search_milvus_documents_by_subject(
                    milvus_data=self._current_milvus_data,
                    collection_name="concepts",
                    subject_area=subject_area,
                    k=20
                )
                
                # 과목명으로 문제 관련 문서 검색
                problem_docs = search_milvus_documents_by_subject(
                    milvus_data=self._current_milvus_data,
                    collection_name="problems",
                    subject_area=subject_area,
                    k=30
                )
                
                # 문서 합치기
                documents = concept_docs + problem_docs
                
                if documents:
                    print(f"✅ MilvusDB 과목 검색 완료: {subject_area} - {len(documents)}개 문서")
                else:
                    print(f"⚠️ MilvusDB에서 {subject_area} 과목 관련 문서를 찾지 못함")
            else:
                print("⚠️ MilvusDB 연결 안됨 - 빈 문서로 진행")

            print(f"[DEBUG] _retrieve_documents: found {len(documents)} documents")
            
            # Milvus 문서에는 source_file이 없을 수 있으므로 보완
            source_files = []
            for doc in documents:
                src = doc.metadata.get('source_file') or doc.metadata.get('subject') or 'milvus'
                source_files.append(src)
            used_sources = list(Counter(source_files).keys())
            return {**state, "documents": documents, "used_sources": used_sources}
        except Exception as e:
            print(f"[DEBUG] _retrieve_documents: error {e}")
            return {**state, "error": f"문서 검색 오류: {e}"}

    def _prepare_context(self, state) -> dict:
        documents = state.get("documents", [])
        key_sents = []
        for doc in documents:
            for line in doc.page_content.split("\n"):
                line = line.strip()
                if len(line) > 100 or any(k in line for k in ["정의", "특징", "종류", "예시", "원리", "구성", "절차", "장점", "단점"]):
                    key_sents.append(line)
        context = "\n".join(key_sents)[:2000]
        # subject_area를 명시적으로 유지
        subject_area = state.get("subject_area", "")
        print(f"[DEBUG] _prepare_context: subject_area='{subject_area}'")
        return {**state, "context": context, "subject_area": subject_area}

# RAGAS 검증 함수 제거 (LLM 기반 검증 사용)

    def _generate_quiz_incremental(self, state) -> dict:
        try:
            context = state.get("context", "")
            target_quiz_count = state.get("target_quiz_count", 5)
            validated_questions = state.get("validated_questions", [])
            subject_area = state.get("subject_area", "")
            needed_count = target_quiz_count - len(validated_questions)
            print(f"[DEBUG] _generate_quiz_incremental: context_len={len(context)}, target={target_quiz_count}, validated={len(validated_questions)}, needed={needed_count}")

            if needed_count <= 0:
                return {**state, "quiz_questions": validated_questions}
            if not context.strip():
                new_attempts = state.get("generation_attempts", 0) + 1
                print(f"[DEBUG] _generate_quiz_incremental: no context, attempts={new_attempts}")
                # 컨텍스트 없을 때 과목 일반 개념 기반 생성 폴백
                fallback_prompt = (
                    f"당신은 정보처리기사 출제 전문가입니다. {subject_area} 과목의 다음 핵심 개념들을 바탕으로 "
                    f"객관식 문제 {needed_count}개를 생성하세요.\n\n"
                    "출제 규칙:\n"
                    "1) 보기에는 번호(1. 2. 3. 4.)를 절대 붙이지 말고, 순수 텍스트만 사용하세요.\n"
                    "2) 정답(answer)은 보기의 '번호'(문자열)로만 적으세요. 예: \"2\"\n"
                    "3) 문제는 중복 없이 간결하고 명확하게 작성하세요.\n"
                    "4) 보기는 상호 배타적이며 길이를 너무 길게 만들지 마세요(각 3~12단어 권장).\n"
                    "5) 해설(explanation)은 정답 근거를 한두 문장으로 명확히 설명하세요.\n"
                    "6) 아래 JSON 외의 텍스트는 절대 출력하지 마세요.\n\n"
                    "{\n  \"questions\": [\n    {\n      \"question\": \"문제 내용을 여기에 작성\",\n      \"options\": [\"선택지1\", \"선택지2\", \"선택지3\", \"선택지4\"],\n      \"answer\": \"1\",\n      \"explanation\": \"정답에 대한 간단한 해설\"\n    }\n  ]\n}"
                )
                try:
                    self.llm.temperature = 0.15
                    self.llm.max_tokens = 900
                    fb_resp = self.llm.invoke(fallback_prompt)
                    fb_content = getattr(fb_resp, "content", str(fb_resp))
                    new_questions = self._parse_quiz_response(fb_content, subject_area)
                    new_questions = self._filter_duplicate_questions(new_questions, validated_questions, subject_area)
                    if new_questions:
                        return {
                            **state,
                            "quiz_questions": new_questions,
                            "validated_questions": validated_questions,
                            "generation_attempts": new_attempts
                        }
                except Exception:
                    pass
                return {
                    **state, 
                    "quiz_questions": [],
                    "validated_questions": validated_questions,
                    "generation_attempts": new_attempts,
                    "error": "검색된 문서 내용이 없습니다."
                }

            # 부족한 문제 수만큼 생성 (최대 20문제, 한 번에 모두 생성)
            generate_count = max(min(needed_count, 20), 1)

            # 🔧 JSON 형식에 주석이 들어가던 문제 수정
            prompt_template = PromptTemplate(
                input_variables=["context", "subject_area", "needed_count"],
                template=(
                    "당신은 정보처리기사 출제 전문가입니다. 아래 문서 내용을 바탕으로 {subject_area} 과목의 객관식 문제 {needed_count}개를 생성하세요.\n\n"
                    "조건:\n"
                    "• 보기에는 번호(1. 2. 3. 4.)를 붙이지 마십시오.\n"
                    "• answer에는 정답의 '번호'만 문자열로 적으십시오. 예: \"2\"\n"
                    "• 각 문제는 서로 다른 내용이어야 합니다.\n"
                    "• 출력은 아래 JSON 형식만 포함하십시오. 다른 텍스트 금지.\n\n"
                    "[문서 내용]\n{context}\n\n"
                    "[응답 형식]\n"
                    "{{\n"
                    "  \"questions\": [\n"
                    "    {{\n"
                    "      \"question\": \"문제 내용을 여기에 작성\",\n"
                    "      \"options\": [\"선택지1\", \"선택지2\", \"선택지3\", \"선택지4\"],\n"
                    "      \"answer\": \"1\",\n"
                    "      \"explanation\": \"정답에 대한 간단한 해설\"\n"
                    "    }},\n"
                    "    {{\n"
                    "      \"question\": \"두 번째 문제 내용\",\n"
                    "      \"options\": [\"선택지1\", \"선택지2\", \"선택지3\", \"선택지4\"],\n"
                    "      \"answer\": \"2\",\n"
                    "      \"explanation\": \"정답에 대한 간단한 해설\"\n"
                    "    }}\n"
                    "    // ... {needed_count}개까지 반복\n"
                    "  ]\n"
                    "}}\n"
                )
            )

            prompt = prompt_template.format(
                context=context, subject_area=subject_area, needed_count=generate_count
            )

            print(f"[DEBUG] _generate_quiz_incremental: calling LLM for {generate_count} questions")
            self.llm.temperature = 0.15
            self.llm.max_tokens = 4000  # 여러 문제 생성에 충분한 토큰 수
            response = self.llm.invoke(prompt)
            response_content = getattr(response, "content", str(response))
            print(f"[DEBUG] _generate_quiz_incremental: LLM response length={len(response_content)}")
            
            new_questions = self._parse_quiz_response(response_content, subject_area)
            print(f"[DEBUG] _generate_quiz_incremental: parsed {len(new_questions)} questions before filtering")
            
            # 중복 필터링(동일 턴/이전/벡터스토어 유사 제거)
            new_questions = self._filter_duplicate_questions(new_questions, validated_questions, subject_area)
            print(f"[DEBUG] _generate_quiz_incremental: after filtering: {len(new_questions)} questions")

            if not new_questions:
                new_attempts = state.get("generation_attempts", 0) + 1
                print(f"[DEBUG] _generate_quiz_incremental: failed to generate questions, attempts={new_attempts}")
                return {
                    **state,
                    "quiz_questions": [],
                    "validated_questions": validated_questions,
                    "generation_attempts": new_attempts,
                    "error": "유효한 문제를 생성하지 못했습니다."
                }

            # LLM 기반 검증으로 변경 (RAGAS 제거)
            new_attempts = state.get("generation_attempts", 0) + 1
            print(f"[DEBUG] _generate_quiz_incremental: generated {len(new_questions)} questions, attempts={new_attempts}")
            return {
                **state,
                "quiz_questions": new_questions,
                "validated_questions": validated_questions,
                "generation_attempts": new_attempts
            }
        except Exception as e:
            new_attempts = state.get("generation_attempts", 0) + 1
            print(f"[DEBUG] _generate_quiz_incremental: exception {e}, attempts={new_attempts}")
            return {
                **state, 
                "quiz_questions": [],
                "validated_questions": state.get("validated_questions", []),
                "generation_attempts": new_attempts,
                "error": f"문제 생성 중 오류 발생: {e}"
            }

    def _validate_quiz_incremental(self, state) -> dict:
        subject_area = state.get("subject_area", "")
        previously_validated = state.get("validated_questions", [])
        new_questions = state.get("quiz_questions", [])
        context = state.get("context", "")
        target_quiz_count = state.get("target_quiz_count", 5)
        generation_attempts = state.get("generation_attempts", 0)
        error = state.get("error", "")

        print(f"[DEBUG] _validate_quiz_incremental: subject={subject_area}, new_questions={len(new_questions)}, previously_validated={len(previously_validated)}, error={error}")

        # 에러가 있으면 검증하지 않고 에러 상태 유지
        if error:
            print(f"[DEBUG] _validate_quiz_incremental: skipping validation due to error: {error}")
            return state

        if not new_questions:
            print(f"[DEBUG] _validate_quiz_incremental: no new questions to validate")
            return {**state, "validated_questions": previously_validated}

        newly_validated = []
        print(f"[DEBUG] _validate_quiz_incremental: validating {len(new_questions)} questions")

        # LLM 기반 검증 (한 번에 모든 문제 검증)
        print(f"[DEBUG] _validate_quiz_incremental: validating all {len(new_questions)} questions at once")
        
        # LLM 기반 검증
        validated_batch = self._llm_validate_questions_batch(new_questions, subject_area)
        newly_validated.extend(validated_batch)

        all_validated = previously_validated + newly_validated
        print(f"[DEBUG] _validate_quiz_incremental: total validated: {len(all_validated)}/{target_quiz_count}")

        return {
            **state,
            "validated_questions": all_validated,
            "quiz_questions": all_validated,
            "error": ""  # 에러 상태 초기화
        }

    def _llm_validate_questions_batch(self, questions: List[Dict[str, Any]], subject_area: str) -> List[Dict[str, Any]]:
        """LLM 기반 문제 배치 검증 (FEVER 형식)"""
        if not questions:
            return []
        
        # 검증 프롬프트 생성
        validation_prompt = self._create_validation_prompt(questions, subject_area)
        
        try:
            # LLM 호출
            response = self.llm.invoke(validation_prompt)
            validation_result = self._parse_validation_response(response.content)
            
            # 검증 결과에 따라 문제 필터링
            validated_questions = []
            for i, question in enumerate(questions):
                if i < len(validation_result) and validation_result[i] == True:
                    # 보기 수 검증 추가
                    if self._validate_options_count(question):
                        validated_questions.append(question)
                        print(f"[DEBUG] LLM 검증 통과: {question.get('question', '')[:50]}...")
                    else:
                        print(f"[DEBUG] 보기 수 검증 실패: {question.get('question', '')[:50]}...")
                else:
                    print(f"[DEBUG] LLM 검증 실패: {question.get('question', '')[:50]}...")
            
            return validated_questions
            
        except Exception as e:
            print(f"[ERROR] LLM 검증 실패: {e}")
            # 에러 시 기본 검증으로 폴백
            return self._basic_validate_questions(questions)

    def _create_validation_prompt(self, questions: List[Dict[str, Any]], subject_area: str) -> str:
        """간단한 검증 프롬프트 생성"""
        questions_text = ""
        for i, q in enumerate(questions, 1):
            questions_text += f"{i}. {q.get('question', '')}\n"
            for j, option in enumerate(q.get('options', []), 1):
                questions_text += f"   {j}) {option}\n"
            questions_text += f"   정답: {q.get('answer', '')}\n\n"
        
        prompt = f"""
다음 {subject_area} 과목의 {len(questions)}개 문제들을 검증해주세요.

{questions_text}

검증 기준:
1. 질문이 명확하고 이해하기 쉬운가?
2. 보기가 4개인가? (정답 1개, 오답 3개)
3. 정답이 명확하고 논리적인가?
4. 해설이 충분하고 정확한가?
5. {subject_area} 과목과 관련성이 있는가?
6. 중복되거나 유사한 문제가 아닌가?

각 문제에 대해 유효하면 "VALID", 무효하면 "INVALID"로 판단해주세요.

응답 형식:
문제1: VALID/INVALID
문제2: VALID/INVALID
...
문제{len(questions)}: VALID/INVALID
"""
        return prompt

    def _parse_validation_response(self, response: str) -> List[Dict[str, Any]]:
        """검증 응답 파싱"""
        try:
            print(f"[DEBUG] _parse_validation_response: response='{response}'")
            
            # 간단한 텍스트 파싱
            lines = response.strip().split('\n')
            valid_questions = []
            
            for line in lines:
                line = line.strip()
                if ':' in line and ('VALID' in line or 'INVALID' in line):
                    if 'VALID' in line:
                        valid_questions.append(True)
                    else:
                        valid_questions.append(False)
            
            print(f"[DEBUG] _parse_validation_response: found {len(valid_questions)} validation results")
            return valid_questions
            
        except Exception as e:
            print(f"[ERROR] 검증 응답 파싱 실패: {e}")
            return []

    def _validate_options_count(self, question: Dict[str, Any]) -> bool:
        """보기 수 검증 (4개 필수)"""
        options = question.get("options", [])
        return len(options) == 4

    def _basic_validate_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """기본 검증 (LLM 실패 시 폴백)"""
        validated = []
        for q in questions:
            if (q.get("question") and q.get("options") and q.get("answer") and 
                q.get("explanation") and self._validate_options_count(q)):
                validated.append(q)
        return validated

    def _check_completion(self, state) -> str:
        validated_count = len(state.get("validated_questions", []))
        target_count = state.get("target_quiz_count", 5)
        generation_attempts = state.get("generation_attempts", 0)
        error = state.get("error", "")
        
        print(f"[DEBUG] _check_completion: validated={validated_count}, target={target_count}, attempts={generation_attempts}, error={error}")
        
        # 목표 달성
        if validated_count >= target_count:
            print(f"[DEBUG] Target reached ({validated_count}/{target_count}), completing")
            return "complete"
        
        # 최대 시도 횟수 도달 (20문제씩 생성하므로 3회로 제한)
        if generation_attempts >= 3:
            print(f"[DEBUG] Max attempts reached ({generation_attempts}), completing with {validated_count} questions")
            return "complete"
        
        # 에러가 있으면 중단
        if error:
            print(f"[DEBUG] Error detected: {error}, completing")
            return "complete"
        
        # 부족한 문제 수만큼 재생성 (최대 20문제)
        remaining = target_count - validated_count
        needed = min(remaining, 20)  # 한 번에 최대 20문제 생성
        
        print(f"[DEBUG] Need {needed} more questions ({validated_count}/{target_count}), continuing generation (attempt {generation_attempts + 1})")
        return "generate_more"

    def _parse_quiz_response(self, response: str, subject_area: str = "") -> List[Dict[str, Any]]:
        try:
            print(f"[DEBUG] _parse_quiz_response: raw response length={len(response)}")
            print(f"[DEBUG] _parse_quiz_response: response preview='{response[:200]}...'")
            
            # 1. JSON 블록 찾기 (```json ... ```)
            json_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1)
                print(f"[DEBUG] _parse_quiz_response: found JSON block, length={len(json_str)}")
            else:
                # 2. 일반 JSON 객체 찾기
                json_str_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response.strip(), re.DOTALL)
                if not json_str_match:
                    print(f"[DEBUG] _parse_quiz_response: no JSON found in response")
                    return []
                json_str = json_str_match.group(0)
                print(f"[DEBUG] _parse_quiz_response: found JSON object, length={len(json_str)}")

            # 3. JSON 문자열 정리 (과도한 백슬래시 제거는 JSON을 깨뜨릴 수 있으므로 최소화)
            json_str = json_str.replace('\\u312f', '').replace('\\n', ' ')
            print(f"[DEBUG] _parse_quiz_response: cleaned JSON='{json_str[:200]}...'")
            
            # 4. JSON 파싱
            data = json.loads(json_str)
            
            # questions 배열이 있는 경우와 단일 문제 객체인 경우 모두 처리
            if "questions" in data and isinstance(data["questions"], list):
                questions = data["questions"]
                print(f"[DEBUG] _parse_quiz_response: found {len(questions)} questions in array")
            elif isinstance(data, dict) and "question" in data:
                # 단일 문제 객체인 경우 배열로 변환
                questions = [data]
                print(f"[DEBUG] _parse_quiz_response: found single question object")
            else:
                print(f"[DEBUG] _parse_quiz_response: invalid data structure, keys={list(data.keys()) if isinstance(data, dict) else 'not dict'}")
                return []
            
            # 5. 각 문제 처리 및 정규화(보기에 번호 제거, 과목 주입, 보기 정리, 정답 보정)
            print(f"[DEBUG] _parse_quiz_response: processing {len(questions)} questions")
            
            processed_questions = []
            for i, question in enumerate(questions):
                print(f"[DEBUG] _parse_quiz_response: processing question {i+1}: {question}")
                
                # 필수 필드 확인
                if not question.get("question") or not question.get("options"):
                    print(f"[DEBUG] _parse_quiz_response: skipping invalid question {i+1}")
                    continue
                
                if "options" in question and isinstance(question["options"], list):
                    numbered_options = []
                    for j, option_text in enumerate(question["options"], 1):
                        cleaned_text = re.sub(r'^\s*\d+\.\s*', '', str(option_text)).strip()
                        numbered_options.append(f"  {j}. {cleaned_text}")
                    question["options"] = numbered_options
                
                if "subject" not in question:
                    question["subject"] = subject_area
                
                # 보기 길이/중복 필터링 및 4개 제한
                dedup_opts, seen = [], set()
                for opt in question["options"]:
                    base = re.sub(r'^\s*\d+\.\s*', '', opt).strip()
                    if not base or base.lower() in seen:
                        continue
                    seen.add(base.lower())
                    dedup_opts.append(opt)
                if len(dedup_opts) >= 4:
                    question["options"] = dedup_opts[:4]
                
                # 정답 인덱스 유효성 보정
                ans = str(question.get("answer", "")).strip()
                if ans not in {"1","2","3","4"}:
                    question["answer"] = "1"
                
                processed_questions.append(question)
                print(f"[DEBUG] _parse_quiz_response: processed question {i+1}: {question.get('question', '')[:50]}...")
            
            print(f"[DEBUG] _parse_quiz_response: returning {len(processed_questions)} processed questions")
            return processed_questions
        except Exception as e:
            print(f"[DEBUG] _parse_quiz_response: exception during parsing: {e}")
            print(f"[DEBUG] _parse_quiz_response: response that caused error: '{response[:500]}...'")
            return []

    # ---------- 중복 탐지/제거 유틸 ----------
    def _norm_text(self, text: str) -> str:
        try:
            s = re.sub(r"\s+", " ", str(text or "")).strip().lower()
            s = re.sub(r"^[0-9]+\.\s*", "", s)
            return s
        except Exception:
            return str(text or "").strip().lower()

    def _jaccard_sim(self, a: str, b: str) -> float:
        ta = set(self._norm_text(a).split())
        tb = set(self._norm_text(b).split())
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / max(union, 1)

    def _filter_duplicate_questions(self, new_qs: List[Dict[str, Any]], prev_validated: List[Dict[str, Any]], subject_area: str) -> List[Dict[str, Any]]:
        if not new_qs:
            return []
        
        print(f"[DEBUG] _filter_duplicate_questions: filtering {len(new_qs)} new questions against {len(prev_validated)} previous")
        
        kept: List[Dict[str, Any]] = []
        seen_norm: set = set()
        prev_texts = [q.get("question", "") for q in (prev_validated or [])]
        
        for i, q in enumerate(new_qs):
            qtext = q.get("question", "")
            norm = self._norm_text(qtext)
            
            print(f"[DEBUG] _filter_duplicate_questions: checking question {i+1}: '{qtext[:50]}...'")
            
            if not norm:
                print(f"[DEBUG] _filter_duplicate_questions: skipping empty question {i+1}")
                continue
                
            if norm in seen_norm:
                print(f"[DEBUG] _filter_duplicate_questions: skipping duplicate in current batch {i+1}")
                continue
                
            # 이전 문제와의 중복 검사 (더 관대한 기준)
            max_similarity = 0.0
            for p in prev_texts:
                sim = self._jaccard_sim(qtext, p)
                max_similarity = max(max_similarity, sim)
            
            if max_similarity >= 0.98:  # 0.95 -> 0.98로 더욱 관대하게 (거의 완전히 같은 경우만)
                print(f"[DEBUG] _filter_duplicate_questions: skipping similar to previous {i+1} (similarity: {max_similarity:.3f})")
                continue
            else:
                print(f"[DEBUG] _filter_duplicate_questions: similarity check passed {i+1} (max: {max_similarity:.3f})")
                
            kept.append(q)
            seen_norm.add(norm)
            print(f"[DEBUG] _filter_duplicate_questions: keeping question {i+1}")
        
        print(f"[DEBUG] _filter_duplicate_questions: returning {len(kept)} questions")
        return kept

    # ---------- 핵심: 그래프 구성 변경 (과목별 2노드 × 5과목 = 10노드) ----------
    def _build_graph(self):
        """
        공통 사전 단계: retrieve -> prepare_context
        이후 과목별 라우팅: (subject)generate -> (subject)validate -> 조건부 루프
        """
        workflow = StateGraph(GraphState)

        # 공통 전처리
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("prepare_context", self._prepare_context)

        # 과목별 노드 함수: subject를 클로저로 묶어 1개 노드 생성 (순차 처리)
        def make_generate_node(subject_name):
            def _gen(state: GraphState) -> GraphState:
                print(f"[DEBUG] {subject_name}_generate 노드 실행")
                # 독립적인 state 생성 (충돌 방지)
                independent_state = {
                    "query": state.get("query", ""),
                    "documents": state.get("documents", []),
                    "context": state.get("context", ""),
                    "quiz_questions": [],
                    "difficulty": state.get("difficulty", "중급"),
                    "error": "",
                    "used_sources": state.get("used_sources", []),
                    "generation_attempts": 0,
                    "target_quiz_count": state.get("target_quiz_count", 5),
                    "subject_area": subject_name,
                    "validated_questions": [],
                    "node_id": 1
                }
                result = self._generate_quiz_incremental(independent_state)
                # 결과를 원본 state에 병합 (충돌 방지)
                return {
                    **state,
                    f"{subject_name}_quiz_questions": result.get("quiz_questions", []),
                    f"{subject_name}_validated_questions": result.get("validated_questions", []),
                    f"{subject_name}_error": result.get("error", "")
                }
            return _gen

        def make_validate_node(subject_name):
            def _val(state: GraphState) -> GraphState:
                print(f"[DEBUG] {subject_name}_validate 노드 실행")
                # 독립적인 state 생성
                independent_state = {
                    "query": state.get("query", ""),
                    "documents": state.get("documents", []),
                    "context": state.get("context", ""),
                    "quiz_questions": state.get(f"{subject_name}_quiz_questions", []),
                    "difficulty": state.get("difficulty", "중급"),
                    "error": "",
                    "used_sources": state.get("used_sources", []),
                    "generation_attempts": 0,
                    "target_quiz_count": state.get("target_quiz_count", 5),
                    "subject_area": subject_name,
                    "validated_questions": [],
                    "node_id": 1
                }
                result = self._validate_quiz_incremental(independent_state)
                # 결과를 원본 state에 병합
                return {
                    **state,
                    f"{subject_name}_validated_questions": result.get("validated_questions", []),
                    f"{subject_name}_error": result.get("error", "")
                }
            return _val

        # 과목별 노드 추가 (각 과목당 1개 노드, 순차 처리)
        subject_to_nodes = {}
        for subj in self.SUBJECT_AREAS.keys():
            gen_name = f"{subj}_generate"
            val_name = f"{subj}_validate"
            workflow.add_node(gen_name, make_generate_node(subj))
            workflow.add_node(val_name, make_validate_node(subj))
            # 과목별 내부 엣지 (순차 처리)
            workflow.add_edge(gen_name, val_name)
            workflow.add_conditional_edges(
                val_name,
                self._check_completion,
                {"generate_more": gen_name, "complete": "merge_results"}
            )
            subject_to_nodes[subj] = (gen_name, val_name)

        # 라우터: prepare_context 이후 과목별 generate로 분기 (단일 노드)
        def _route_to_subject(state: GraphState) -> str:
            subj = state.get("subject_area", "")
            print(f"[DEBUG] _route_to_subject: subject_area='{subj}', available_subjects={list(subject_to_nodes.keys())}")
            if subj in subject_to_nodes:
                gen_name, val_name = subject_to_nodes[subj]
                print(f"[DEBUG] Found subject '{subj}', returning node: {gen_name}")
                return gen_name
            # 기본값(안 맞으면 설계로)
            print(f"[DEBUG] Subject '{subj}' not found, using default: 소프트웨어설계")
            gen_name, val_name = subject_to_nodes["소프트웨어설계"]
            return gen_name

        # 결과 합치기 노드 추가
        workflow.add_node("merge_results", self._merge_results)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "prepare_context")
        
        # 순차 처리: prepare_context 이후 과목별 노드로 분기
        workflow.add_conditional_edges(
            "prepare_context",
            _route_to_subject,
            {f"{subj}_generate": f"{subj}_generate" 
             for subj in self.SUBJECT_AREAS.keys()}
        )
        
        # 모든 노드에서 merge_results로 수렴
        for subj in self.SUBJECT_AREAS.keys():
            workflow.add_edge(f"{subj}_validate", "merge_results")
        
        # merge_results에서 종료
        workflow.add_edge("merge_results", END)

        self.workflow = workflow.compile()
    
    def _build_independent_workflow(self):
        """과목별 독립적인 워크플로우 생성 (충돌 방지)"""
        workflow = StateGraph(SubjectState)
        
        # 공통 전처리
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("prepare_context", self._prepare_context)
        
        # 과목별 노드 함수 (독립적인 처리)
        def make_generate_node():
            def _gen(state: SubjectState) -> SubjectState:
                print(f"[DEBUG] {state['subject_area']}_generate 노드 실행")
                result = self._generate_quiz_incremental(state)
                return result
            return _gen

        def make_validate_node():
            def _val(state: SubjectState) -> SubjectState:
                print(f"[DEBUG] {state['subject_area']}_validate 노드 실행")
                result = self._validate_quiz_incremental(state)
                return result
            return _val

        # 노드 추가
        workflow.add_node("generate", make_generate_node())
        workflow.add_node("validate", make_validate_node())
        
        # 워크플로우 연결
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "prepare_context")
        workflow.add_edge("prepare_context", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_conditional_edges(
            "validate",
            self._check_completion,
            {"generate_more": "generate", "complete": END}
        )
        
        return workflow.compile()

    def _merge_results(self, state: GraphState) -> GraphState:
        """결과 합치기 (과목별 순차 처리)"""
        print(f"[DEBUG] _merge_results: 결과 합치기 시작")
        
        # 과목별 결과를 수집
        all_validated_questions = []
        subject_area = state.get("subject_area", "")
        
        # 현재 과목의 결과 수집
        subject_key = f"{subject_area}_validated_questions"
        subject_questions = state.get(subject_key, [])
        if subject_questions:
            all_validated_questions.extend(subject_questions)
            print(f"[DEBUG] _merge_results: {subject_area}에서 {len(subject_questions)}개 문제 수집")
        
        print(f"[DEBUG] _merge_results: 총 {len(all_validated_questions)}개 문제 수집")
        
        # 중복 제거 (동일한 질문 내용)
        unique_questions = []
        seen_questions = set()
        
        for question in all_validated_questions:
            question_text = question.get("question", "").strip()
            if question_text and question_text not in seen_questions:
                unique_questions.append(question)
                seen_questions.add(question_text)
        
        print(f"[DEBUG] _merge_results: 중복 제거 후 {len(unique_questions)}개 문제")
        
        return {
            **state,
            "validated_questions": unique_questions,
            "quiz_questions": unique_questions,
            "error": ""
        }
    # --------------------------------------------------------------------

    # 단일 과목 생성 (독립적인 워크플로우 사용)
    def _generate_subject_quiz(self, subject_area: str, target_count: int = 5, difficulty: str = "중급", milvus_data: Dict[str, Any] = None) -> Dict[str, Any]:
        # MilvusDB 연결 정보를 전역 변수로 저장
        self._current_milvus_data = milvus_data
        
        # MilvusDB 연결 정보 확인
        if not milvus_data or not milvus_data.get("connection_status", False):
            print("⚠️ MilvusDB 연결 안됨 - 컨텍스트 없이 문제 생성")
        
        if subject_area not in self.SUBJECT_AREAS:
            return {"error": f"유효하지 않은 과목: {subject_area}"}
        
        # 독립적인 워크플로우 인스턴스 생성 (충돌 방지)
        independent_workflow = self._build_independent_workflow()
        
        # 초기 상태 설정 (SubjectState 형식)
        initial_state = {
            "query": f"{subject_area} {difficulty} 문제",
            "documents": [],
            "context": "",
            "quiz_questions": [],
                    "difficulty": difficulty,
            "error": "",
            "used_sources": [],
                    "generation_attempts": 0,
            "target_quiz_count": target_count,
                    "subject_area": subject_area,
            "validated_questions": [],
            "node_id": 1
        }
        
        try:
            # 독립적인 워크플로우 실행
            result = independent_workflow.invoke(initial_state)
            
            # 결과 반환
            validated_questions = result.get("validated_questions", [])
            return {
                "success": True,
                "questions": validated_questions,  # 이 키가 중요!
                "status": "SUCCESS" if len(validated_questions) > 0 else "FAILED",
                "result": {
                    "exam_title": f"{subject_area} {difficulty} 문제집",
                    "total_questions": len(validated_questions),
                    "difficulty": difficulty,
                    "subjects": {
                        subject_area: {
                            "requested_count": target_count,
                            "actual_count": len(validated_questions),
                            "questions": validated_questions,
                            "status": "SUCCESS" if len(validated_questions) > 0 else "FAILED"
                        }
                    },
                    "all_questions": validated_questions,
                    "generation_summary": {
                        "target_total": target_count,
                        "actual_total": len(result.get("validated_questions", [])),
                        "success_rate": f"{(len(result.get('validated_questions', [])) / target_count * 100):.1f}%",
                        "successful_subjects": 1 if len(result.get("validated_questions", [])) > 0 else 0,
                        "failed_subjects": 0 if len(result.get("validated_questions", [])) > 0 else 1,
                        "completion_status": "COMPLETE" if len(result.get("validated_questions", [])) >= target_count else "PARTIAL",
                        "generation_time": "0.0초"
                    },
                    "failed_subjects": [],
                    "model_info": OPENAI_LLM_MODEL,
                    "parallel_agents": 1
                }
            }
            
        except Exception as e:
            err = str(e)
            print(f"❌ 문제 생성 실패: {err}")
            return {
                "success": False,
                "questions": [],  # 빈 리스트 반환
                "status": "FAILED",
                "error": err,
                "result": {
                    "exam_title": f"{subject_area} {difficulty} 문제집",
                    "total_questions": 0,
                    "difficulty": difficulty,
                    "subjects": {
                        subject_area: {
                            "requested_count": target_count,
                            "actual_count": 0,
                            "questions": [],
                            "status": "FAILED"
                        }
                    },
                    "all_questions": [],
                    "generation_summary": {
                        "target_total": target_count,
                        "actual_total": 0,
                        "success_rate": "0.0%",
                        "successful_subjects": 0,
                        "failed_subjects": 1,
                        "completion_status": "FAILED",
                        "generation_time": "0.0초"
                    },
                    "failed_subjects": [{"subject": subject_area, "error": str(e)}],
                    "model_info": OPENAI_LLM_MODEL,
                    "parallel_agents": 1
                }
        }

    # 3) 사용자 지정 병렬 실행로 5과목 동시 처리(최대 parallel_agents 동시)
    def _generate_full_exam(self, difficulty: str = "중급", parallel_agents: int = 2, milvus_data: Dict[str, Any] = None) -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        start_time = time.time()

        requested_per_subject = {s: info["count"] for s, info in self.SUBJECT_AREAS.items()}

        full_exam_result = {
            "exam_title": "정보처리기사 모의고사",
            "total_questions": 0,
            "difficulty": difficulty,
            "subjects": {},
            "all_questions": [],
            "generation_summary": {},
            "failed_subjects": [],
            "model_info": "Groq llama-4-scout-17b-16e-instruct",
            "parallel_agents": parallel_agents
        }

        # 병렬로 과목 생성 실행
        futures = {}
        with ThreadPoolExecutor(max_workers=parallel_agents) as ex:
            for subject_area, target in requested_per_subject.items():
                futures[ex.submit(
                    self._generate_subject_quiz,
                    subject_area=subject_area,
                    target_count=target,
                    difficulty=difficulty,
                    milvus_data=milvus_data
                )] = subject_area

            per_subject_results = {}
            for fut in as_completed(futures):
                subject_area = futures[fut]
                try:
                    per_subject_results[subject_area] = fut.result()
                except Exception as e:
                    per_subject_results[subject_area] = {"error": str(e)}

        # 4) 머지 순서에 따라 취합
        total_generated = 0
        merged_questions = []
        for subject_area in self.MERGE_ORDER:
            res = per_subject_results.get(subject_area, {"error": "결과 없음"})
            if "error" in res:
                full_exam_result["failed_subjects"].append({
                    "subject": subject_area,
                    "error": res["error"]
                })
                full_exam_result["subjects"][subject_area] = {
                    "requested_count": requested_per_subject[subject_area],
                    "actual_count": 0,
                    "questions": [],
                    "status": "FAILED"
                }
            else:
                qs = res.get("questions", [])
                total_generated += len(qs)
                merged_questions.extend(qs)
                full_exam_result["subjects"][subject_area] = {
                    "requested_count": requested_per_subject[subject_area],
                    "actual_count": len(qs),
                    "questions": qs,
                    "status": res.get("status", "UNKNOWN")
                }

        elapsed_time = time.time() - start_time
        full_exam_result["total_questions"] = total_generated
        full_exam_result["all_questions"] = merged_questions
        full_exam_result["generation_summary"] = {
            "target_total": sum(requested_per_subject.values()),  # 100
            "actual_total": total_generated,
            "success_rate": f"{(total_generated / max(1, sum(requested_per_subject.values())))*100:.1f}%",
            "successful_subjects": 5 - len(full_exam_result["failed_subjects"]),
            "failed_subjects": len(full_exam_result["failed_subjects"]),
            "completion_status": "COMPLETE" if total_generated >= sum(requested_per_subject.values()) else "PARTIAL",
            "generation_time": f"{elapsed_time:.1f}초"
        }
        return full_exam_result

    def _generate_partial_exam(self, selected_subjects: List[str], questions_per_subject: int = 10, 
                              difficulty: str = "중급", parallel_agents: int = 2, milvus_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """선택된 과목들에 대해 지정된 문제 수만큼 생성"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        start_time = time.time()

        # 워크플로우 다이어그램 생성 제거

        partial_exam_result = {
            "exam_title": f"정보처리기사 선택과목 모의고사 ({len(selected_subjects)}과목)",
            "total_questions": 0,
            "difficulty": difficulty,
            "selected_subjects": selected_subjects,
            "questions_per_subject": questions_per_subject,
            "subjects": {},
            "all_questions": [],
            "generation_summary": {},
            "failed_subjects": [],
            "model_info": "Groq llama-4-scout-17b-16e-instruct",
            "parallel_agents": parallel_agents
        }

        # 병렬로 선택된 과목 생성 실행
        futures = {}
        with ThreadPoolExecutor(max_workers=parallel_agents) as ex:
            for subject_area in selected_subjects:
                futures[ex.submit(
                    self._generate_subject_quiz,
                    subject_area=subject_area,
                    target_count=questions_per_subject,
                    difficulty=difficulty,
                    milvus_data=milvus_data
                )] = subject_area

            per_subject_results = {}
            for fut in as_completed(futures):
                subject_area = futures[fut]
                try:
                    per_subject_results[subject_area] = fut.result()
                except Exception as e:
                    per_subject_results[subject_area] = {"error": str(e)}

        # 결과 취합
        total_generated = 0
        merged_questions = []
        for subject_area in selected_subjects:
            res = per_subject_results.get(subject_area, {"error": "결과 없음"})
            if "error" in res:
                partial_exam_result["failed_subjects"].append({
                    "subject": subject_area,
                    "error": res["error"]
                })
                partial_exam_result["subjects"][subject_area] = {
                    "requested_count": questions_per_subject,
                    "actual_count": 0,
                    "questions": [],
                    "status": "FAILED"
                }
            else:
                qs = res.get("questions", [])
                total_generated += len(qs)
                merged_questions.extend(qs)
                partial_exam_result["subjects"][subject_area] = {
                    "requested_count": questions_per_subject,
                    "actual_count": len(qs),
                    "questions": qs,
                    "status": res.get("status", "UNKNOWN")
                }

        elapsed_time = time.time() - start_time
        partial_exam_result["total_questions"] = total_generated
        partial_exam_result["all_questions"] = merged_questions
        partial_exam_result["generation_summary"] = {
            "target_total": len(selected_subjects) * questions_per_subject,
            "actual_total": total_generated,
            "success_rate": f"{(total_generated / max(1, len(selected_subjects) * questions_per_subject))*100:.1f}%",
            "successful_subjects": len(selected_subjects) - len(partial_exam_result["failed_subjects"]),
            "failed_subjects": len(partial_exam_result["failed_subjects"]),
            "completion_status": "COMPLETE" if total_generated >= len(selected_subjects) * questions_per_subject else "PARTIAL",
            "generation_time": f"{elapsed_time:.1f}초"
        }
        return partial_exam_result

    # 파일 저장 함수는 기존과 동일(중복 정의는 마지막 정의가 유효)
    def _save_to_json(self, exam_result: Dict[str, Any], filename: str = None) -> str:
        save_dir = "C:\\LLM-T\\teacher\\TestGenerator\\test"
        os.makedirs(save_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if "exam_title" in exam_result:
                filename = f"정보처리기사_모의고사_100문제_{timestamp}.json"
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

    def _save_to_json(self, exam_result: Dict[str, Any], filename: str = None) -> str:
        """시험 결과를 JSON 파일로 저장"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"정보처리기사_문제생성_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), "test", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(exam_result, f, ensure_ascii=False, indent=2)
        return filename

# 워크플로우 다이어그램 함수 제거 (사용하지 않음)
