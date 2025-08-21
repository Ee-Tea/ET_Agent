import os
import glob
from typing import List, Dict, Any, TypedDict
from abc import ABC, abstractmethod
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langgraph.graph import StateGraph, END
import json
import re
from collections import Counter
import time
from datetime import datetime
from pathlib import Path

# .env 파일 로드를 위한 임포트
from dotenv import load_dotenv
import sys
import os

# 상대 임포트 대신 절대 경로로 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_agent import BaseAgent

# OpenAI 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# .env 파일 로드
load_dotenv()

class GraphState(TypedDict):
    """그래프 상태 정의"""
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
    validated_questions: List[Dict[str, Any]]  # 문제에 답 해설까지 한 번에 나옴, 보기는 1. 2. 3. 4. 으로 번호가 붙음, 문제에는 번호 안 붙음


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
        self.vectorstore = None
        self.retriever = None
        self.workflow = None

        self.files_in_vectorstore = []

        self._initialize_models()
        self._build_graph()  # 2) 과목별 2노드(생성/검증) 구축

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
        """
        try:
            mode = input_data.get("mode", "full_exam")
            difficulty = input_data.get("difficulty", "중급")
            save_to_file = input_data.get("save_to_file", False)
            filename = input_data.get("filename")
            parallel_agents = max(1, int(input_data.get("parallel_agents", 2)))  # 3) 병렬 개수

            if not self._build_vectorstore_from_all_pdfs():
                return {
                    "success": False,
                    "error": f"'{self.data_folder}' 폴더에 PDF 파일이 없습니다."
                }

            if mode == "full_exam":
                # 1) 5과목 × 20문항 = 총 100문항
                result = self._generate_full_exam(difficulty=difficulty,
                                                  parallel_agents=parallel_agents)
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
                    parallel_agents=parallel_agents
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
                    difficulty=difficulty
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

    def _build_vectorstore_from_all_pdfs(self) -> bool:
        pdf_files = self.get_pdf_files()
        if not pdf_files:
            return False
        if self.vectorstore and set(self.files_in_vectorstore) == set(pdf_files):
            return True

        all_documents = []
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(pdf_path)
                all_documents.extend(documents)
            except Exception:
                continue

        if not all_documents:
            return False

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(all_documents)
        self.vectorstore = FAISS.from_documents(splits, self.embeddings_model)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        self.files_in_vectorstore = pdf_files
        return True

    def get_pdf_files(self) -> List[str]:
        return glob.glob(os.path.join(self.data_folder, "*.pdf"))

    # ---- 공통 노드 구현(그대로 재사용) ----
    def _retrieve_documents(self, state: GraphState) -> GraphState:
        try:
            query = state["query"]
            subject_area = state.get("subject_area", "")
            enhanced_query = f"{subject_area} {query}".strip()
            print(f"[DEBUG] _retrieve_documents: query='{query}', subject_area='{subject_area}', enhanced_query='{enhanced_query}'")
            documents = self.retriever.invoke(enhanced_query)
            print(f"[DEBUG] _retrieve_documents: found {len(documents)} documents")
            source_files = [doc.metadata.get('source_file', 'Unknown') for doc in documents]
            used_sources = list(Counter(source_files).keys())
            return {**state, "documents": documents, "used_sources": used_sources}
        except Exception as e:
            print(f"[DEBUG] _retrieve_documents: error {e}")
            return {**state, "error": f"문서 검색 오류: {e}"}

    def _prepare_context(self, state: GraphState) -> GraphState:
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

    def _generate_quiz_incremental(self, state: GraphState) -> GraphState:
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
                return {
                    **state, 
                    "quiz_questions": [],
                    "validated_questions": validated_questions,
                    "generation_attempts": new_attempts,
                    "error": "검색된 문서 내용이 없습니다."
                }

            generate_count = max(min(needed_count, 10), 1)

            prompt_template = PromptTemplate(
                input_variables=["context", "subject_area", "needed_count"],
                template=(
                    "당신은 정보처리기사 출제 전문가입니다. 아래 문서 내용을 바탕으로 {subject_area} 과목의 객관식 문제 {needed_count}개를 생성하세요.\n\n"
                    "**중요: 반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트나 설명은 포함하지 마세요.**\n\n"
                    "[문서 내용]\n{context}\n\n"
                    "[응답 형식]\n"
                    "{{\n"
                    "  \"questions\": [\n"
                    "    {{\n"
                    "      \"question\": \"문제 내용을 여기에 작성\",\n"
                    "      \"options\": [\"선택지1\", \"선택지2\", \"선택지3\", \"선택지4\"],\n"
                    "      \"answer\": \"1\",선택지의 숫자만 출력해야 함\n"
                    "      \"explanation\": \"정답에 대한 간단한 해설\"\n"
                    "    }}\n"
                    "  ]\n"
                    "}}\n\n"
                    "**응답은 위 JSON 형식만 출력하세요. 다른 텍스트는 절대 포함하지 마세요.**"
                )
            )

            prompt = prompt_template.format(
                context=context, subject_area=subject_area, needed_count=generate_count
            )

            print(f"[DEBUG] _generate_quiz_incremental: calling LLM for {generate_count} questions")
            self.llm.temperature = 0.2
            self.llm.max_tokens = 1024
            response = self.llm.invoke(prompt)
            response_content = getattr(response, "content", str(response))
            print(f"[DEBUG] _generate_quiz_incremental: LLM response length={len(response_content)}")
            
            new_questions = self._parse_quiz_response(response_content, subject_area)
            print(f"[DEBUG] _generate_quiz_incremental: parsed {len(new_questions)} questions")

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

    def _validate_quiz_incremental(self, state: GraphState) -> GraphState:
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

        # 간단한 검증: 모든 문제를 유효하다고 가정 (LLM 호출 없이)
        # 실제로는 LLM 검증을 할 수 있지만, 테스트를 위해 간단하게 처리
        for q in new_questions:
            if len(newly_validated) >= target_quiz_count - len(previously_validated):
                break
            # 기본 검증: 필수 필드가 있는지 확인
            if q.get("question") and q.get("options") and q.get("answer") and q.get("explanation"):
                newly_validated.append(q)
                print(f"[DEBUG] _validate_quiz_incremental: validated question: {q.get('question', '')[:50]}...")

        all_validated = previously_validated + newly_validated
        print(f"[DEBUG] _validate_quiz_incremental: total validated: {len(all_validated)}/{target_quiz_count}")

        return {
            **state,
            "validated_questions": all_validated,
            "quiz_questions": all_validated,
            "error": ""  # 에러 상태 초기화
        }

    def _check_completion(self, state: GraphState) -> str:
        validated_count = len(state.get("validated_questions", []))
        target_count = state.get("target_quiz_count", 5)
        generation_attempts = state.get("generation_attempts", 0)
        error = state.get("error", "")
        
        print(f"[DEBUG] _check_completion: validated={validated_count}, target={target_count}, attempts={generation_attempts}, error={error}")
        
        # 목표 달성
        if validated_count >= target_count:
            print(f"[DEBUG] Target reached ({validated_count}/{target_count}), completing")
            return "complete"
        
        # 최대 시도 횟수 도달
        if generation_attempts >= 5:  # 5회로 증가
            print(f"[DEBUG] Max attempts reached ({generation_attempts}), completing")
            return "complete"
        
        # 에러가 있으면 중단
        if error:
            print(f"[DEBUG] Error detected: {error}, completing")
            return "complete"
        
        # 계속 생성
        print(f"[DEBUG] Need more questions ({validated_count}/{target_count}), continuing generation (attempt {generation_attempts})")
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

            # 3. JSON 문자열 정리
            json_str = json_str.replace('\\u312f', '').replace('\\n', ' ').replace('\\', '')
            print(f"[DEBUG] _parse_quiz_response: cleaned JSON='{json_str[:200]}...'")
            
            # 4. JSON 파싱
            data = json.loads(json_str)
            if "questions" not in data or not isinstance(data["questions"], list):
                print(f"[DEBUG] _parse_quiz_response: invalid data structure, keys={list(data.keys()) if isinstance(data, dict) else 'not dict'}")
                return []

            questions = data.get("questions", [])
            print(f"[DEBUG] _parse_quiz_response: found {len(questions)} questions")
            
            # 5. 각 문제 처리
            for i, question in enumerate(questions):
                if "options" in question and isinstance(question["options"], list):
                    numbered_options = []
                    for j, option_text in enumerate(question["options"], 1):
                        cleaned_text = re.sub(r'^\s*\d+\.\s*', '', option_text).strip()
                        numbered_options.append(f"  {j}. {cleaned_text}")
                    question["options"] = numbered_options
                if "subject" not in question:
                    question["subject"] = subject_area
                print(f"[DEBUG] _parse_quiz_response: processed question {i+1}: {question.get('question', '')[:50]}...")
            
            return questions
        except Exception as e:
            print(f"[DEBUG] _parse_quiz_response: exception during parsing: {e}")
            print(f"[DEBUG] _parse_quiz_response: response that caused error: '{response[:500]}...'")
            return []

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

        # 과목별 노드 함수: subject를 클로저로 묶어 2개 노드 생성
        def make_generate_node(subject_name):
            def _gen(state: GraphState) -> GraphState:
                # subject_name을 state에 보증
                print(f"[DEBUG] {subject_name}_generate 노드 실행")
                state = {**state, "subject_area": subject_name}
                return self._generate_quiz_incremental(state)
            return _gen

        def make_validate_node(subject_name):
            def _val(state: GraphState) -> GraphState:
                state = {**state, "subject_area": subject_name}
                return self._validate_quiz_incremental(state)
            return _val

        # 과목별 노드 추가
        subject_to_nodes = {}
        for subj in self.SUBJECT_AREAS.keys():
            gen_name = f"{subj}_generate"
            val_name = f"{subj}_validate"
            workflow.add_node(gen_name, make_generate_node(subj))
            workflow.add_node(val_name, make_validate_node(subj))
            # 과목별 내부 엣지
            workflow.add_edge(gen_name, val_name)
            workflow.add_conditional_edges(
                val_name,
                self._check_completion,
                {"generate_more": gen_name, "complete": END}
            )
            subject_to_nodes[subj] = (gen_name, val_name)

        # 라우터: prepare_context 이후 과목별 generate로 분기
        def _route_to_subject(state: GraphState) -> str:
            subj = state.get("subject_area", "")
            print(f"[DEBUG] _route_to_subject: subject_area='{subj}', available_subjects={list(subject_to_nodes.keys())}")
            if subj in subject_to_nodes:
                gen_name, val_name = subject_to_nodes[subj]  # 튜플 언패킹
                print(f"[DEBUG] Found subject '{subj}', returning generate node: {gen_name}")
                return gen_name  # generate 노드명만 반환
            # 기본값(안 맞으면 설계로)
            print(f"[DEBUG] Subject '{subj}' not found, using default: 소프트웨어설계")
            gen_name, val_name = subject_to_nodes["소프트웨어설계"]
            return gen_name

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "prepare_context")
        
        # 수정: _route_to_subject 함수가 반환하는 값과 노드명을 매핑하는 딕셔너리 생성
        # _route_to_subject는 노드명을 반환하므로, routing_dict는 {노드명: 노드명} 형태여야 함
        routing_dict = {subject_to_nodes[subj][0]: subject_to_nodes[subj][0] for subj in subject_to_nodes.keys()}
        print(f"[DEBUG] routing_dict: {routing_dict}")
        print(f"[DEBUG] Available nodes: {list(workflow.nodes.keys())}")
        print(f"[DEBUG] routing_dict keys: {list(routing_dict.keys())}")
        print(f"[DEBUG] routing_dict keys in nodes: {[k in workflow.nodes for k in routing_dict.keys()]}")
        workflow.add_conditional_edges("prepare_context", _route_to_subject, routing_dict)

        self.workflow = workflow.compile()
    # --------------------------------------------------------------------

    # 단일 과목 생성(내부는 그래프 한 번 실행)
    def _generate_subject_quiz(self, subject_area: str, target_count: int = 5, difficulty: str = "중급") -> Dict[str, Any]:
        if subject_area not in self.SUBJECT_AREAS:
            return {"error": f"유효하지 않은 과목: {subject_area}"}
        keywords = self.SUBJECT_AREAS[subject_area]["keywords"]

        all_validated_questions = []
        max_rounds = 10
        current_round = 0

        while len(all_validated_questions) < target_count and current_round < max_rounds:
            current_round += 1
            remaining_needed = target_count - len(all_validated_questions)

            for i in range(0, len(keywords), 2):
                if len(all_validated_questions) >= target_count:
                    break
                combo = " ".join(keywords[i:i+3])

                initial_state = {
                    "query": combo,
                    "target_quiz_count": remaining_needed,
                    "difficulty": difficulty,
                    "generation_attempts": 0,
                    "quiz_questions": [],
                    "validated_questions": [],
                    "subject_area": subject_area
                }
                # 과목별 라우팅 그래프 단발 실행
                result = self.workflow.invoke(initial_state)

                if result.get("error"):
                    continue

                new_qs = result.get("validated_questions", [])
                if new_qs:
                    # 중복 제거
                    exists = {q.get("question", "") for q in all_validated_questions}
                    for q in new_qs:
                        if q.get("question", "") not in exists:
                            all_validated_questions.append(q)
                            exists.add(q.get("question", ""))

                if len(all_validated_questions) >= target_count:
                    break

            if current_round < max_rounds and len(all_validated_questions) < target_count:
                time.sleep(1.5)

        final_questions = all_validated_questions[:target_count]
        return {
            "subject_area": subject_area,
            "difficulty": difficulty,
            "requested_count": target_count,
            "quiz_count": len(final_questions),
            "questions": final_questions,
            "status": "SUCCESS" if len(final_questions) >= target_count else "PARTIAL"
        }

    # 3) 사용자 지정 병렬 실행로 5과목 동시 처리(최대 parallel_agents 동시)
    def _generate_full_exam(self, difficulty: str = "중급", parallel_agents: int = 2) -> Dict[str, Any]:
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
                    difficulty=difficulty
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
                              difficulty: str = "중급", parallel_agents: int = 2) -> Dict[str, Any]:
        """선택된 과목들에 대해 지정된 문제 수만큼 생성"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        start_time = time.time()

        self._generate_workflow_diagram("partial_exam", {
            "selected_subjects": selected_subjects,
            "questions_per_subject": questions_per_subject,
            "difficulty": difficulty,
            "parallel_agents": parallel_agents
        })

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
                    difficulty=difficulty
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

    def _save_to_json(self, exam_result: Dict[str, Any], filename: str = None) -> str:
        save_dir = "C:\\ET_Agent\\teacher\\TestGenerator\\test"
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

    def _generate_workflow_diagram(self, mode: str, params: Dict[str, Any]) -> None:
        """Graphviz를 사용하여 문제 생성 워크플로우 다이어그램을 생성합니다."""
        try:
            from graphviz import Digraph
            import time
            
            # 다이어그램 생성
            dot = Digraph(comment=f'정보처리기사 문제 생성 워크플로우 - {mode}')
            dot.attr(rankdir='TB', size='12,8')
            
            # 노드 스타일 정의
            dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')
            
            # 시작 노드
            dot.node('start', '사용자 입력', fillcolor='lightgreen')
            
            # 입력 파싱 노드들
            dot.node('parse', 'LLM 기반\n입력 파싱', fillcolor='lightblue')
            dot.node('validate', '파라미터 검증\n(과목/문제수/난이도)', fillcolor='lightyellow')
            
            # 모드별 분기
            if mode == "partial_exam":
                dot.node('mode', 'PARTIAL_EXAM\n(선택과목 모드)', fillcolor='orange')
                dot.node('parallel', '병렬 처리\n(ThreadPoolExecutor)', fillcolor='lightcoral')
                dot.node('merge', '결과 통합\n(과목별 결과 병합)', fillcolor='lightcoral')
            elif mode == "single_subject":
                dot.node('mode', 'SINGLE_SUBJECT\n(단일 과목 모드)', fillcolor='orange')
                dot.node('single', '단일 에이전트\n(직렬 처리)', fillcolor='lightcoral')
            else:
                dot.node('mode', 'FULL_EXAM\n(전체 과목 모드)', fillcolor='orange')
                dot.node('full_parallel', '전체 병렬\n(5과목 동시)', fillcolor='lightcoral')
            
            # 에이전트 실행
            dot.node('agent', 'TestGenerator\n.execute()', fillcolor='lightpink')
            dot.node('result', '결과 처리\n(모드별 결과 추출)', fillcolor='lightcyan')
            
            # 데이터 변환
            dot.node('transform', '데이터 변환\n(QA 형식)', fillcolor='lightcyan')
            dot.node('output', '출력\n(JSON/PDF)', fillcolor='lightgreen')
            
            # 엣지 연결
            dot.edge('start', 'parse')
            dot.edge('parse', 'validate')
            dot.edge('validate', 'mode')
            
            if mode == "partial_exam":
                dot.edge('mode', 'parallel')
                dot.edge('parallel', 'agent')
                dot.edge('agent', 'merge')
                dot.edge('merge', 'result')
            elif mode == "single_subject":
                dot.edge('mode', 'single')
                dot.edge('single', 'agent')
                dot.edge('agent', 'result')
            else:
                dot.edge('mode', 'full_parallel')
                dot.edge('full_parallel', 'agent')
                dot.edge('agent', 'result')
            
            dot.edge('result', 'transform')
            dot.edge('transform', 'output')
            
            # 서브그래프로 과목별 처리 구조 표시
            if mode == "partial_exam":
                with dot.subgraph(name='cluster_subjects') as c:
                    c.attr(label='과목별 병렬 처리', style='filled', fillcolor='lightgray')
                    subjects = params.get("selected_subjects", [])
                    count_per_subject = params.get("questions_per_subject", 10)
                    for i, subject in enumerate(subjects):
                        c.node(f'subject_{i}', f'{subject}\n({count_per_subject}문제)')
                        if i > 0:
                            c.edge(f'subject_{i-1}', f'subject_{i}', style='dashed')
            
            # 파일 저장
            output_dir = os.path.join(os.path.dirname(__file__), "workflow_diagrams")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"generation_workflow_{mode}_{timestamp}"
            filepath = os.path.join(output_dir, filename)
            
            dot.render(filepath, format='png', cleanup=True)
            print(f"\n 워크플로우 다이어그램 생성 완료: {filepath}.png")
            
            # 간단한 텍스트 요약도 출력
            print(f"\n📊 워크플로우 요약:")
            print(f"   ┌─ 모드: {mode.upper()}")
            if mode == "partial_exam":
                subjects = params.get("selected_subjects", [])
                count_per_subject = params.get("questions_per_subject", 10)
                print(f"   ├─ 선택된 과목: {', '.join(subjects)}")
                print(f"   ├─ 과목당 문제 수: {count_per_subject}개")
                print(f"   └─ 병렬 에이전트: {params.get('parallel_agents', 2)}개")
            
        except ImportError:
            print("\n⚠️  Graphviz가 설치되지 않았습니다. pip install graphviz로 설치하세요.")
        except Exception as e:
            print(f"\n❌ 다이어그램 생성 실패: {e}")
