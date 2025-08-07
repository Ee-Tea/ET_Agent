import os
import glob
from typing import List, Dict, Any, TypedDict
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime

# .env 파일 로드를 위한 임포트
from dotenv import load_dotenv

***REMOVED*** 관련 임포트
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings  # 임베딩은 여전히 HuggingFace 사용

# .env 파일 로드
load_dotenv()

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

class InfoProcessingExamRAG:
    """
    정보처리기사 출제기준에 맞는 50문제 자동 출제 시스템 (Groq 버전)
    병렬 처리 및 증분 생성 지원
    """
    
    # 정보처리기사 5개 과목 정의
    SUBJECT_AREAS = {
        "소프트웨어설계": {
            "count": 10,
            "keywords": ["요구사항", "UI 설계", "애플리케이션 설계", "인터페이스", "UML", "객체지향", "디자인패턴", "모듈화", "결합도", "응집도"]
        },
        "소프트웨어개발": {
            "count": 10,
            "keywords": ["자료구조", "스택", "큐", "리스트", "통합구현", "모듈", "패키징", "테스트케이스", "알고리즘", "인터페이스"]
        },
        "데이터베이스구축": {
            "count": 10,
            "keywords": ["SQL", "트리거", "DML", "DDL", "DCL", "정규화", "관계형모델", "E-R모델", "데이터모델링", "무결성"]
        },
        "프로그래밍언어활용": {
            "count": 10,
            "keywords": ["개발환경", "프로그래밍언어", "라이브러리", "운영체제", "네트워크", "데이터타입", "변수", "연산자"]
        },
        "정보시스템구축관리": {
            "count": 10,
            "keywords": ["소프트웨어개발방법론", "프로젝트관리", "보안", "시스템보안", "네트워크보안", "테일러링", "생명주기모델"]
        }
    }
    
    def __init__(self, data_folder="data", max_workers=2, groq_api_key=None):  # 3 → 2로 축소
        """
        초기화
        
        Args:
            data_folder: PDF 파일이 있는 폴더
            max_workers: 병렬 처리 워커 수
            groq_api_key: Groq API 키 (환경변수 GROQ_API_KEY=REDACTED 파일에서 자동 로드)
        """
        self.data_folder = data_folder
        self.max_workers = max_workers
        os.makedirs(self.data_folder, exist_ok=True)
        
        ***REMOVED*** API 키 설정 (.env 파일에서 자동 로드됨)
        if groq_api_key:
            os.environ["GROQ_API_KEY=REDACTED
        elif not os.getenv("GROQ_API_KEY=REDACTED ValueError(
                "Groq API 키가 필요합니다. 다음 중 하나를 설정해주세요:\n"
                "1. .env 파일에 GROQ_API_KEY=REDACTED 추가\n"
                "2. 환경변수 GROQ_API_KEY=REDACTED"
                "3. groq_api_key 매개변수로 직접 전달"
            )
        
        self.embeddings_model = None
        self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.workflow = None
        
        self.files_in_vectorstore = []
        self.lock = threading.Lock()
        
        self._initialize_models()
        self._build_graph()

    def _initialize_models(self):
        """임베딩 및 LLM 모델 초기화 (Groq 사용)"""
        try:
            print("임베딩 모델 초기화 중... (ko-sroberta-multitask)")
            # 임베딩은 여전히 HuggingFace 사용 (Groq는 임베딩 API 제공하지 않음)
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("임베딩 모델 초기화 완료.")
            
            print("Groq LLM 모델 초기화 중... (llama-4-scout-17b-16e-instruct)")
            ***REMOVED*** ChatGroq 모델 사용 - 업데이트된 모델명
            self.llm = ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.0,
                max_tokens=1024,  # 2048 → 1024로 축소
                timeout=120,
                max_retries=3
            )
            
            # 연결 테스트
            test_response = self.llm.invoke("안녕하세요")
            print("Groq LLM 모델 초기화 및 연결 확인 완료.")
            
        except Exception as e:
            print(f"모델 초기화 중 오류 발생: {e}")
            print("Groq API 키가 올바른지 확인해주세요.")
            raise

    def _build_vectorstore_from_all_pdfs(self) -> bool:
        """PDF를 로드하여 벡터 스토어를 생성/업데이트"""
        pdf_files = self.get_pdf_files()
        if not pdf_files:
            print(f"'{self.data_folder}' 폴더에 PDF 파일이 없습니다.")
            return False

        if self.vectorstore and set(self.files_in_vectorstore) == set(pdf_files):
            print("기존 벡터스토어를 재사용합니다 (파일 변경 없음).")
            return True

        print("새로운 벡터스토어를 생성합니다...")
        all_documents = []
        for pdf_path in pdf_files:
            try:
                print(f"  - 로딩: {os.path.basename(pdf_path)}")
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(pdf_path)
                all_documents.extend(documents)
            except Exception as e:
                error_msg = str(e)
                if "cryptography" in error_msg and "is required" in error_msg:
                    print(f"    [암호화 오류] {os.path.basename(pdf_path)} 파일은 암호화되어 있습니다.")
                    print("    해당 파일을 읽으려면 'pip install pypdf[crypto]'를 실행해주세요.")
                else:
                    print(f"    [오류] {os.path.basename(pdf_path)} 파일 로딩 실패: {e}")
                continue
        
        if not all_documents:
            print("문서를 처리할 수 없습니다.")
            return False

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_documents)
        
        self.vectorstore = FAISS.from_documents(splits, self.embeddings_model)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})  # 15 → 8로 축소
        
        self.files_in_vectorstore = pdf_files
        print(f"벡터스토어 생성 완료. 총 {len(pdf_files)}개 PDF, {len(splits)}개 청크.")
        return True

    def get_pdf_files(self) -> List[str]:
        """data 폴더에서 PDF 파일 목록 가져오기"""
        return glob.glob(os.path.join(self.data_folder, "*.pdf"))

    def list_available_pdfs(self):
        """사용 가능한 PDF 파일 목록 출력"""
        pdf_files = self.get_pdf_files()
        if not pdf_files:
            print(f"'{self.data_folder}' 폴더에 PDF 파일이 없습니다.")
            return
        
        print(f"\n=== '{self.data_folder}' 폴더의 PDF 파일 목록 ===")
        for i, file_path in enumerate(pdf_files, 1):
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / 1024
            print(f"{i}. {filename} ({file_size:.1f} KB)")

    def _retrieve_documents(self, state: GraphState) -> GraphState:
        """문서 검색 및 출처 분석 노드"""
        try:
            query = state["query"]
            subject_area = state.get("subject_area", "")
            
            enhanced_query = f"{subject_area} {query}"
            
            documents = self.retriever.invoke(enhanced_query)
            source_files = [doc.metadata.get('source_file', 'Unknown') for doc in documents]
            used_sources = list(Counter(source_files).keys())
            return {**state, "documents": documents, "used_sources": used_sources}
        except Exception as e:
            return {**state, "error": f"문서 검색 오류: {e}"}

    def _prepare_context(self, state: GraphState) -> GraphState:
        """컨텍스트 준비 노드"""
        documents = state["documents"]
        context = "\n\n".join([doc.page_content for doc in documents])
        return {**state, "context": context}

    def _generate_quiz_incremental(self, state: GraphState) -> GraphState:
        """증분 방식으로 문제 생성 - Groq 사용 (토큰 최적화)"""
        try:
            context = state["context"]
            target_quiz_count = state.get("target_quiz_count", 5)
            difficulty = state.get("difficulty", "중급")
            generation_attempts = state.get("generation_attempts", 0)
            
            validated_questions = state.get("validated_questions", [])
            subject_area = state.get("subject_area", "")
            
            needed_count = target_quiz_count - len(validated_questions)
            
            if needed_count <= 0:
                print(f"[{subject_area}] 이미 목표 문제 수({target_quiz_count}개)에 도달했습니다.")
                return {**state, "quiz_questions": validated_questions}
            
            if not context.strip():
                return {**state, "error": "검색된 문서 내용이 없습니다. 다른 키워드로 시도해보세요."}
            
            # 토큰 절약: 한 번에 생성할 문제 수를 줄임
            generate_count = min(needed_count, 3)  # 최대 3개씩만 생성
            
            print(f"\n[{subject_area}] Groq로 증분 생성 중... (보유: {len(validated_questions)}개, 필요: {needed_count}개, 생성: {generate_count}개)")
            
            # 토큰 절약: 프롬프트 간소화
            prompt_template = PromptTemplate(
                input_variables=["context", "quiz_count", "difficulty", "subject_area"],
                template="""문서 내용을 바탕으로 정보처리기사 {subject_area} 과목의 {difficulty} 난이도 문제 {quiz_count}개를 생성하세요.

[문서]
{context}

[출력 형식]
```json
{{
  "questions": [
    {{
      "question": "문제",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "answer": "정답번호",
      "explanation": "해설",
      "subject": "{subject_area}"
    }}
  ]
}}
```"""
            )
            
            # 토큰 절약: 컨텍스트 길이 제한
            prompt = prompt_template.format(
                context=context[:2000],  # 4500 → 2000으로 축소
                quiz_count=generate_count,
                difficulty=difficulty,
                subject_area=subject_area
            )
            
            ***REMOVED*** ChatGroq는 .content로 접근
            response = self.llm.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            new_questions = self._parse_quiz_response(response_content, subject_area)
            
            if not new_questions:
                return {**state, "error": "Groq가 유효한 문제를 생성하지 못했습니다."}
            
            print(f"  → 새로 생성된 문제 {len(new_questions)}개")
            
            return {
                **state,
                "quiz_questions": new_questions,
                "validated_questions": validated_questions,
                "generation_attempts": generation_attempts + 1
            }
        except Exception as e:
            return {**state, "error": f"문제 생성 중 오류 발생: {e}"}

    def _validate_quiz_incremental(self, state: GraphState) -> GraphState:
        """증분 검증 - Groq 사용 (토큰 최적화)"""
        subject_area = state.get("subject_area", "")
        print(f"\n[{subject_area}] Groq로 새로 생성된 문제 검증 시작...")
        
        previously_validated = state.get("validated_questions", [])
        new_questions = state.get("quiz_questions", [])
        context = state.get("context", "")
        target_quiz_count = state.get("target_quiz_count", 5)
        generation_attempts = state.get("generation_attempts", 0)

        if not new_questions:
            print("검증할 새 문제가 없습니다.")
            return {**state, "validated_questions": previously_validated}

        newly_validated = []
        
        # 토큰 절약: 간소화된 검증 프롬프트
        validation_prompt_template = PromptTemplate(
            input_variables=["context", "question_data"],
            template="""문서 내용에 근거하여 다음 문제가 유효한지 평가하세요.

[문서]
{context}

[문제]
{question_data}

유효하면 true, 아니면 false로만 응답하세요."""
        )
        
        needed = target_quiz_count - len(previously_validated)
        
        for i, q in enumerate(new_questions):
            if len(newly_validated) >= needed:
                print(f"  → 필요한 {needed}개 문제 검증 완료!")
                break
                
            try:
                question_str = json.dumps(q, ensure_ascii=False, indent=2)
                # 토큰 절약: 컨텍스트 길이 제한
                prompt = validation_prompt_template.format(context=context[:1500], question_data=question_str)
                
                response = self.llm.invoke(prompt)
                response_str = response.content if hasattr(response, 'content') else str(response)
                
                # 간단한 검증: true/false만 확인
                if "true" in response_str.lower():
                    print(f"  - [VALID] 문제 {i+1}: \"{q.get('question', '')[:40]}...\"")
                    newly_validated.append(q)
                else:
                    print(f"  - [INVALID] 문제 {i+1}: 문서 내용과 불일치")

            except Exception as e:
                print(f"  - [검증 오류] 문제 {i+1}: {e}")
        
        all_validated = previously_validated + newly_validated
        
        print(f"[{subject_area}] 검증 결과: 기존 {len(previously_validated)}개 + 신규 {len(newly_validated)}개 = 총 {len(all_validated)}개")
        
        need_more_questions = (len(all_validated) < target_quiz_count and 
                              generation_attempts < 15)
        
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
        subject_area = state.get("subject_area", "")
        
        if validated_count >= target_count:
            print(f"[{subject_area}] 목표 문제 수 {target_count}개 달성! ✓")
            return "complete"
        elif generation_attempts < 15:
            print(f"[{subject_area}] 추가 생성 필요 (현재: {validated_count}개/{target_count}개, 시도: {generation_attempts+1}회)")
            return "generate_more"
        else:
            print(f"[{subject_area}] 최대 시도 횟수(15회)에 도달. 현재까지 {validated_count}개 생성.")
            return "complete"

    def _parse_quiz_response(self, response: str, subject_area: str = "") -> List[Dict[str, Any]]:
        """Groq 응답에서 JSON 형식의 문제를 파싱"""
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
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return []
        except Exception as e:
            print(f"응답 파싱 중 오류: {e}")
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

    def generate_subject_quiz_parallel(self, subject_area: str, target_count: int = 5, difficulty: str = "중급") -> Dict[str, Any]:
        """특정 과목의 문제를 병렬로 생성"""
        if subject_area not in self.SUBJECT_AREAS:
            return {"error": f"유효하지 않은 과목: {subject_area}"}
        
        keywords = self.SUBJECT_AREAS[subject_area]["keywords"]
        
        print(f"\n=== {subject_area} 문제 생성 시작 (목표: {target_count}개, Groq 사용) ===")
        
        keyword_combinations = []
        for i in range(0, len(keywords), 2):
            combo = " ".join(keywords[i:i+3])
            keyword_combinations.append(combo)
        
        all_validated_questions = []
        
        with ThreadPoolExecutor(max_workers=min(2, len(keyword_combinations))) as executor:  # 3 → 2로 축소
            futures = {}
            
            # 목표 달성까지 계속 시도하는 루프
            max_rounds = 10  # 무한루프 방지를 위한 안전장치
            current_round = 0
            
            while len(all_validated_questions) < target_count and current_round < max_rounds:
                current_round += 1
                remaining_needed = target_count - len(all_validated_questions)
                print(f"\n[{subject_area}] 🔄 Round {current_round}: 추가로 {remaining_needed}개 문제 필요")
                
                # 남은 키워드 조합으로 병렬 시도
                for combo in keyword_combinations:
                    if len(all_validated_questions) >= target_count:
                        break
                        
                    future = executor.submit(self._generate_with_keywords, 
                                           combo, subject_area, remaining_needed, difficulty)
                    futures[future] = combo
                
                # 결과 수집
                completed_futures = []
                for future in as_completed(futures):
                    combo = futures[future]
                    completed_futures.append(future)
                    try:
                        result = future.result(timeout=120)
                        if "questions" in result and result["questions"]:
                            with self.lock:
                                # 중복 방지를 위해 기존 문제와 비교
                                new_questions = []
                                existing_questions = [q.get('question', '') for q in all_validated_questions]
                                
                                for q in result["questions"]:
                                    if q.get('question', '') not in existing_questions:
                                        new_questions.append(q)
                                        existing_questions.append(q.get('question', ''))
                                
                                all_validated_questions.extend(new_questions)
                                if new_questions:
                                    print(f"  → [{subject_area}] '{combo}' 키워드로 {len(new_questions)}개 추가 (총 {len(all_validated_questions)}개)")
                            
                            if len(all_validated_questions) >= target_count:
                                print(f"🎉 [{subject_area}] 목표 달성! 총 {len(all_validated_questions)}개")
                                break
                    except Exception as e:
                        print(f"  → [{subject_area}] '{combo}' 키워드 실패: {e}")
                
                # 완료된 future들을 딕셔너리에서 제거
                for future in completed_futures:
                    futures.pop(future, None)
                
                if len(all_validated_questions) >= target_count:
                    break
                    
                # 다음 라운드 전 잠시 대기
                if current_round < max_rounds and len(all_validated_questions) < target_count:
                    print(f"  ⏳ 다음 라운드까지 2초 대기...")
                    time.sleep(2)
        
            if len(all_validated_questions) < target_count:
                print(f"⚠️ [{subject_area}] {max_rounds} 라운드 완료했지만 목표 미달성: {len(all_validated_questions)}/{target_count}개")
            
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
        """특정 키워드로 문제 생성 (스레드 안전)"""
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

    def generate_full_exam_parallel(self, difficulty: str = "중급") -> Dict[str, Any]:
        """정보처리기사 전체 50문제를 병렬로 생성 (Groq 사용)"""
        if not self._build_vectorstore_from_all_pdfs():
            return {"error": f"'{self.data_folder}' 폴더에 PDF가 없어 문제를 생성할 수 없습니다."}
        
        print("\n" + "="*80)
        print("  정보처리기사 50문제 자동 생성을 시작합니다! (Groq + 병렬 처리)")
        print("  ⚠️  각 과목별로 반드시 10문제씩 생성합니다.")
        print("="*80)
        
        start_time = time.time()
        
        full_exam_result = {
            "exam_title": "정보처리기사 모의고사 (Groq 버전)",
            "total_questions": 0,
            "difficulty": difficulty,
            "subjects": {},
            "all_questions": [],
            "generation_summary": {},
            "failed_subjects": [],
            "model_info": "Groq llama-4-scout-17b-16e-instruct"
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for subject_area, subject_info in self.SUBJECT_AREAS.items():
                target_count = subject_info["count"]
                print(f"📚 [{subject_area}] 과목 문제 생성 시작... (목표: {target_count}문제, Groq 사용)")
                
                future = executor.submit(
                    self.generate_subject_quiz_parallel,
                    subject_area=subject_area,
                    target_count=target_count,
                    difficulty=difficulty
                )
                futures[future] = subject_area
            
            completed_subjects = 0
            for future in as_completed(futures):
                subject_area = futures[future]
                completed_subjects += 1
                
                try:
                    subject_result = future.result(timeout=1800)  # 30분 타임아웃으로 증가
                    
                    if "error" in subject_result:
                        print(f"❌ [{completed_subjects}/5] {subject_area}: 실패")
                        full_exam_result["failed_subjects"].append({
                            "subject": subject_area,
                            "error": subject_result["error"]
                        })
                    else:
                        questions = subject_result["questions"]
                        actual_count = len(questions)
                        
                        full_exam_result["subjects"][subject_area] = {
                            "requested_count": 10,
                            "actual_count": actual_count,
                            "questions": questions,
                            "status": subject_result.get("status", "UNKNOWN")
                        }
                        
                        full_exam_result["all_questions"].extend(questions)
                        print(f"✅ [{completed_subjects}/5] {subject_area}: {actual_count}/10문제 완료")
                        
                except Exception as e:
                    print(f"❌ [{completed_subjects}/5] {subject_area}: 예외 발생 - {e}")
                    full_exam_result["failed_subjects"].append({
                        "subject": subject_area,
                        "error": str(e)
                    })
        
        total_generated = len(full_exam_result["all_questions"])
        elapsed_time = time.time() - start_time
        
        full_exam_result["total_questions"] = total_generated
        full_exam_result["generation_summary"] = {
            "target_total": 50,
            "actual_total": total_generated,
            "success_rate": f"{total_generated/50*100:.1f}%",
            "successful_subjects": 5 - len(full_exam_result["failed_subjects"]),
            "failed_subjects": len(full_exam_result["failed_subjects"]),
            "completion_status": "COMPLETE" if total_generated >= 50 else "PARTIAL",
            "generation_time": f"{elapsed_time:.1f}초"
        }
        
        print(f"\n" + "="*80)
        print(f"🎯 최종 결과: {total_generated}/50문제 생성 완료! (Groq 사용)")
        print(f"⏱️  소요 시간: {elapsed_time:.1f}초")
        print(f"✅ 성공한 과목: {5 - len(full_exam_result['failed_subjects'])}/5개")
        print(f"❌ 실패한 과목: {len(full_exam_result['failed_subjects'])}/5개")
        
        if full_exam_result["failed_subjects"]:
            print(f"\n실패한 과목:")
            for failed in full_exam_result["failed_subjects"]:
                print(f"  - {failed['subject']}")
        
        print("="*80)
        
        return full_exam_result

    def save_exam_to_json(self, exam_result: Dict[str, Any], filename: str = None):
        """50문제 시험을 JSON 파일로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"정보처리기사_50문제_Groq_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(exam_result, f, ensure_ascii=False, indent=2)
            
            total_questions = exam_result.get("total_questions", 0)
            print(f"\n" + "="*60)
            print(f"  정보처리기사 모의고사 ({total_questions}문제)가 '{filename}' 파일로 저장되었습니다!")
            print("="*60)
            
            for subject, info in exam_result.get("subjects", {}).items():
                if "error" not in info:
                    print(f"  - {subject}: {info['actual_count']}/{info['requested_count']}문제")
                else:
                    print(f"  - {subject}: 생성 실패")
                    
        except Exception as e:
            print(f"파일 저장 오류: {e}")

    def display_exam_summary(self, exam_result: Dict[str, Any]):
        """시험 결과 요약 출력"""
        print("\n" + "="*80)
        print(f"  {exam_result.get('exam_title', '시험')} 생성 완료!")
        print("="*80)
        
        summary = exam_result.get("generation_summary", {})
        print(f"📊 전체 문제 수: {summary.get('actual_total', 0)}/{summary.get('target_total', 50)}문제")
        print(f"🤖 사용 모델: {exam_result.get('model_info', 'Unknown')}")
        print(f"📈 성공률: {summary.get('success_rate', '0%')}")
        print(f"⏱️  소요 시간: {summary.get('generation_time', 'N/A')}")
        print(f"✅ 성공한 과목: {summary.get('successful_subjects', 0)}/5개")
        print(f"❌ 실패한 과목: {summary.get('failed_subjects', 0)}/5개")
        print(f"🎯 완성도: {summary.get('completion_status', 'UNKNOWN')}")
        
        print("\n[과목별 상세 결과]")
        for subject, info in exam_result.get("subjects", {}).items():
            status_icon = "✅" if info.get("status") == "SUCCESS" else "⚠️" if info.get("status") == "PARTIAL" else "❌"
            if "error" not in info:
                print(f"  {status_icon} {subject}: {info['actual_count']}/{info['requested_count']}문제")
            else:
                print(f"  {status_icon} {subject}: 생성 실패")
        
        failed_subjects = exam_result.get("failed_subjects", [])
        if failed_subjects:
            print(f"\n⚠️  실패한 과목이 있습니다:")
            for failed in failed_subjects:
                print(f"   - {failed['subject']}: PDF 문서에 관련 내용이 부족하거나 키워드가 적합하지 않을 수 있습니다.")
            print(f"   💡 해결 방안: 해당 과목의 상세한 PDF 자료를 추가하거나 다시 시도해보세요.")
        
        print()

    def generate_subject_quiz(self, subject_area: str, target_count: int = 5, difficulty: str = "중급") -> Dict[str, Any]:
        """특정 과목의 문제를 생성 (비병렬 버전 - 호환성 유지)"""
        return self.generate_subject_quiz_parallel(subject_area, target_count, difficulty)

    def generate_full_exam(self, difficulty: str = "중급", parallel: bool = True) -> Dict[str, Any]:
        """정보처리기사 전체 50문제를 생성"""
        if parallel:
            return self.generate_full_exam_parallel(difficulty)
        else:
            return self._generate_full_exam_sequential(difficulty)

    def _generate_full_exam_sequential(self, difficulty: str = "중급") -> Dict[str, Any]:
        """순차 처리 방식 (기존 방식)"""
        if not self._build_vectorstore_from_all_pdfs():
            return {"error": f"'{self.data_folder}' 폴더에 PDF가 없어 문제를 생성할 수 없습니다."}
        
        print("\n" + "="*80)
        print("  정보처리기사 50문제 자동 생성을 시작합니다! (Groq + 순차 처리)")
        print("="*80)
        
        start_time = time.time()
        
        full_exam_result = {
            "exam_title": "정보처리기사 모의고사 (Groq 순차 버전)",
            "total_questions": 0,
            "difficulty": difficulty,
            "subjects": {},
            "all_questions": [],
            "generation_summary": {},
            "failed_subjects": [],
            "model_info": "Groq llama-4-scout-17b-16e-instruct"
        }
        
        total_generated = 0
        
        for i, (subject_area, subject_info) in enumerate(self.SUBJECT_AREAS.items(), 1):
            target_count = subject_info["count"]
            
            print(f"\n📚 [{i}/5] {subject_area} 과목 시작 (목표: {target_count}문제, Groq 사용)")
            print("─" * 60)
            
            subject_result = self.generate_subject_quiz_parallel(
                subject_area=subject_area,
                target_count=target_count,
                difficulty=difficulty
            )
            
            if "error" in subject_result:
                print(f"❌ [{subject_area}] 실패: {subject_result['error']}")
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
                
                print(f"✅ [{subject_area}] 완료: {actual_count}/{target_count}개 문제 생성")
                print(f"   📊 현재까지 총 {total_generated}개 문제 생성됨")
            
            if i < 5:
                print(f"   ⏳ 다음 과목 준비 중... (2초 대기)")
                time.sleep(2)
        
        elapsed_time = time.time() - start_time
        
        full_exam_result["total_questions"] = total_generated
        full_exam_result["generation_summary"] = {
            "target_total": 50,
            "actual_total": total_generated,
            "success_rate": f"{total_generated/50*100:.1f}%",
            "successful_subjects": 5 - len(full_exam_result["failed_subjects"]),
            "failed_subjects": len(full_exam_result["failed_subjects"]),
            "completion_status": "COMPLETE" if total_generated >= 50 else "PARTIAL",
            "generation_time": f"{elapsed_time:.1f}초"
        }
        
        print(f"\n" + "="*80)
        print(f"🎯 최종 결과: {total_generated}/50문제 생성 완료! (Groq 사용)")
        print(f"⏱️  소요 시간: {elapsed_time:.1f}초")
        print("="*80)
        
        return full_exam_result


# --- 대화형 인터페이스 ---
def interactive_menu(rag_system):
    """사용자와 상호작용하는 메뉴 시스템"""
    while True:
        print("\n" + "="*70)
        print("  정보처리기사 50문제 자동 출제 시스템 (Groq 버전)")
        print("  [병렬 처리 & 증분 생성 지원]")
        print("="*70)
        print("1. 정보처리기사 전체 50문제 생성 (Groq + 병렬 처리 - 추천)")
        print("2. 정보처리기사 전체 50문제 생성 (Groq + 순차 처리)")
        print("3. 특정 과목만 문제 생성 (Groq)")
        print("4. 사용 가능한 PDF 목록 보기")
        print("0. 종료")
        print("-"*70)
        
        choice = input("선택하세요: ").strip()
        
        if choice == "1":
            difficulty_str = input("난이도를 입력하세요 (초급/중급/고급, 기본값: 중급): ").strip()
            difficulty = difficulty_str if difficulty_str in ["초급", "중급", "고급"] else "중급"
            
            print(f"\n정보처리기사 50문제 생성을 시작합니다. (난이도: {difficulty}, Groq + 병렬 처리)")
            print("Groq API를 사용한 병렬 처리로 더 빠르고 정확하게 생성됩니다...")
            
            exam_result = rag_system.generate_full_exam(difficulty=difficulty, parallel=True)
            
            if "error" not in exam_result:
                rag_system.display_exam_summary(exam_result)
                
                if input("\n생성된 문제를 JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower() == 'y':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    default_filename = f"정보처리기사_50문제_Groq_{difficulty}_{timestamp}.json"
                    filename = input(f"저장할 파일명 (기본값: {default_filename}): ").strip() or default_filename
                    rag_system.save_exam_to_json(exam_result, filename)
            else:
                print(f"오류: {exam_result['error']}")
        
        elif choice == "2":
            difficulty_str = input("난이도를 입력하세요 (초급/중급/고급, 기본값: 중급): ").strip()
            difficulty = difficulty_str if difficulty_str in ["초급", "중급", "고급"] else "중급"
            
            print(f"\n정보처리기사 50문제 생성을 시작합니다. (난이도: {difficulty}, Groq + 순차 처리)")
            print("Groq API를 사용하여 순차적으로 각 과목을 처리합니다...")
            
            exam_result = rag_system.generate_full_exam(difficulty=difficulty, parallel=False)
            
            if "error" not in exam_result:
                rag_system.display_exam_summary(exam_result)
                
                if input("\n생성된 문제를 JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower() == 'y':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    default_filename = f"정보처리기사_50문제_Groq_{difficulty}_{timestamp}.json"
                    filename = input(f"저장할 파일명 (기본값: {default_filename}): ").strip() or default_filename
                    rag_system.save_exam_to_json(exam_result, filename)
            else:
                print(f"오류: {exam_result['error']}")
        
        elif choice == "3":
            print("\n[정보처리기사 과목 선택]")
            subjects = list(rag_system.SUBJECT_AREAS.keys())
            for i, subject in enumerate(subjects, 1):
                count = rag_system.SUBJECT_AREAS[subject]["count"]
                print(f"{i}. {subject} ({count}문제)")
            
            try:
                subject_choice = int(input("과목 번호를 선택하세요: "))
                if 1 <= subject_choice <= len(subjects):
                    selected_subject = subjects[subject_choice - 1]
                    target_count = rag_system.SUBJECT_AREAS[selected_subject]["count"]
                    
                    count_input = input(f"생성할 문제 수 (기본값: {target_count}): ").strip()
                    if count_input:
                        try:
                            target_count = int(count_input)
                        except ValueError:
                            target_count = rag_system.SUBJECT_AREAS[selected_subject]["count"]
                    
                    difficulty_str = input("난이도 (초급/중급/고급, 기본값: 중급): ").strip()
                    difficulty = difficulty_str if difficulty_str in ["초급", "중급", "고급"] else "중급"
                    
                    print(f"\n{selected_subject} 문제 {target_count}개를 Groq로 생성합니다...")
                    
                    subject_result = rag_system.generate_subject_quiz(
                        subject_area=selected_subject,
                        target_count=target_count,
                        difficulty=difficulty
                    )
                    
                    if "error" not in subject_result:
                        questions = subject_result["questions"]
                        actual_count = len(questions)
                        
                        print(f"\n[{selected_subject}] {actual_count}/{target_count}문제 생성 완료! (Groq 사용)")
                        
                        if questions and input("생성된 문제를 미리보시겠습니까? (y/n): ").strip().lower() == 'y':
                            for i, q in enumerate(questions[:3], 1):
                                print(f"\n[문제 {i}] {q.get('question', '')}")
                                for option in q.get('options', []):
                                    print(f"{option}")
                                print(f"▶ 정답: {q.get('answer', '')}")
                                print(f"▶ 해설: {q.get('explanation', '')}")
                                if i < 3 and i < len(questions):
                                    input("다음 문제를 보려면 Enter를 누르세요...")
                            
                            if len(questions) > 3:
                                print(f"\n... 외 {len(questions)-3}개 문제가 더 있습니다.")
                        
                        if input("\n생성된 문제를 JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower() == 'y':
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            default_filename = f"{selected_subject}_{actual_count}문제_Groq_{difficulty}_{timestamp}.json"
                            filename = input(f"파일명 (기본값: {default_filename}): ").strip() or default_filename
                            
                            try:
                                with open(filename, 'w', encoding='utf-8') as f:
                                    json.dump(subject_result, f, ensure_ascii=False, indent=2)
                                print(f"'{filename}' 파일로 저장되었습니다.")
                            except Exception as e:
                                print(f"파일 저장 오류: {e}")
                    else:
                        print(f"오류: {subject_result['error']}")
                else:
                    print("유효하지 않은 과목 번호입니다.")
            except ValueError:
                print("숫자를 입력해주세요.")

        elif choice == "4":
            rag_system.list_available_pdfs()
            
        elif choice == "0":
            print("시스템을 종료합니다.")
            break
            
        else:
            print("잘못된 선택입니다. 0, 1, 2, 3, 4 중에서 선택해주세요.")


def main():
    """메인 실행 함수"""
    try:
        print("\n" + "="*80)
        print("  정보처리기사 50문제 자동 출제 시스템 (Groq 버전) 초기화 중...")
        print("="*80)
        
        # .env 파일에서 Groq API 키 확인
        groq_api_key = os.getenv("GROQ_API_KEY=REDACTED not groq_api_key:
            print("\n⚠️  Groq API 키가 설정되지 않았습니다!")
            print("다음 방법으로 API 키를 설정하세요:")
            print("1. 프로젝트 폴더에 .env 파일 생성 후 다음 내용 추가:")
            print("   GROQ_API_KEY=REDACTED")
            print("2. 또는 환경변수 설정: export GROQ_API_KEY=REDACTED'")
            print("\n📁 .env 파일 예시:")
            print("   # .env")
            print("   GROQ_API_KEY=REDACTED")
            print("\nGroq API 키는 https://console.groq.com 에서 발급받을 수 있습니다.")
            return
        else:
            print(f"✅ .env 파일에서 Groq API 키를 성공적으로 로드했습니다.")
        
        # 시스템 초기화
        rag_system = InfoProcessingExamRAG(
            data_folder="data", 
            max_workers=2  # 3 → 2로 축소
        )
        
        print("\n[시스템 초기화 완료]")
        print(f"🤖 사용 모델: Groq llama-4-scout-17b-16e-instruct")
        print(f"📁 데이터 폴더: '{rag_system.data_folder}' (정보처리기사 PDF 파일 위치)")
        print(f"⚡ 병렬 처리 워커 수: {rag_system.max_workers}개")
        print("\n[정보처리기사 출제 과목]")
        for subject, info in rag_system.SUBJECT_AREAS.items():
            print(f"  - {subject}: {info['count']}문제")
        print(f"  📊 총 50문제")
        
        print("\n💡 Groq + .env 설정의 장점:")
        print("  - 안전한 API 키 관리 (.env 파일)")
        print("  - 더 빠른 응답 속도 (Groq)")
        print("  - 높은 품질의 문제 생성")
        print("  - 안정적인 JSON 출력")
        
        interactive_menu(rag_system)
        
    except Exception as e:
        print(f"프로그램 실행 중 오류가 발생했습니다: {e}")
        if "GROQ_API_KEY=REDACTED str(e) or "API" in str(e):
            print("\n💡 해결 방법:")
            print("1. .env 파일이 프로젝트 루트 폴더에 있는지 확인")
            print("2. .env 파일에 GROQ_API_KEY=REDACTED 형식으로 작성되었는지 확인")
            print("3. Groq API 키가 유효한지 https://console.groq.com 에서 확인")
        else:
            print("시스템 설정을 확인해주세요.")


if __name__ == "__main__":
    main()