import os
import glob
from typing import List, Dict, Any, TypedDict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
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
    validated_questions: List[Dict[str, Any]]  # 검증된 문제들을 별도로 저장

class InfoProcessingExamRAG:
    """
    정보처리기사 출제기준에 맞는 100문제 자동 출제 시스템
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
    
    def __init__(self, data_folder="data", max_workers=3):
        """초기화"""
        self.data_folder = data_folder
        self.max_workers = max_workers  # 병렬 처리 워커 수
        os.makedirs(self.data_folder, exist_ok=True)
        
        self.embeddings_model = None
        self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.workflow = None
        
        self.files_in_vectorstore = []
        self.lock = threading.Lock()  # 스레드 안전성을 위한 락
        
        self._initialize_models()
        self._build_graph()

    def _initialize_models(self):
        """임베딩 및 LLM 모델 초기화"""
        try:
            print("임베딩 모델 초기화 중... (ko-sroberta-multitask)")
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("임베딩 모델 초기화 완료.")
            
            print("LLM 모델 초기화 중... (llama3.2:3b)")
            self.llm = OllamaLLM(model="llama3.2:3b", temperature=0.1)
            self.llm.invoke("안녕하세요")
            print("LLM 모델 초기화 및 연결 확인 완료.")
            
        except Exception as e:
            print(f"모델 초기화 중 심각한 오류 발생: {e}")
            print("Ollama 서버가 실행 중인지 확인해주세요. (명령어: ollama serve)")
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
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        
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
            
            # 과목별 키워드를 쿼리에 추가
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
        """증분 방식으로 문제 생성 - 이미 검증된 문제는 유지"""
        try:
            context = state["context"]
            target_quiz_count = state.get("target_quiz_count", 5)
            difficulty = state.get("difficulty", "중급")
            generation_attempts = state.get("generation_attempts", 0)
            
            # 이미 검증된 문제들 유지
            validated_questions = state.get("validated_questions", [])
            subject_area = state.get("subject_area", "")
            
            # 필요한 추가 문제 수 계산
            needed_count = target_quiz_count - len(validated_questions)
            
            if needed_count <= 0:
                print(f"[{subject_area}] 이미 목표 문제 수({target_quiz_count}개)에 도달했습니다.")
                return {**state, "quiz_questions": validated_questions}
            
            if not context.strip():
                return {**state, "error": "검색된 문서 내용이 없습니다. 다른 키워드로 시도해보세요."}
            
            # 검증 실패를 고려해 여분 생성
            generate_count = max(needed_count * 2, 5)
            
            print(f"\n[{subject_area}] 증분 생성 중... (보유: {len(validated_questions)}개, 필요: {needed_count}개, 생성: {generate_count}개)")
            
            prompt_template = PromptTemplate(
                input_variables=["context", "quiz_count", "difficulty", "subject_area", "existing_count"],
                template="""You are a machine that only outputs a single, valid JSON object. Do not add any text before or after the JSON.
Based *only* on the provided document about 정보처리기사, create {quiz_count} NEW multiple-choice questions in Korean at the {difficulty} level for the subject area: {subject_area}.

Note: You already have {existing_count} validated questions. Create NEW, different questions.

IMPORTANT GUIDELINES:
1. Questions MUST be directly verifiable from the provided document content
2. Focus on {subject_area} topics and concepts
3. Create practical questions that test understanding of key concepts
4. Each question should have 4 options with only ONE correct answer
5. Provide clear explanations based on the document content
6. Make questions vary in difficulty and topics within {subject_area}
7. Avoid duplicating existing questions

[Document Content]
{context}

[Required JSON Format]
```json
{{
  "questions": [
    {{
      "question": "문제 내용",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "answer": "정답 번호 (예: 1)",
      "explanation": "정답에 대한 간결한 해설",
      "subject": "{subject_area}"
    }}
  ]
}}
```

Your JSON output:"""
            )
            
            prompt = prompt_template.format(
                context=context[:4500],
                quiz_count=generate_count,
                difficulty=difficulty,
                subject_area=subject_area,
                existing_count=len(validated_questions)
            )
            response = self.llm.invoke(prompt)
            new_questions = self._parse_quiz_response(response, subject_area)
            
            if not new_questions:
                return {**state, "error": "LLM이 유효한 문제를 생성하지 못했습니다."}
            
            print(f"  → 새로 생성된 문제 {len(new_questions)}개")
            
            # 새로 생성된 문제만 검증 대상으로
            return {
                **state,
                "quiz_questions": new_questions,  # 새로 생성된 것만
                "validated_questions": validated_questions,  # 기존 검증된 것 유지
                "generation_attempts": generation_attempts + 1
            }
        except Exception as e:
            return {**state, "error": f"문제 생성 중 오류 발생: {e}"}

    def _validate_quiz_incremental(self, state: GraphState) -> GraphState:
        """증분 검증 - 새로 생성된 문제만 검증하고 기존 검증된 문제와 합침"""
        subject_area = state.get("subject_area", "")
        print(f"\n[{subject_area}] 새로 생성된 문제 검증 시작...")
        
        # 기존 검증된 문제들
        previously_validated = state.get("validated_questions", [])
        # 새로 생성된 문제들
        new_questions = state.get("quiz_questions", [])
        context = state.get("context", "")
        target_quiz_count = state.get("target_quiz_count", 5)
        generation_attempts = state.get("generation_attempts", 0)

        if not new_questions:
            print("검증할 새 문제가 없습니다.")
            return {**state, "validated_questions": previously_validated}

        newly_validated = []
        
        validation_prompt_template = PromptTemplate(
            input_variables=["context", "question_data"],
            template="""당신은 정보처리기사 출제기준에 맞는 문제를 검증하는 전문가입니다.
아래 제공된 '문서 내용'에만 근거하여 다음 '퀴즈 문제'를 평가해주세요.

[문서 내용]
{context}

[평가할 퀴즈 문제]
{question_data}

[평가 기준]
1. 이 질문과 정답이 '문서 내용'에서 직접적으로 확인 가능한가?
2. 정보처리기사 시험 수준에 적합한 난이도인가?
3. 4개 선택지가 명확하고 정답이 유일한가?
4. 해설이 문서 내용을 정확히 반영하고 있는가?

[응답 형식]
'is_valid'(boolean)와 'reason'(한국어 설명) 키를 가진 JSON 객체로만 응답해주세요.

Your JSON response:"""
        )
        
        # 필요한 추가 문제 수
        needed = target_quiz_count - len(previously_validated)
        
        for i, q in enumerate(new_questions):
            if len(newly_validated) >= needed:
                print(f"  → 필요한 {needed}개 문제 검증 완료!")
                break
                
            try:
                question_str = json.dumps(q, ensure_ascii=False, indent=2)
                prompt = validation_prompt_template.format(context=context[:4000], question_data=question_str)
                response_str = self.llm.invoke(prompt)
                
                match = re.search(r'\{.*\}', response_str, re.DOTALL)
                if not match:
                    print(f"  - [검증 오류] 문제 {i+1}: LLM이 유효한 JSON을 반환하지 않음.")
                    continue

                validation_result = json.loads(match.group(0))

                if validation_result.get("is_valid") is True:
                    print(f"  - [VALID] 문제 {i+1}: \"{q.get('question', '')[:40]}...\"")
                    newly_validated.append(q)
                else:
                    reason = validation_result.get('reason', '이유 없음')
                    print(f"  - [INVALID] 문제 {i+1}: {reason}")

            except Exception as e:
                print(f"  - [검증 오류] 문제 {i+1}: {e}")
        
        # 기존 검증된 문제와 새로 검증된 문제 합치기
        all_validated = previously_validated + newly_validated
        
        print(f"[{subject_area}] 검증 결과: 기존 {len(previously_validated)}개 + 신규 {len(newly_validated)}개 = 총 {len(all_validated)}개")
        
        # 목표 달성 여부 판단
        need_more_questions = (len(all_validated) < target_quiz_count and 
                              generation_attempts < 15)  # 최대 시도 횟수 증가
        
        return {
            **state,
            "validated_questions": all_validated,
            "quiz_questions": all_validated,  # 최종 결과용
            "need_more_questions": need_more_questions
        }

    def _check_completion(self, state: GraphState) -> str:
        """문제 생성 완료 여부를 체크하는 조건부 노드"""
        validated_count = len(state.get("validated_questions", []))
        target_count = state.get("target_quiz_count", 5)
        need_more = state.get("need_more_questions", False)
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
            return "complete"  # 부분 완성도 허용

    def _parse_quiz_response(self, response: str, subject_area: str = "") -> List[Dict[str, Any]]:
        """LLM 응답에서 JSON 형식의 문제를 파싱"""
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
            
            # 각 문제에 과목 정보 추가 및 선택지 번호 정리
            for question in data["questions"]:
                if "options" in question and isinstance(question["options"], list):
                    numbered_options = []
                    for i, option_text in enumerate(question["options"], 1):
                        cleaned_text = re.sub(r'^\s*\d+\.\s*', '', option_text).strip()
                        numbered_options.append(f"  {i}. {cleaned_text}")
                    question["options"] = numbered_options
                
                # 과목 정보 추가
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
        """LangGraph 워크플로우 구성 - 증분 생성 지원"""
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

    def generate_subject_quiz_parallel(self, subject_area: str, target_count: int = 20, difficulty: str = "중급") -> Dict[str, Any]:
        """특정 과목의 문제를 병렬로 생성"""
        if subject_area not in self.SUBJECT_AREAS:
            return {"error": f"유효하지 않은 과목: {subject_area}"}
        
        keywords = self.SUBJECT_AREAS[subject_area]["keywords"]
        
        print(f"\n=== {subject_area} 문제 생성 시작 (목표: {target_count}개) ===")
        
        # 여러 키워드 조합을 동시에 시도
        keyword_combinations = []
        for i in range(0, len(keywords), 2):
            combo = " ".join(keywords[i:i+3])
            keyword_combinations.append(combo)
        
        all_validated_questions = []
        
        # 병렬로 여러 키워드 조합 시도
        with ThreadPoolExecutor(max_workers=min(3, len(keyword_combinations))) as executor:
            futures = {}
            
            for combo in keyword_combinations[:3]:  # 처음 3개 조합만 병렬 시도
                if len(all_validated_questions) >= target_count:
                    break
                    
                future = executor.submit(self._generate_with_keywords, 
                                       combo, subject_area, target_count - len(all_validated_questions), difficulty)
                futures[future] = combo
            
            for future in as_completed(futures):
                combo = futures[future]
                try:
                    result = future.result(timeout=60)
                    if "questions" in result:
                        with self.lock:
                            all_validated_questions.extend(result["questions"])
                            print(f"  → [{subject_area}] '{combo}' 키워드로 {len(result['questions'])}개 추가 (총 {len(all_validated_questions)}개)")
                        
                        if len(all_validated_questions) >= target_count:
                            print(f"🎉 [{subject_area}] 목표 달성!")
                            break
                except Exception as e:
                    print(f"  → [{subject_area}] '{combo}' 키워드 실패: {e}")
        
        # 결과 정리
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
        """정보처리기사 전체 100문제를 병렬로 생성"""
        if not self._build_vectorstore_from_all_pdfs():
            return {"error": f"'{self.data_folder}' 폴더에 PDF가 없어 문제를 생성할 수 없습니다."}
        
        print("\n" + "="*80)
        print("  정보처리기사 50문제 자동 생성을 시작합니다! (병렬 처리 모드)")
        print("  ⚠️  각 과목별로 반드시 10문제씩 생성합니다.")
        print("="*80)
        
        start_time = time.time()
        
        full_exam_result = {
            "exam_title": "정보처리기사 모의고사",
            "total_questions": 0,
            "difficulty": difficulty,
            "subjects": {},
            "all_questions": [],
            "generation_summary": {},
            "failed_subjects": []
        }
        
        # 병렬로 모든 과목 동시 처리
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for subject_area, subject_info in self.SUBJECT_AREAS.items():
                target_count = subject_info["count"]
                print(f"📚 [{subject_area}] 과목 문제 생성 시작... (목표: {target_count}문제)")
                
                future = executor.submit(
                    self.generate_subject_quiz_parallel,
                    subject_area=subject_area,
                    target_count=target_count,
                    difficulty=difficulty
                )
                futures[future] = subject_area
            
            # 결과 수집
            completed_subjects = 0
            for future in as_completed(futures):
                subject_area = futures[future]
                completed_subjects += 1
                
                try:
                    subject_result = future.result(timeout=300)  # 5분 타임아웃
                    
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
        
        # 최종 집계
        total_generated = len(full_exam_result["all_questions"])
        elapsed_time = time.time() - start_time
        
        full_exam_result["total_questions"] = total_generated
        full_exam_result["generation_summary"] = {
            "target_total": 100,
            "actual_total": total_generated,
            "success_rate": f"{total_generated/100*100:.1f}%",
            "successful_subjects": 5 - len(full_exam_result["failed_subjects"]),
            "failed_subjects": len(full_exam_result["failed_subjects"]),
            "completion_status": "COMPLETE" if total_generated >= 100 else "PARTIAL",
            "generation_time": f"{elapsed_time:.1f}초"
        }
        
        # 최종 결과 출력
        print(f"\n" + "="*80)
        print(f"🎯 최종 결과: {total_generated}/100문제 생성 완료!")
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
        """100문제 시험을 JSON 파일로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"정보처리기사_50문제_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(exam_result, f, ensure_ascii=False, indent=2)
            
            total_questions = exam_result.get("total_questions", 0)
            print(f"\n" + "="*60)
            print(f"  정보처리기사 모의고사 ({total_questions}문제)가 '{filename}' 파일로 저장되었습니다!")
            print("="*60)
            
            # 과목별 요약 출력
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
        print(f"📊 전체 문제 수: {summary.get('actual_total', 0)}/{summary.get('target_total', 100)}문제")
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
        
        # 실패한 과목이 있는 경우 안내
        failed_subjects = exam_result.get("failed_subjects", [])
        if failed_subjects:
            print(f"\n⚠️  실패한 과목이 있습니다:")
            for failed in failed_subjects:
                print(f"   - {failed['subject']}: PDF 문서에 관련 내용이 부족하거나 키워드가 적합하지 않을 수 있습니다.")
            print(f"   💡 해결 방안: 해당 과목의 상세한 PDF 자료를 추가하거나 다시 시도해보세요.")
        
        print()

    def generate_subject_quiz(self, subject_area: str, target_count: int = 20, difficulty: str = "중급") -> Dict[str, Any]:
        """특정 과목의 문제를 생성 (비병렬 버전 - 호환성 유지)"""
        return self.generate_subject_quiz_parallel(subject_area, target_count, difficulty)

    def generate_full_exam(self, difficulty: str = "중급", parallel: bool = True) -> Dict[str, Any]:
        """정보처리기사 전체 50문제를 생성"""
        if parallel:
            return self.generate_full_exam_parallel(difficulty)
        else:
            # 기존 순차 처리 방식도 지원 (필요시)
            return self._generate_full_exam_sequential(difficulty)

    def _generate_full_exam_sequential(self, difficulty: str = "중급") -> Dict[str, Any]:
        """순차 처리 방식 (기존 방식)"""
        if not self._build_vectorstore_from_all_pdfs():
            return {"error": f"'{self.data_folder}' 폴더에 PDF가 없어 문제를 생성할 수 없습니다."}
        
        print("\n" + "="*80)
        print("  정보처리기사 50문제 자동 생성을 시작합니다! (순차 처리 모드)")
        print("="*80)
        
        start_time = time.time()
        
        full_exam_result = {
            "exam_title": "정보처리기사 모의고사",
            "total_questions": 0,
            "difficulty": difficulty,
            "subjects": {},
            "all_questions": [],
            "generation_summary": {},
            "failed_subjects": []
        }
        
        total_generated = 0
        
        for i, (subject_area, subject_info) in enumerate(self.SUBJECT_AREAS.items(), 1):
            target_count = subject_info["count"]
            
            print(f"\n📚 [{i}/5] {subject_area} 과목 시작 (목표: {target_count}문제)")
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
        print(f"🎯 최종 결과: {total_generated}/50문제 생성 완료!")
        print(f"⏱️  소요 시간: {elapsed_time:.1f}초")
        print("="*80)
        
        return full_exam_result


# --- 대화형 인터페이스 ---
def interactive_menu(rag_system):
    """사용자와 상호작용하는 메뉴 시스템"""
    while True:
        print("\n" + "="*70)
        print("  정보처리기사 50문제 자동 출제 시스템")
        print("  [병렬 처리 & 증분 생성 지원]")
        print("="*70)
        print("1. 정보처리기사 전체 50문제 생성 (병렬 처리 - 추천)")
        print("2. 정보처리기사 전체 50문제 생성 (순차 처리)")
        print("3. 특정 과목만 문제 생성")
        print("4. 사용 가능한 PDF 목록 보기")
        print("0. 종료")
        print("-"*70)
        
        choice = input("선택하세요: ").strip()
        
        if choice == "1":
            difficulty_str = input("난이도를 입력하세요 (초급/중급/고급, 기본값: 중급): ").strip()
            difficulty = difficulty_str if difficulty_str in ["초급", "중급", "고급"] else "중급"
            
            print(f"\n정보처리기사 50문제 생성을 시작합니다. (난이도: {difficulty}, 병렬 처리)")
            print("병렬 처리로 더 빠르게 생성됩니다...")
            
            exam_result = rag_system.generate_full_exam(difficulty=difficulty, parallel=True)
            
            if "error" not in exam_result:
                rag_system.display_exam_summary(exam_result)
                
                if input("\n생성된 문제를 JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower() == 'y':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    default_filename = f"정보처리기사_100문제_{difficulty}_{timestamp}.json"
                    filename = input(f"저장할 파일명 (기본값: {default_filename}): ").strip() or default_filename
                    rag_system.save_exam_to_json(exam_result, filename)
            else:
                print(f"오류: {exam_result['error']}")
        
        elif choice == "2":
            difficulty_str = input("난이도를 입력하세요 (초급/중급/고급, 기본값: 중급): ").strip()
            difficulty = difficulty_str if difficulty_str in ["초급", "중급", "고급"] else "중급"
            
            print(f"\n정보처리기사 100문제 생성을 시작합니다. (난이도: {difficulty}, 순차 처리)")
            print("순차적으로 각 과목을 처리합니다...")
            
            exam_result = rag_system.generate_full_exam(difficulty=difficulty, parallel=False)
            
            if "error" not in exam_result:
                rag_system.display_exam_summary(exam_result)
                
                if input("\n생성된 문제를 JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower() == 'y':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    default_filename = f"정보처리기사_100문제_{difficulty}_{timestamp}.json"
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
                    
                    print(f"\n{selected_subject} 문제 {target_count}개를 생성합니다...")
                    
                    subject_result = rag_system.generate_subject_quiz(
                        subject_area=selected_subject,
                        target_count=target_count,
                        difficulty=difficulty
                    )
                    
                    if "error" not in subject_result:
                        questions = subject_result["questions"]
                        actual_count = len(questions)
                        
                        print(f"\n[{selected_subject}] {actual_count}/{target_count}문제 생성 완료!")
                        
                        # 생성된 문제 미리보기
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
                        
                        # 파일 저장 옵션
                        if input("\n생성된 문제를 JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower() == 'y':
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            default_filename = f"{selected_subject}_{actual_count}문제_{difficulty}_{timestamp}.json"
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
        print("  정보처리기사 50문제 자동 출제 시스템 초기화 중...")
        print("="*80)
        
        # max_workers 파라미터로 병렬 처리 워커 수 조정 가능
        rag_system = InfoProcessingExamRAG(data_folder="data", max_workers=3)
        
        print("\n[시스템 초기화 완료]")
        print(f"'{rag_system.data_folder}' 폴더에 정보처리기사 관련 PDF 파일을 넣어주세요.")
        print(f"병렬 처리 워커 수: {rag_system.max_workers}개")
        print("\n[정보처리기사 출제 과목]")
        for subject, info in rag_system.SUBJECT_AREAS.items():
            print(f"  - {subject}: {info['count']}문제")
        print(f"  총 50문제")
        
        interactive_menu(rag_system)
        
    except Exception as e:
        print(f"프로그램 실행 중 오류가 발생했습니다: {e}")
        print("Ollama 서버가 실행 중인지 확인해주세요.")


if __name__ == "__main__":
    main()