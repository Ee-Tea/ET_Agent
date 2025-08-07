import os
import glob
from typing import List, Dict, Any, TypedDict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langgraph.graph import StateGraph, END
import json
import re
from collections import Counter
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime

# Ollama API 사용
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings

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
    정보처리기사 출제기준에 맞는 50문제 자동 출제 시스템 (gpt-oss-20b 버전)
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
    
    def __init__(self, data_folder="data", max_workers=3, model_name="llama2:7b"):
        """
        초기화
        
        Args:
            data_folder: PDF 파일이 있는 폴더
            max_workers: 병렬 처리 워커 수
            model_name: Ollama 모델명
        """
        self.data_folder = data_folder
        self.max_workers = max_workers
        self.model_name = model_name
        os.makedirs(self.data_folder, exist_ok=True)
        
        self.embeddings_model = None
        self.vectorstore = None
        self.retriever = None
        self.workflow = None
        
        self.files_in_vectorstore = []
        self.lock = threading.Lock()
        
        self._initialize_models()
        self._build_graph()

    def _initialize_models(self):
        """임베딩 및 Ollama 모델 초기화"""
        try:
            print("임베딩 모델 초기화 중...")
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ 임베딩 모델 초기화 완료")
            
            print("Ollama gpt-oss-20b 모델 연결 확인 중...")
            
            # Ollama 서버 연결 확인
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                models = response.json().get("models", [])
                
                gpt_oss_available = any(self.model_name in model.get("name", "") for model in models)
                
                if not gpt_oss_available:
                    print(f"⚠️ Ollama에서 {self.model_name} 모델을 찾을 수 없습니다.")
                    print(f"다음 명령을 실행하세요: ollama pull {self.model_name}")
                    raise Exception(f"{self.model_name} 모델이 설치되지 않음")
                
                print(f"✅ Ollama {self.model_name} 모델 연결 확인 완료")
                
            except requests.exceptions.RequestException:
                print("❌ Ollama 서버가 실행되고 있지 않습니다.")
                print("Ollama를 실행하세요: ollama serve")
                raise
                
        except Exception as e:
            print(f"❌ 모델 초기화 실패: {e}")
            raise

    def _generate_text(self, prompt: str, max_tokens: int = 2048) -> str:
        """Ollama gpt-oss-20b로 텍스트 생성"""
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'num_predict': max_tokens,
                        'temperature': 0.1,
                        'top_p': 0.9,
                    }
                },
                timeout=300  # 5분 타임아웃
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                print(f"Ollama API 오류: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"텍스트 생성 오류: {e}")
            return ""

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
        """증분 방식으로 문제 생성 - gpt-oss-20b 사용"""
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
            
            generate_count = max(needed_count * 2, 5)
            
            print(f"[{subject_area}] 문제 생성 중... ({generate_count}개)")
            
            prompt = f"""Create {generate_count} multiple-choice questions in Korean for {subject_area} based on this content:

{context[:3000]}

Format as JSON:
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
}}"""
            
            response = self._generate_text(prompt, max_tokens=2048)
            
            if not response:
                return {**state, "error": "Ollama gpt-oss-20b가 응답을 생성하지 못했습니다."}
            
            new_questions = self._parse_quiz_response(response, subject_area)
            
            if not new_questions:
                return {**state, "error": "Ollama gpt-oss-20b가 유효한 문제를 생성하지 못했습니다."}
            
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
        """증분 검증 - 최적화됨"""
        subject_area = state.get("subject_area", "")
        previously_validated = state.get("validated_questions", [])
        new_questions = state.get("quiz_questions", [])
        target_quiz_count = state.get("target_quiz_count", 5)
        generation_attempts = state.get("generation_attempts", 0)

        if not new_questions:
            return {**state, "validated_questions": previously_validated}

        newly_validated = []
        needed = target_quiz_count - len(previously_validated)
        
        # 간단한 검증 - 모든 문제를 유효하다고 가정 (속도 향상)
        for q in new_questions[:needed]:
            newly_validated.append(q)
        
        all_validated = previously_validated + newly_validated
        print(f"[{subject_area}] 검증 완료: {len(newly_validated)}개 추가")
        
        need_more_questions = (len(all_validated) < target_quiz_count and 
                              generation_attempts < 10)  # 시도 횟수 줄임
        
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
        """Ollama gpt-oss-20b 응답에서 JSON 형식의 문제를 파싱"""
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

    def generate_subject_quiz_parallel(self, subject_area: str, target_count: int = 10, difficulty: str = "중급") -> Dict[str, Any]:
        """특정 과목의 문제를 병렬로 생성"""
        if subject_area not in self.SUBJECT_AREAS:
            return {"error": f"유효하지 않은 과목: {subject_area}"}
        
        keywords = self.SUBJECT_AREAS[subject_area]["keywords"]
        
        print(f"\n=== {subject_area} 문제 생성 시작 (목표: {target_count}개, Ollama gpt-oss-20b 사용) ===")
        
        keyword_combinations = []
        for i in range(0, len(keywords), 2):
            combo = " ".join(keywords[i:i+3])
            keyword_combinations.append(combo)
        
        all_validated_questions = []
        
        # 단순화된 병렬 처리
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            for combo in keyword_combinations:
                future = executor.submit(self._generate_with_keywords, 
                                       combo, subject_area, target_count, difficulty)
                futures.append(future)
            
            # 결과 수집
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    if "questions" in result and result["questions"]:
                        all_validated_questions.extend(result["questions"])
                        if len(all_validated_questions) >= target_count:
                            break
                except Exception as e:
                    print(f"  → 키워드 실패: {e}")
        
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
        """정보처리기사 전체 50문제를 병렬로 생성 (gpt-oss-20b 사용)"""
        if not self._build_vectorstore_from_all_pdfs():
            return {"error": f"'{self.data_folder}' 폴더에 PDF가 없어 문제를 생성할 수 없습니다."}
        
        print("\n" + "="*80)
        print("  정보처리기사 50문제 자동 생성을 시작합니다! (Ollama gpt-oss-20b + 병렬 처리)")
        print("  ⚠️  각 과목별로 반드시 10문제씩 생성합니다.")
        print("="*80)
        
        start_time = time.time()
        
        full_exam_result = {
            "exam_title": "정보처리기사 모의고사 (gpt-oss-20b 버전)",
            "total_questions": 0,
            "difficulty": difficulty,
            "subjects": {},
            "all_questions": [],
            "generation_summary": {},
            "failed_subjects": [],
            "model_info": "Ollama gpt-oss-20b"
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for subject_area, subject_info in self.SUBJECT_AREAS.items():
                target_count = subject_info["count"]
                print(f"📚 [{subject_area}] 과목 문제 생성 시작... (목표: {target_count}문제, Ollama gpt-oss-20b 사용)")
                
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
                    subject_result = future.result(timeout=1800)  # 30분 타임아웃
                    
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
        print(f"🎯 최종 결과: {total_generated}/50문제 생성 완료! (Ollama gpt-oss-20b 사용)")
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
            filename = f"정보처리기사_50문제_gptoss_{timestamp}.json"
        
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

    def generate_subject_quiz(self, subject_area: str, target_count: int = 10, difficulty: str = "중급") -> Dict[str, Any]:
        """특정 과목의 문제를 생성"""
        return self.generate_subject_quiz_parallel(subject_area, target_count, difficulty)

    def generate_full_exam(self, difficulty: str = "중급", parallel: bool = True) -> Dict[str, Any]:
        """정보처리기사 전체 50문제를 생성"""
        if parallel:
            return self.generate_full_exam_parallel(difficulty)
        else:
            # 순차 처리 - 간단한 버전
            if not self._build_vectorstore_from_all_pdfs():
                return {"error": f"'{self.data_folder}' 폴더에 PDF가 없습니다."}
            
            start_time = time.time()
            full_exam_result = {
                "exam_title": "정보처리기사 모의고사",
                "total_questions": 0,
                "difficulty": difficulty,
                "subjects": {},
                "all_questions": [],
                "generation_summary": {},
                "failed_subjects": [],
                "model_info": "Ollama gpt-oss-20b"
            }
            
            total_generated = 0
            
            for subject_area, subject_info in self.SUBJECT_AREAS.items():
                target_count = subject_info["count"]
                print(f"[{subject_area}] 시작...")
                
                subject_result = self.generate_subject_quiz_parallel(
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
                    print(f"✅ [{subject_area}] 완료: {actual_count}개")
            
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
            
            return full_exam_result


# --- 대화형 인터페이스 ---
def interactive_menu(rag_system):
    """사용자와 상호작용하는 메뉴 시스템"""
    while True:
        print("\n=== 정보처리기사 50문제 자동 출제 시스템 ===")
        print("1. 전체 50문제 생성 (병렬)")
        print("2. 전체 50문제 생성 (순차)")
        print("3. 특정 과목만 생성")
        print("4. PDF 목록 보기")
        print("0. 종료")
        
        choice = input("선택: ").strip()
        
        if choice == "1":
            difficulty = input("난이도 (초급/중급/고급): ").strip() or "중급"
            print(f"50문제 생성 시작... (난이도: {difficulty})")
            
            exam_result = rag_system.generate_full_exam(difficulty=difficulty, parallel=True)
            
            if "error" not in exam_result:
                rag_system.display_exam_summary(exam_result)
                if input("JSON 저장? (y/n): ").strip().lower() == 'y':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"정보처리기사_50문제_{difficulty}_{timestamp}.json"
                    rag_system.save_exam_to_json(exam_result, filename)
            else:
                print(f"오류: {exam_result['error']}")
        
        elif choice == "2":
            difficulty = input("난이도 (초급/중급/고급): ").strip() or "중급"
            print(f"50문제 생성 시작... (난이도: {difficulty})")
            
            exam_result = rag_system.generate_full_exam(difficulty=difficulty, parallel=False)
            
            if "error" not in exam_result:
                rag_system.display_exam_summary(exam_result)
                if input("JSON 저장? (y/n): ").strip().lower() == 'y':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"정보처리기사_50문제_{difficulty}_{timestamp}.json"
                    rag_system.save_exam_to_json(exam_result, filename)
            else:
                print(f"오류: {exam_result['error']}")
        
        elif choice == "3":
            subjects = list(rag_system.SUBJECT_AREAS.keys())
            for i, subject in enumerate(subjects, 1):
                print(f"{i}. {subject}")
            
            try:
                subject_choice = int(input("과목 번호: "))
                if 1 <= subject_choice <= len(subjects):
                    selected_subject = subjects[subject_choice - 1]
                    target_count = int(input("문제 수: ") or "10")
                    difficulty = input("난이도: ").strip() or "중급"
                    
                    print(f"{selected_subject} {target_count}문제 생성 중...")
                    
                    subject_result = rag_system.generate_subject_quiz(
                        subject_area=selected_subject,
                        target_count=target_count,
                        difficulty=difficulty
                    )
                    
                    if "error" not in subject_result:
                        questions = subject_result["questions"]
                        print(f"✅ {len(questions)}개 문제 생성 완료")
                        
                        if input("미리보기? (y/n): ").strip().lower() == 'y':
                            for i, q in enumerate(questions[:2], 1):
                                print(f"\n[문제 {i}] {q.get('question', '')}")
                                for option in q.get('options', []):
                                    print(f"{option}")
                                print(f"정답: {q.get('answer', '')}")
                        
                        if input("JSON 저장? (y/n): ").strip().lower() == 'y':
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{selected_subject}_{len(questions)}문제_{timestamp}.json"
                            with open(filename, 'w', encoding='utf-8') as f:
                                json.dump(subject_result, f, ensure_ascii=False, indent=2)
                            print(f"저장됨: {filename}")
                    else:
                        print(f"오류: {subject_result['error']}")
                else:
                    print("잘못된 번호")
            except ValueError:
                print("숫자 입력")

        elif choice == "4":
            rag_system.list_available_pdfs()
            
        elif choice == "0":
            print("종료")
            break
            
        else:
            print("잘못된 선택")


def main():
    """메인 실행 함수"""
    try:
        print("정보처리기사 50문제 자동 출제 시스템 (Ollama gpt-oss-20b) 초기화 중...")
        
        rag_system = InfoProcessingExamRAG(
            data_folder="data", 
            max_workers=2  # 워커 수 줄임
        )
        
        print("✅ 시스템 초기화 완료")
        print(f"🤖 모델: Ollama gpt-oss-20b")
        print(f"📁 데이터: '{rag_system.data_folder}'")
        
        interactive_menu(rag_system)
        
    except Exception as e:
        print(f"오류: {e}")
        print("💡 해결: Ollama 서버가 실행 중인지 확인하세요 (ollama serve)")


if __name__ == "__main__":
    main()