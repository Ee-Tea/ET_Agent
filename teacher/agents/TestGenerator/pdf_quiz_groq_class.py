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

# Groq 관련 임포트
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# .env 파일 로드
load_dotenv()

class BaseAgent(ABC):
    """
    모든 에이전트가 상속받아야 하는 기본 추상 클래스입니다.
    모든 에이전트는 'execute' 메서드를 구현해야 합니다.
    """
    
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
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트의 주된 로직을 실행하는 메서드입니다.
        
        Args:
            input_data (Dict[str, Any]): 에이전트 실행에 필요한 입력 데이터입니다.
            
        Returns:
            Dict[str, Any]: 에이전트 실행 결과 데이터입니다.
        """
        pass


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
    정보처리기사 출제기준에 맞는 25문제 자동 출제 에이전트 (순차 처리 버전)
    """
    
    # 정보처리기사 5개 과목 정의
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
    
    def __init__(self, data_folder=None, groq_api_key=None):
        if data_folder is None:
            # 현재 파일 기준으로 data 폴더 설정
            base_dir = Path(__file__).resolve().parent  # TestGenerator 폴더
            data_folder = base_dir / "data"
        self.data_folder = Path(data_folder)
        os.makedirs(self.data_folder, exist_ok=True)
        
        # Groq API 키 설정
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        elif not os.getenv("GROQ_API_KEY"):
            raise ValueError("Groq API 키가 필요합니다.")
        
        self.embeddings_model = None
        self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.workflow = None
        
        self.files_in_vectorstore = []
        
        self._initialize_models()
        self._build_graph()

    @property
    def name(self) -> str:
        """에이전트의 고유한 이름을 반환합니다."""
        return "InfoProcessingExamAgent"

    @property
    def description(self) -> str:
        """에이전트의 역할에 대한 설명을 반환합니다."""
        return "정보처리기사 출제기준에 맞는 25문제를 자동으로 생성하는 에이전트입니다. PDF 문서를 기반으로 5개 과목별로 문제를 생성합니다."

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트의 주된 로직을 실행하는 메서드입니다.
        
        Args:
            input_data (Dict[str, Any]): 에이전트 실행에 필요한 입력 데이터
                - mode: "full_exam" 또는 "subject_quiz"
                - difficulty: "초급", "중급", "고급" (기본값: "중급")
                - subject_area: 특정 과목명 (subject_quiz 모드일 때)
                - target_count: 생성할 문제 수 (subject_quiz 모드일 때)
                - save_to_file: JSON 파일 저장 여부 (기본값: False)
                - filename: 저장할 파일명 (선택사항)
                
        Returns:
            Dict[str, Any]: 에이전트 실행 결과 데이터
                - success: 성공 여부
                - result: 생성된 시험 데이터
                - error: 오류 메시지 (실패시)
                - file_path: 저장된 파일 경로 (저장시)
        """
        try:
            # 입력 데이터 검증 및 기본값 설정
            # 사전 사용자 입력에서 추출해야 함. 
            mode = input_data.get("mode", "full_exam") #full_exam / subject_quiz
            difficulty = input_data.get("difficulty", "중급") # 초급, 중급, 고급
            save_to_file = input_data.get("save_to_file", False) #json 파일로 저장 여부
            filename = input_data.get("filename") # "저장할 파일명" (선택사항)
            
            # 벡터 스토어 초기화 확인
            if not self._build_vectorstore_from_all_pdfs():
                return {
                    "success": False,
                    "error": f"'{self.data_folder}' 폴더에 PDF 파일이 없습니다."
                }
            
            if mode == "full_exam":
                # 전체 25문제 생성
                result = self._generate_full_exam(difficulty)
            elif mode == "subject_quiz":
                # 특정 과목 문제 생성
                subject_area = input_data.get("subject_area") # 과목 명 (한 번에 하나 밖에 안됨)
                target_count = input_data.get("target_count", 5) # 문제 수
                
                if not subject_area or subject_area not in self.SUBJECT_AREAS:
                    return {
                        "success": False,
                        "error": f"유효하지 않은 과목명입니다. 가능한 과목: {list(self.SUBJECT_AREAS.keys())}"
                    }
                
                result = self._generate_subject_quiz(subject_area, target_count, difficulty)
            else:
                return {
                    "success": False,
                    "error": "유효하지 않은 모드입니다. 'full_exam' 또는 'subject_quiz'를 사용하세요."
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

    def _initialize_models(self):
        """임베딩 및 LLM 모델 초기화"""
        try:
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.llm = ChatGroq(
                model="moonshotai/kimi-k2-instruct",
                temperature=0.0,
                max_tokens=2048,
                timeout=120,
                max_retries=3
            )
            
            # 연결 테스트
            test_response = self.llm.invoke("안녕하세요")
            
        except Exception as e:
            raise ValueError(f"모델 초기화 중 오류 발생: {e}")

    def _build_vectorstore_from_all_pdfs(self) -> bool:
        """PDF를 로드하여 벡터 스토어를 생성/업데이트"""
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
            except Exception as e:
                continue
        
        if not all_documents:
            return False

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_documents)
        
        self.vectorstore = FAISS.from_documents(splits, self.embeddings_model)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        
        self.files_in_vectorstore = pdf_files
        return True

    def get_pdf_files(self) -> List[str]:
        """data 폴더에서 PDF 파일 목록 가져오기"""
        return glob.glob(os.path.join(self.data_folder, "*.pdf"))

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
        key_sents = []
        for doc in documents:
            lines = doc.page_content.split("\n")
            for line in lines:
                line = line.strip()
                if len(line) > 100 or any(k in line for k in ["정의", "특징", "종류", "예시", "원리", "구성", "절차", "장점", "단점"]):
                    key_sents.append(line)
        
        context = "\n".join(key_sents)[:2000]
        return {**state, "context": context}

    def _generate_quiz_incremental(self, state: GraphState) -> GraphState:
        """문제 생성"""
        try:
            context = state["context"]
            target_quiz_count = state.get("target_quiz_count", 5)
            validated_questions = state.get("validated_questions", [])
            subject_area = state.get("subject_area", "")
            needed_count = target_quiz_count - len(validated_questions)
            
            if needed_count <= 0:
                return {**state, "quiz_questions": validated_questions}
            
            if not context.strip():
                return {**state, "error": "검색된 문서 내용이 없습니다."}
            
            # 정확히 필요한 개수만 생성하여 속도와 일관성 개선
            generate_count = max(min(needed_count, 10), 1)
            
            prompt_template = PromptTemplate(
                input_variables=["context", "subject_area", "needed_count"],
                template="""아래 문서 내용을 바탕으로 {subject_area} 과목의 객관식 문제 {needed_count}개를 반드시 생성하세요.\n각 문제는 4지선다, 정답 번호와 간단한 해설을 포함해야 하며, 반드시 JSON만 출력하세요.\n\n[문서]\n{context}\n\n[출력 예시]\n{{\n  \"questions\": [\n    {{\n      \"question\": \"문제 내용\",\n      \"options\": [\"선택지1\", \"선택지2\", \"선택지3\", \"선택지4\"],\n      \"answer\": \"정답 번호(예: 1)\",\n      \"explanation\": \"간단한 해설\"\n    }}\n  ]\n}}\n"""
            )
            
            prompt = prompt_template.format(
                context=context,
                subject_area=subject_area,
                needed_count=generate_count
            )
            
            self.llm.temperature = 0.2
            self.llm.max_tokens = 1024
            response = self.llm.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            new_questions = self._parse_quiz_response(response_content, subject_area)
            if not new_questions:
                # 재시도 유도: 시도 횟수 증가시켜 루프가 계속 돌도록 함
                return {
                    **state,
                    "quiz_questions": [],
                    "validated_questions": validated_questions,
                    "generation_attempts": state.get("generation_attempts", 0) + 1,
                    "error": "유효한 문제를 생성하지 못했습니다."
                }
            
            return {
                **state,
                "quiz_questions": new_questions,
                "validated_questions": validated_questions,
                "generation_attempts": state.get("generation_attempts", 0) + 1
            }
        except Exception as e:
            return {**state, "error": f"문제 생성 중 오류 발생: {e}"}

    def _validate_quiz_incremental(self, state: GraphState) -> GraphState:
        """증분 검증"""
        subject_area = state.get("subject_area", "")
        
        previously_validated = state.get("validated_questions", [])
        new_questions = state.get("quiz_questions", [])
        context = state.get("context", "")
        target_quiz_count = state.get("target_quiz_count", 5)
        generation_attempts = state.get("generation_attempts", 0)

        if not new_questions:
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
5. 사용자의 질문을 파악하고 정답을 추론할 수 있는 문제인가?

[응답 형식]
'is_valid'(boolean)와 'reason'(한국어 설명) 키를 가진 JSON 객체로만 응답해주세요.

Your JSON response:"""
        )
        
        needed = target_quiz_count - len(previously_validated)
        
        for i, q in enumerate(new_questions):
            if len(newly_validated) >= needed:
                break
                
            try:
                question_str = json.dumps(q, ensure_ascii=False, indent=2)
                prompt = validation_prompt_template.format(context=context[:4000], question_data=question_str)
                
                response = self.llm.invoke(prompt)
                response_str = response.content if hasattr(response, 'content') else str(response)
                
                match = re.search(r'\{.*\}', response_str, re.DOTALL)
                if not match:
                    continue

                validation_result = json.loads(match.group(0))

                if validation_result.get("is_valid") is True:
                    newly_validated.append(q)

            except Exception:
                continue
        
        all_validated = previously_validated + newly_validated
        
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
        
        if validated_count >= target_count:
            return "complete"
        elif generation_attempts < 15:
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
        if subject_area not in self.SUBJECT_AREAS:
            return {"error": f"유효하지 않은 과목: {subject_area}"}
        
        keywords = self.SUBJECT_AREAS[subject_area]["keywords"]
        
        all_validated_questions = []
        
        max_rounds = 10
        current_round = 0
        
        while len(all_validated_questions) < target_count and current_round < max_rounds:
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
            
            if current_round < max_rounds and len(all_validated_questions) < target_count:
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
            "model_info": "Groq llama-4-scout-17b-16e-instruct"
        }
        
        total_generated = 0
        
        for i, (subject_area, subject_info) in enumerate(self.SUBJECT_AREAS.items(), 1):
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
        save_dir = "C:\\ET_Agent\\teacher\\TestGenerator\\test"
        os.makedirs(save_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if "exam_title" in exam_result:
                filename = f"정보처리기사_25문제_{timestamp}.json"
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


# 사용 예시
def example_usage():
    """에이전트 사용 예시"""
    try:
        # 에이전트 초기화
        agent = InfoProcessingExamAgent(
            data_folder="C:\\ET_Agent\\teacher\\TestGenerator\\data"
        )
        
        print(f"에이전트명: {agent.name}")
        print(f"설명: {agent.description}")
        
        # 전체 25문제 생성
        input_data = {
            "mode": "full_exam",
            "difficulty": "중급",
            "save_to_file": True
        }
        
        result = agent.execute(input_data)
        
        if result["success"]:
            exam_data = result["result"]
            print(f"성공! 총 {exam_data['total_questions']}문제 생성")
            if "file_path" in result:
                print(f"파일 저장: {result['file_path']}")
        else:
            print(f"실패: {result['error']}")
            
        # 특정 과목 문제 생성
        input_data = {
            "mode": "subject_quiz",
            "subject_area": "소프트웨어설계",
            "target_count": 3,
            "difficulty": "중급",
            "save_to_file": False
        }
        
        result = agent.execute(input_data)
        
        if result["success"]:
            subject_data = result["result"]
            print(f"성공! {subject_data['subject_area']} {subject_data['quiz_count']}문제 생성")
        else:
            print(f"실패: {result['error']}")
            
    except Exception as e:
        print(f"에이전트 초기화 실패: {e}")


# 대화형 인터페이스 (옵션)
def interactive_menu():
    """에이전트를 활용한 대화형 메뉴 시스템"""
    try:
        agent = InfoProcessingExamAgent(
            data_folder="C:\\ET_Agent\\teacher\\TestGenerator\\data"
        )
        
        print(f"\n{agent.name} 초기화 완료")
        print(f"설명: {agent.description}")
        
        while True:
            print("\n" + "="*60)
            print("  정보처리기사 25문제 자동 출제 에이전트")
            print("="*60)
            print("1. 전체 25문제 생성")
            print("2. 특정 과목만 문제 생성")
            print("3. 사용 가능한 PDF 목록 보기")
            print("0. 종료")
            print("-"*60)
            
            choice = input("선택하세요: ").strip()
            
            if choice == "1":
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
                    
                    if "save_error" in result:
                        print(f"⚠️ 저장 오류: {result['save_error']}")
                else:
                    print(f"❌ 실패: {result['error']}")
            
            elif choice == "2":
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
                            
                            # 문제 미리보기
                            questions = subject_data.get("questions", [])
                            if questions and input("\n생성된 문제를 미리보시겠습니까? (y/n): ").strip().lower() == 'y':
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
                            
                            if "file_path" in result:
                                print(f"📁 저장 경로: {result['file_path']}")
                            
                            if "save_error" in result:
                                print(f"⚠️ 저장 오류: {result['save_error']}")
                        else:
                            print(f"❌ 실패: {result['error']}")
                    else:
                        print("유효하지 않은 과목 번호입니다.")
                except ValueError:
                    print("숫자를 입력해주세요.")
            
            elif choice == "3":
                pdf_files = agent.get_pdf_files()
                if pdf_files:
                    print(f"\n=== '{agent.data_folder}' 폴더의 PDF 파일 목록 ===")
                    for i, file_path in enumerate(pdf_files, 1):
                        filename = os.path.basename(file_path)
                        file_size = os.path.getsize(file_path) / 1024
                        print(f"{i}. {filename} ({file_size:.1f} KB)")
                else:
                    print(f"'{agent.data_folder}' 폴더에 PDF 파일이 없습니다.")
            
            elif choice == "0":
                print("에이전트를 종료합니다.")
                break
            
            else:
                print("잘못된 선택입니다. 0~3 중에서 선택해주세요.")
    
    except Exception as e:
        print(f"에이전트 초기화 실패: {e}")


def main():
    """메인 실행 함수"""
    # Groq API 키 확인
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    # 사용 방법 선택
    print("정보처리기사 문제 생성 에이전트")
    print("1. 대화형 인터페이스 사용")
    print("2. 코드 예시 실행")
    
    choice = input("선택하세요 (1 또는 2): ").strip()
    
    if choice == "1":
        interactive_menu()
    elif choice == "2":
        example_usage()
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()