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

class GraphState(TypedDict):
    """그래프 상태 정의"""
    query: str
    documents: List[Document]
    context: str
    quiz_questions: List[Dict[str, Any]] # 생성 및 검증을 거친 최종 문제
    quiz_count: int
    difficulty: str
    error: str
    used_sources: List[str] # 어떤 파일이 사용되었는지 추적
    generation_attempts: int # 생성 시도 횟수 추가
    target_quiz_count: int # 목표 문제 수 추가

class PDFQuizRAG:
    """
    키워드를 먼저 입력받아 관련된 PDF를 자동으로 찾고,
    생성된 문제를 검증하여 객관식 문제를 출제하는 RAG 시스템
    """
    def __init__(self, data_folder="data"):
        """초기화"""
        self.data_folder = data_folder
        os.makedirs(self.data_folder, exist_ok=True)
        
        self.embeddings_model = None
        self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.workflow = None
        
        # 벡터스토어에 포함된 파일 목록을 추적
        self.files_in_vectorstore = []
        
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
            # Ollama 연결 테스트
            self.llm.invoke("안녕하세요")
            print("LLM 모델 초기화 및 연결 확인 완료.")
            
        except Exception as e:
            print(f"모델 초기화 중 심각한 오류 발생: {e}")
            print("Ollama 서버가 실행 중인지 확인해주세요. (명령어: ollama serve)")
            raise

    def _build_vectorstore_from_all_pdfs(self) -> bool:
        """
        data 폴더의 모든 PDF를 로드하여 벡터 스토어를 생성/업데이트합니다.
        파일 변경이 없으면 다시 빌드하지 않습니다.
        """
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
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
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
            documents = self.retriever.invoke(query)
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

    def _generate_quiz(self, state: GraphState) -> GraphState:
        """객관식 문제 생성 노드"""
        try:
            context = state["context"]
            target_quiz_count = state.get("target_quiz_count", 3)
            difficulty = state.get("difficulty", "중급")
            generation_attempts = state.get("generation_attempts", 0)
            existing_questions = state.get("quiz_questions", [])
            
            # 이미 충분한 문제가 있다면 추가 생성하지 않음
            if len(existing_questions) >= target_quiz_count:
                print(f"목표 문제 수({target_quiz_count}개)에 도달했습니다.")
                return state
            
            if not context.strip():
                return {**state, "error": "검색된 문서 내용이 없습니다. 다른 키워드로 시도해보세요."}
            
            # 필요한 문제 수 계산 (여분을 두어 검증 실패에 대비)
            needed_count = target_quiz_count - len(existing_questions)
            generate_count = max(needed_count * 2, 5)  # 검증 실패를 고려해 여분 생성
            
            print(f"\n문제 생성 중... (키워드: {state['query']}, 필요한 문제: {needed_count}개, 생성할 문제: {generate_count}개, 난이도: {difficulty}, 시도: {generation_attempts + 1})")
            
            prompt_template = PromptTemplate(
                input_variables=["context", "quiz_count", "difficulty"],
                template="""You are a machine that only outputs a single, valid JSON object. Do not add any text before or after the JSON.
Based *only* on the provided document, select a key sentence or core concept, and create {quiz_count} multiple-choice questions from it in Korean at the {difficulty} level.
The question, the correct answer, and the explanation **MUST** be directly verifiable from the provided document content.
DO NOT create questions about topics not explicitly mentioned in the document.
The quiz questions should be well-formed and easy to understand for a human.
Try to create questions about different topics or aspects covered in the document to ensure variety.

[Document]
{context}

[Required JSON Format]
```json
{{
  "questions": [
    {{
      "question": "문제 내용",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "answer": "정답 번호 (예: 1)",
      "explanation": "정답에 대한 간결한 해설"
    }}
  ]
}}
```

Your JSON output:"""
            )
            
            prompt = prompt_template.format(context=context[:4000], quiz_count=generate_count, difficulty=difficulty)
            response = self.llm.invoke(prompt)
            new_questions = self._parse_quiz_response(response)
            
            if not new_questions:
                return {**state, "error": "LLM이 유효한 문제를 생성하지 못했습니다."}
            
            print(f"새로 생성된 문제 {len(new_questions)}개.")
            
            # 기존 문제와 새 문제를 합침
            all_questions = existing_questions + new_questions
            
            return {
                **state, 
                "quiz_questions": all_questions,
                "generation_attempts": generation_attempts + 1
            }
        except Exception as e:
            return {**state, "error": f"문제 생성 중 오류 발생: {e}"}

    def _validate_quiz(self, state: GraphState) -> GraphState:
        """생성된 문제가 컨텍스트에 부합하는지 검증하는 노드"""
        print("\n생성된 문제 검증을 시작합니다...")
        validated_questions = []
        questions = state.get("quiz_questions", [])
        context = state.get("context", "")
        target_quiz_count = state.get("target_quiz_count", 3)
        generation_attempts = state.get("generation_attempts", 0)

        if not questions:
            print("검증할 문제가 없습니다.")
            return state

        validation_prompt_template = PromptTemplate(
            input_variables=["context", "question_data"],
            template="""당신은 사실 관계를 꼼꼼하게 확인하는 검증 전문가입니다.
오직 아래 제공된 '문서 내용'에만 근거하여 다음 '퀴즈 문제'를 평가해주세요.

[문서 내용]
{context}

[평가할 퀴즈 문제]
{question_data}

[평가 기준]
1. 이 질문과 정답이 '문서 내용'의 **특정 문장**에 의해 직접적으로 뒷받침됩니까?
2. 퀴즈 문제에 제시된 '정답'이 '문서 내용'에 의해 한 치의 오차 없이 명확하게 뒷받침됩니까?
3. '해설'이 '문서 내용'의 정보를 정확하고 간결하게 요약하고 있습니까?
4. '정답'에 해당하는 내용이 '문서 내용'에 실제로 존재합니까?

[응답 형식]
'is_valid'(boolean)와 'reason'(한국어로 된 간략한 설명) 두 개의 키를 가진 단일 JSON 객체로만 응답해주세요.
is_valid가 false인 경우, 그 이유를 구체적으로 작성해주세요.
예시: {{"is_valid": true, "reason": "모든 정보가 문서 내용에 명확히 나타나 있습니다."}}

Your JSON response:"""
        )
        
        for q in questions:
            try:
                question_str = json.dumps(q, ensure_ascii=False, indent=2)
                prompt = validation_prompt_template.format(context=context[:4000], question_data=question_str)
                response_str = self.llm.invoke(prompt)
                
                match = re.search(r'\{.*\}', response_str, re.DOTALL)
                if not match:
                    print(f"  - [검증 오류] 문제 \"{q.get('question', '')[:30]}...\" - LLM이 유효한 JSON을 반환하지 않음.")
                    continue

                validation_result = json.loads(match.group(0))

                if validation_result.get("is_valid") is True:
                    print(f"  - [VALID] 문제: \"{q.get('question', '')[:30]}...\"")
                    validated_questions.append(q)
                    
                    # 목표 수에 도달하면 더 이상 검증하지 않음
                    if len(validated_questions) >= target_quiz_count:
                        print(f"  - 목표 문제 수({target_quiz_count}개)에 도달했습니다. 검증을 종료합니다.")
                        break
                else:
                    reason = validation_result.get('reason', '이유 없음')
                    print(f"  - [INVALID] 문제: \"{q.get('question', '')[:30]}...\" (사유: {reason})")

            except Exception as e:
                print(f"  - [검증 오류] 문제 \"{q.get('question', '')[:30]}...\" - {e}")
        
        print(f"\n검증 완료: 총 {len(questions)}개 중 {len(validated_questions)}개 통과 (목표: {target_quiz_count}개)")
        
        # 목표 수에 못 미치고 재시도 가능한 경우 추가 생성 플래그 설정
        need_more_questions = (len(validated_questions) < target_quiz_count and 
                              generation_attempts < 3)  # 최대 3번까지 시도
        
        return {
            **state, 
            "quiz_questions": validated_questions,
            "need_more_questions": need_more_questions
        }

    def _check_completion(self, state: GraphState) -> str:
        """문제 생성 완료 여부를 체크하는 조건부 노드"""
        validated_count = len(state.get("quiz_questions", []))
        target_count = state.get("target_quiz_count", 3)
        need_more = state.get("need_more_questions", False)
        generation_attempts = state.get("generation_attempts", 0)
        
        if validated_count >= target_count:
            print(f"목표 문제 수({target_count}개)에 도달했습니다. 완료!")
            return "complete"
        elif need_more and generation_attempts < 3:
            print(f"추가 문제 생성이 필요합니다. (현재: {validated_count}개, 목표: {target_count}개, 시도: {generation_attempts}회)")
            return "generate_more"
        else:
            if generation_attempts >= 3:
                print(f"최대 시도 횟수(3회)에 도달했습니다. 현재까지 생성된 {validated_count}개 문제로 완료합니다.")
            return "complete"

    def _parse_quiz_response(self, response: str) -> List[Dict[str, Any]]:
        """LLM 응답에서 JSON 형식의 문제를 파싱 (더욱 강력하게)"""
        try:
            json_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1)
            else:
                json_str_match = re.search(r'\{.*\}', response.strip(), re.DOTALL)
                if not json_str_match:
                    print(f"JSON 파싱 오류: 응답에서 JSON 형식을 찾을 수 없습니다.\nLLM 응답: \n{response}")
                    return []
                json_str = json_str_match.group(0)

            json_str = json_str.replace('\\u312f', '').replace('\\n', ' ')

            data = json.loads(json_str)
            
            if "questions" not in data or not isinstance(data["questions"], list):
                print("JSON 파싱 오류: 'questions' 키가 없거나 리스트 형식이 아닙니다.")
                return []
            
            for question in data["questions"]:
                if "options" in question and isinstance(question["options"], list):
                    numbered_options = []
                    for i, option_text in enumerate(question["options"], 1):
                        cleaned_text = re.sub(r'^\s*\d+\.\s*', '', option_text).strip()
                        numbered_options.append(f"  {i}. {cleaned_text}")
                    question["options"] = numbered_options
            
            return data.get("questions", [])
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}\nLLM 응답: \n{response}")
            return []
        except Exception as e:
            print(f"응답 파싱 중 알 수 없는 오류: {e}")
            return []

    def _build_graph(self):
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("generate_quiz", self._generate_quiz)
        workflow.add_node("validate_quiz", self._validate_quiz)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "prepare_context")
        workflow.add_edge("prepare_context", "generate_quiz")
        workflow.add_edge("generate_quiz", "validate_quiz")
        
        # 조건부 엣지: 검증 후 완료 여부 확인
        workflow.add_conditional_edges(
            "validate_quiz",
            self._check_completion,
            {
                "generate_more": "generate_quiz",  # 추가 생성 필요
                "complete": END  # 완료
            }
        )
        
        self.workflow = workflow.compile()

    def generate_quiz_by_keyword(self, topic: str, quiz_count: int = 3, difficulty: str = "중급") -> Dict[str, Any]:
        """키워드를 기반으로 전체 PDF에서 문제를 생성 및 검증하는 메인 함수"""
        if not self._build_vectorstore_from_all_pdfs():
            return {"error": f"'{self.data_folder}' 폴더에 PDF가 없어 문제를 생성할 수 없습니다."}
        
        try:
            print(f"\n키워드 '{topic}'에 대한 문서 검색 및 문제 생성을 시작합니다.")
            initial_state = {
                "query": topic, 
                "quiz_count": quiz_count, 
                "target_quiz_count": quiz_count,  # 목표 문제 수 저장
                "difficulty": difficulty,
                "generation_attempts": 0,
                "quiz_questions": []
            }
            result = self.workflow.invoke(initial_state)
            
            if result.get("error"):
                return {"error": result["error"]}
            
            final_questions = result["quiz_questions"]
            actual_count = len(final_questions)
            
            # 목표 수에 못 미쳤을 때 경고 메시지
            if actual_count < quiz_count:
                print(f"\n[주의] 목표 문제 수({quiz_count}개)보다 적은 {actual_count}개의 문제만 생성되었습니다.")
                print("더 구체적인 키워드를 사용하거나 관련된 PDF 문서를 추가해보세요.")
            
            return {
                "topic": topic,
                "difficulty": difficulty,
                "requested_count": quiz_count,
                "quiz_count": actual_count,
                "questions": final_questions,
                "used_sources": result.get("used_sources", [])
            }
        except Exception as e:
            return {"error": f"최상위 실행 오류: {e}"}

    def display_quiz(self, quiz_data: Dict[str, Any]):
        """결과를 화면에 보기 좋게 출력"""
        if "error" in quiz_data:
            print(f"\n[오류] {quiz_data['error']}")
            return
        
        print("\n" + "="*60)
        print(f"  키워드: '{quiz_data['topic']}' | 난이도: {quiz_data['difficulty']}")
        print(f"  요청 문제 수: {quiz_data.get('requested_count', 'N/A')}개 | 실제 생성 수: {quiz_data['quiz_count']}개")
        print("="*60)
        
        if quiz_data.get("used_sources"):
            print("  [참고한 PDF 파일]")
            for source in quiz_data["used_sources"]:
                print(f"  - {source}")
            print("-"*60)

        if not quiz_data.get('questions'):
            print("\n생성된 문제 중 최종 통과한 문제가 없습니다. 키워드를 더 구체적으로 입력하거나, 다른 PDF를 추가하여 시도해보세요.")
            return

        for i, q in enumerate(quiz_data['questions'], 1):
            print(f"\n[문제 {i}] {q.get('question', '질문 없음')}")
            for option in q.get('options', []):
                print(f"{option}")
            print(f"▶ 정답: {q.get('answer', '정답 없음')}")
            print(f"▶ 해설: {q.get('explanation', '해설 없음')}")
        print("\n" + "="*60)

    def save_quiz_to_file(self, quiz_data: Dict[str, Any], filename: str):
        """생성된 퀴즈를 JSON 파일로 저장"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(quiz_data, f, ensure_ascii=False, indent=2)
            print(f"\n퀴즈가 '{filename}' 파일로 성공적으로 저장되었습니다.")
        except Exception as e:
            print(f"파일 저장 오류: {e}")

# --- 대화형 인터페이스 ---
def interactive_menu(rag_system):
    """사용자와 상호작용하는 메뉴 시스템"""
    while True:
        print("\n" + "="*50)
        print("  PDF 기반 객관식 문제 자동 출제 시스템 (검증 기능 포함)")
        print("="*50)
        print("1. 객관식 문제 생성 (키워드 입력)")
        print("2. 사용 가능한 PDF 목록 보기")
        print("0. 종료")
        print("-"*50)
        
        choice = input("선택하세요: ").strip()
        
        if choice == "1":
            topic = input("문제 출제 키워드를 입력하세요: ").strip()
            if not topic:
                print("키워드를 입력해야 합니다.")
                continue
            
            try:
                count_str = input("출제할 문제 수를 입력하세요 (기본값: 3): ").strip()
                quiz_count = int(count_str) if count_str else 3
            except ValueError:
                quiz_count = 3
            
            difficulty_str = input("난이도를 입력하세요 (초급/중급/고급, 기본값: 중급): ").strip()
            difficulty = difficulty_str if difficulty_str in ["초급", "중급", "고급"] else "중급"
            
            quiz_result = rag_system.generate_quiz_by_keyword(
                topic=topic, quiz_count=quiz_count, difficulty=difficulty
            )
            
            rag_system.display_quiz(quiz_result)
            
            if "error" not in quiz_result and quiz_result.get("questions"):
                if input("\n생성된 문제를 파일로 저장하시겠습니까? (y/n): ").strip().lower() == 'y':
                    default_filename = f"quiz_{topic.replace(' ', '_')}.json"
                    filename = input(f"저장할 파일명을 입력하세요 (기본값: {default_filename}): ").strip() or default_filename
                    rag_system.save_quiz_to_file(quiz_result, filename)

        elif choice == "2":
            rag_system.list_available_pdfs()
        elif choice == "0":
            print("시스템을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 1, 2, 0 중에서 선택해주세요.")

def main():
    """메인 실행 함수"""
    try:
        rag_system = PDFQuizRAG(data_folder="data")
        print("\n[시스템 초기화 완료]")
        print(f"'{rag_system.data_folder}' 폴더에 퀴즈를 만들 PDF 파일을 넣어주세요.")
        interactive_menu(rag_system)
    except Exception as e:
        print(f"프로그램 실행 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()