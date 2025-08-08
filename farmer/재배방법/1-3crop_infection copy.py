import os
import re # 정규식 모듈 임포트
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- 1. 환경 설정 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

print(f"OPENAI_API_KEY 로드 완료: {OPENAI_API_KEY[:5]}...")

# 데이터 파일 경로 및 벡터 DB 저장 경로
PDF_FILE_PATHS = [
    'data/pdf/과수 병해충 (병충해).PDF',
    'data/pdf/채소병해충 (병충해).PDF'
]
VECTOR_DB_PATH = 'faiss_crop_pest_db'

# --- 2. LLM 및 프롬프트 설정 ---
llm = ChatGroq(model_name="llama3-70b-8192", 
               temperature=0.7, 
               api_key=OPENAI_API_KEY)

RAG_PROMPT_TEMPLATE = """
당신은 대한민국 농업 작물 병충해 전문가입니다. 

다음은 작물 재배 중 발생하는 병충해에 대한 정보를 담고 있습니다. 제공된 다음의 정보를 사용하여 사용자의 질문에 답변해 주세요.
{context}

당신이 지킬 규칙들은 다음과 같습니다:
- 제공된 정보에 없는 정보, 저의 상식, 추측, 거짓 정보 등은 절대 답변에 넣지 마세요.
- 모든 답변은 한글로만 답해야 합니다. 한글이 아니면 절대 출력하지 말고, 한글로만 답변을 완성해야 합니다.
- 각 병충해 단계나 설명은 반드시 "한 문장씩 줄바꿈"해서 써주세요.
- 답변이 불가능하거나 제공된 정보에 관련 정보가 없으면 "주어진 정보로는 답변할 수 없습니다."라고만 답해야 합니다.
- 친근하게 대화하듯, 자연스럽게 설명해주세요.

질문: {question}
답변:
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# --- 3. LangGraph 상태 정의 ---
class GraphState(TypedDict):
    """
    LangGraph 상태를 나타내는 TypedDict.
    """
    question: Optional[str]
    documents: Optional[List[Dict[str, Any]]]
    splits: Optional[List[Dict[str, Any]]]
    vectorstore: Optional[FAISS]
    context: Optional[str]
    answer: Optional[str]

# --- 4. 핵심 기능 함수 정의 ---

def load_csv_data(file_path: str) -> List[Dict[str, Any]]:
    """
    (함수 이름은 유지) PDF 파일들을 로드하여 문서 리스트(페이지 단위)를 반환합니다.
    """
    print("---기능: PDF 문서 로드 시작 (load_csv_data 함수 사용)---")
    all_documents = []
    for path in PDF_FILE_PATHS: 
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            documents = loader.load_and_split()
            all_documents.extend(documents)
            print(f"'{path}'에서 {len(documents)}개 페이지 문서 로드 완료.")
        else:
            print(f"경고: 파일이 존재하지 않습니다. '{path}'")
    
    if not all_documents:
        raise ValueError("로드된 PDF 문서가 없습니다. 파일 경로를 확인해주세요.")
    print(f"총 {len(all_documents)}개 페이지 문서 로드 완료.")
    return all_documents

def split_documents(documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """문서 리스트(PDF 페이지)를 더 작은 청크로 분할합니다. (사용자 요청 로직)"""
    print("---기능: 문서 청크 분할 시작---")
    
    # 1. 텍스트 정리 (줄바꿈 제거 + 공백 정리)
    cleaned_docs = []
    for doc in documents:
        content = doc.page_content
        content = re.sub(r'\s+', ' ', content)  # 모든 공백을 하나의 space로
        doc.page_content = content.strip()
        cleaned_docs.append(doc)

    # 2. 문장 단위 분할 우선
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", ". ", "? ", "! ", "\n", " "],  # 문장 단위 우선 분리
    )
    splits = splitter.split_documents(cleaned_docs)

    print(f"총 {len(documents)}개 페이지 문서를 {len(splits)}개 청크로 분할했습니다.")
    return splits

def embed_and_store_vector_db(splits: List[Dict[str, Any]], db_path: str) -> FAISS:
    """청크를 임베딩하고 벡터 DB에 저장(또는 로드)합니다."""
    print("---기능: 임베딩 및 벡터 DB 저장/로드 시작---")
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    if not os.path.exists(db_path):
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(db_path)
        print(f"벡터 DB를 '{db_path}' 경로에 새로 생성 및 저장했습니다.")
    else:
        vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print(f"'{db_path}' 경로에서 기존 벡터 DB를 로드했습니다.")
    return vectorstore

def retrieve_relevant_chunks(vectorstore: FAISS, question: str) -> str:
    """
    벡터 DB에서 사용자 질문에 가장 관련성 높은 청크를 유사도 검색으로 찾고,
    출처와 내용을 출력합니다.
    """
    print("---기능: 관련 청크 유사도 검색 시작---")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    
    # 검색된 문서의 출처와 내용을 출력하는 부분
    print("\n[참고 문서 출처 및 내용]")
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', '알 수 없는 출처')
        page = doc.metadata.get('page', '알 수 없는 페이지')
        print(f"--- 청크 {i+1} ---")
        print(f" 출처: {source}")
        print(f" 페이지: {page}")
        print(f" 내용:\n{doc.page_content}\n")
    
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"검색된 청크 수: {len(docs)}개")
    return context

def generate_answer_with_llm(context: str, question: str, llm: ChatGroq, prompt_template: ChatPromptTemplate) -> str:
    """검색된 문맥과 질문을 바탕으로 LLM을 사용하여 답변을 생성합니다."""
    print("---기능: 답변 생성 시작---")
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    inputs = {"context": context, "question": question}
    answer = rag_chain.invoke(inputs)
    return answer

# --- 5. LangGraph 노드 함수 정의 ---

def load_data_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: PDF 데이터를 로드합니다."""
    print("\n---노드: 데이터 로드 (Load Data) 실행---")
    documents = load_csv_data(PDF_FILE_PATHS[0] if PDF_FILE_PATHS else "") 
    question_from_state = state.get('question') 
    return {"question": question_from_state, "documents": documents}

def split_documents_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 로드된 문서를 청크로 분할합니다."""
    print("\n---노드: 문서 분할 (Split Documents) 실행---")
    if "documents" not in state or not state["documents"]: 
        raise ValueError("문서 분할을 위해 로드된 문서가 필요합니다.")
    
    # 텍스트 정리 및 문장 단위 분할 로직 적용
    print("✂️ 문서 분할 중 (문장 단위)...")
    cleaned_docs = []
    for doc in state["documents"]:
        content = doc.page_content
        content = re.sub(r'\s+', ' ', content)
        doc.page_content = content.strip()
        cleaned_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", ". ", "? ", "! ", "\n", " "],
    )
    splits = splitter.split_documents(cleaned_docs)
    
    print(f"✅ {len(splits)}개의 청크로 분할됨.")
    return {**state, "splits": splits}

def embed_and_store_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 청크를 임베딩하고 벡터 DB에 저장(또는 로드)합니다."""
    print("\n---노드: 임베딩 및 벡터 DB 저장 (Embed & Store) 실행---")
    if "splits" not in state or not state["splits"]:
        raise ValueError("임베딩을 위해 분할된 청크가 필요합니다.")
    
    vectorstore = embed_and_store_vector_db(state["splits"], VECTOR_DB_PATH)
    
    return {**state, "vectorstore": vectorstore}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 벡터 DB에서 관련 청크를 검색합니다."""
    print("\n---노드: 검색 (Retrieve) 실행---")
    if "vectorstore" not in state or not state["vectorstore"]:
        raise ValueError("검색을 위해 벡터 DB가 필요합니다.")
    question = state["question"] 
    vectorstore = state["vectorstore"]
    context = retrieve_relevant_chunks(vectorstore, question)
    
    return {**state, "context": context}

def generate_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 검색된 문맥과 질문을 바탕으로 답변을 생성합니다."""
    print("\n---노드: 생성 (Generate) 실행---")
    if ("context" not in state or not state["context"] or 
        "question" not in state or not state["question"]):
        raise ValueError("답변 생성을 위해 문맥과 질문이 필요합니다.")
    context = state["context"]
    question = state["question"]
    answer = generate_answer_with_llm(context, question, llm, rag_prompt)
    
    return {**state, "answer": answer}

# --- 6. LangGraph 워크플로우 빌드 및 컴파일 ---
def build_rag_workflow():
    """RAG 워크플로우를 빌드하고 컴파일합니다."""
    workflow = StateGraph(GraphState) 

    workflow.add_node("load_data", load_data_node)
    workflow.add_node("split_documents", split_documents_node)
    workflow.add_node("embed_and_store", embed_and_store_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge("load_data", "split_documents")
    workflow.add_edge("split_documents", "embed_and_store")
    workflow.add_edge("embed_and_store", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    workflow.set_entry_point("load_data")

    return workflow.compile()

# --- 7. 메인 실행 부분 ---
if __name__ == "__main__":
    print("\n---농작물 병충해 챗봇 에이전트 시작 (종료하려면 'exit' 또는 'quit' 입력)---")
    
    app = build_rag_workflow()

    while True:
        user_question = input("\n질문을 입력하세요: ")
        if user_question.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다.")
            break

        if not user_question.strip():
            print("질문을 입력해주세요.")
            continue
        
        try:
            final_state = app.invoke({"question": user_question})

            print("\n---챗봇 답변---")
            print(final_state['answer'])

        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")

    print("\n---농작물 병충해 챗봇 에이전트 종료---")