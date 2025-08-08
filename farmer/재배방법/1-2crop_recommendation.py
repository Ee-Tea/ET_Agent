import os
from dotenv import load_dotenv
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, Optional, TypedDict

# --- 1. 환경 설정 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

print(f"OPENAI_API_KEY 로드 완료: {OPENAI_API_KEY[:5]}...") # 보안을 위해 키 앞부분만 출력

# 데이터 파일 경로 및 벡터 DB 저장 경로
CSV_FILE_PATH = 'data/작물농업 작물재배 정보_20190925.csv'
VECTOR_DB_PATH = 'faiss_crop_re_db' # 벡터 DB 저장 경로 (함수형 전용으로 경로 변경)

# --- 2. LLM 및 프롬프트 설정 ---
llm = ChatGroq(model_name="llama3-70b-8192", 
                temperature=0.5, 
                api_key=OPENAI_API_KEY)

RAG_PROMPT_TEMPLATE = """
당신은 대한민국 농업 작물 추천 전문가입니다. 

다음은 다양한 작물에 대한 정보를 담고 있습니다. 사용자의 질문에 따라 가장 적합한 작물을 추천해주세요.
{context}

당신이 지킬 규칙들은 다음과 같습니다:
- 가장 중요한 규칙입니다. 모든 답변은 무조건 한글로만 답해야 합니다. 영어나 한자는 절대 사용하지 마세요. 한글이 아닌 답변은 내지 마세요.
- 사용자의 조건(예: 기후, 토양, 재배 시기, 난이도 등)에 맞는 작물을 1가지 이상 추천해주세요.
- 추천하는 작물이 왜 적합한지 근거(예: 재배 환경, 쉬운 난이도 등)를 명확히 설명해야 합니다.
- 추천 작물의 파종 시기, 재배 특징, 수확 시기 등 제공된 정보에 있는 상세 내용을 함께 설명해 주세요.
- 각 추천 작물에 대한 설명은 반드시 "한 문장씩 줄바꿈"해서 써주세요.
- 제공된 정보에 없는 정보, 저의 상식, 추측, 거짓 정보 등은 절대 답변에 넣지 마세요.
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
    LangGraph는 TypedDict 기반 상태를 더 유연하게 처리합니다.
    """
    question: Optional[str]
    documents: Optional[List[Dict[str, Any]]]
    splits: Optional[List[Dict[str, Any]]]
    vectorstore: Optional[FAISS]
    context: Optional[str]
    # 요약된 컨텍스트를 저장할 새 필드 추가
    summarized_context: Optional[str]
    answer: Optional[str]

# --- 4. 핵심 기능 함수 정의 ---

def load_csv_data(file_path: str) -> List[Dict[str, Any]]:
    """CSV 파일을 로드하여 문서 리스트를 반환합니다."""
    print("---기능: 데이터 로드 시작---")
    loader = CSVLoader(file_path=file_path, encoding='utf-8')
    documents = loader.load()
    print(f"로드된 문서 개수: {len(documents)}개")
    return documents

def split_documents(documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """문서 리스트를 청크로 분할합니다."""
    print("---기능: 문서 분할 시작---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(documents)
    print(f"분할된 청크 개수: {len(splits)}개")
    return splits

def embed_and_store_vector_db(splits: List[Dict[str, Any]], db_path: str) -> FAISS:
    """청크를 임베딩하고 벡터 DB에 저장(또는 로드)합니다."""
    print("---기능: 임베딩 및 벡터 DB 저장 시작---")
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    if not os.path.exists(db_path):
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(db_path)
        print(f"벡터 DB를 '{db_path}' 경로에 새로 생성 및 저장했습니다.")
    else:
        vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print(f"'{db_path}' 경로에서 기존 벡터 DB를 로드했습니다.")
    return vectorstore

def retrieve_relevant_chunks(vectorstore: FAISS, question: str) -> tuple[str, str]:
    """벡터 DB에서 사용자 질문에 가장 관련성 높은 청크를 유사도 검색으로 찾습니다."""
    print("---기능: 관련 청크 유사도 검색 시작---")
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    docs = retriever.invoke(question)
    
    # 원본 컨텍스트와 요약된 컨텍스트를 모두 생성
    context = "\n\n".join([doc.page_content for doc in docs])
    
    summarized_chunks = []
    for doc in docs:
        content = doc.page_content
        summarized_chunks.append(content[:100] + "..." if len(content) > 100 else content)
    summarized_context = "\n\n".join(summarized_chunks)
    
    print(f"검색된 청크 수: {len(docs)}개")
    
    return context, summarized_context

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
    """LangGraph 노드: 데이터 로드"""
    print("\n---노드: 데이터 로드 (Load Data) 실행---")
    documents = load_csv_data(CSV_FILE_PATH)
    
    question_from_state = state.get('question') 
    
    return {"question": question_from_state, "documents": documents}

def split_documents_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 문서 분할"""
    print("\n---노드: 문서 분할 (Split Documents) 실행---")
    if "documents" not in state or not state["documents"]: 
        raise ValueError("문서 분할을 위해 로드된 문서가 필요합니다.")
    splits = split_documents(state["documents"])
    
    return {**state, "splits": splits} 

def embed_and_store_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 임베딩 및 벡터 DB 저장"""
    print("\n---노드: 임베딩 및 벡터 DB 저장 (Embed & Store) 실행---")
    if "splits" not in state or not state["splits"]:
        raise ValueError("임베딩을 위해 분할된 청크가 필요합니다.")
    vectorstore = embed_and_store_vector_db(state["splits"], VECTOR_DB_PATH)
    
    return {**state, "vectorstore": vectorstore}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 검색"""
    print("\n---노드: 검색 (Retrieve) 실행---")
    if "vectorstore" not in state or not state["vectorstore"]:
        raise ValueError("검색을 위해 벡터 DB가 필요합니다.")
    question = state["question"] 
    vectorstore = state["vectorstore"]
    
    context, summarized_context = retrieve_relevant_chunks(vectorstore, question)
    
    return {**state, "context": context, "summarized_context": summarized_context}

def generate_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 답변 생성"""
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
    print("\n---농작물 챗봇 에이전트 시작 (종료하려면 'exit' 또는 'quit' 입력)---")

    # 워크플로우 컴파일
    app = build_rag_workflow()

    # 챗봇 루프 시작
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
            
            print("\n---참고한 원본 텍스트 요약---")
            print(final_state.get('summarized_context', '참고한 텍스트가 없습니다.'))

        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")

    print("\n---농작물 챗봇 에이전트 종료---")