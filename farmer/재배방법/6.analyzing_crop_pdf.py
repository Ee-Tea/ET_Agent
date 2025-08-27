import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict
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

# 사용할 벡터 DB 경로를 'faiss_pdf_db'로 변경
# VECTOR_DB_PATH = 'faiss_pdf_db' 
VECTOR_DB_PATH = 'faiss_db_bge_m3' 

# --- 2. LLM 및 프롬프트 설정 ---
llm = ChatGroq(model_name="llama3-70b-8192", 
               temperature=0.7, 
               api_key=OPENAI_API_KEY)

RAG_PROMPT_TEMPLATE = """
당신은 대한민국 농작물 정보 전문가입니다. 

다음은 작물 재배에 대한 정보를 담고 있습니다. 제공된 다음의 정보를 사용하여 사용자의 질문에 답변해 주세요.
{context}

당신이 지킬 규칙들은 다음과 같습니다:
- 제공된 정보에 없는 정보, 저의 상식, 추측, 거짓 정보, 한자 등은 절대 답변에 넣지 마세요.
- 모든 답변은 한글로만 답해야 합니다. 한글이 아니면 절대 출력하지 말고, 한글로만 답변을 완성해야 합니다.
- 각 재배 단계나 설명은 반드시 "한 문장씩 줄바꿈"해서 써주세요.
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
    답변에 사용된 소스 파일을 추적하기 위해 `sources` 키를 추가했습니다.
    """
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    vectorstore: Optional[FAISS]
    # 답변에 사용된 소스 파일 경로를 저장하기 위한 리스트
    sources: Optional[List[str]]

# --- 4. 핵심 기능 함수 정의 ---
def load_vector_db(db_path: str) -> FAISS:
    """기존에 저장된 벡터 DB를 로드합니다."""
    print("---기능: 벡터 DB 로드 시작---")
    #embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    #BAAI/bge-m3 모델로 변경
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'}, # GPU가 없다면 'cpu'를 사용하세요.
        encode_kwargs={'normalize_embeddings': True}
    )
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"'{db_path}' 경로에 벡터 DB 파일이 없습니다. 먼저 DB를 생성해야 합니다.")
    
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    print(f"'{db_path}' 경로에서 기존 벡터 DB를 로드했습니다.")
    return vectorstore

def retrieve_relevant_chunks(vectorstore: FAISS, question: str) -> Dict[str, Any]:
    """
    벡터 DB에서 관련 문서를 검색하고, 문서 내용과 소스 파일 경로를 반환합니다.
    """
    print("---기능: 관련 청크 유사도 검색 시작---")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    
    # 문서 내용과 소스 파일 경로를 별도로 추출합니다.
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 중복을 제거하기 위해 set을 사용합니다.
    sources = list(set([doc.metadata.get('source') for doc in docs if doc.metadata.get('source')]))
    
    print(f"검색된 청크 수: {len(docs)}개")
    return {"context": context, "sources": sources}

# --- 5. LangGraph 노드 함수 정의 ---

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph 노드: 검색
    이제 이 노드는 문맥(context)과 소스(sources)를 함께 반환합니다.
    """
    print("\n---노드: 검색 (Retrieve) 실행---")
    if "vectorstore" not in state or not state["vectorstore"]:
        raise ValueError("검색을 위해 벡터 DB가 필요합니다.")
    question = state["question"]
    vectorstore = state["vectorstore"]
    
    # 검색 함수가 이제 context와 sources를 모두 반환합니다.
    retrieval_result = retrieve_relevant_chunks(vectorstore, question)
    
    # 반환된 결과를 LangGraph 상태에 업데이트합니다.
    return {
        "context": retrieval_result["context"],
        "sources": retrieval_result["sources"],
        "question": question
    }

def generate_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph 노드: 답변 생성
    이 노드는 워크플로우를 구성하기 위해 존재합니다.
    실제 답변 생성은 메인 루프의 스트리밍 체인에서 처리합니다.
    """
    return {}

# --- 6. LangGraph 워크플로우 빌드 및 컴파일 ---
def build_rag_workflow():
    """간소화된 RAG 워크플로우를 빌드하고 컴파일합니다."""
    workflow = StateGraph(GraphState) 
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    workflow.set_entry_point("retrieve")
    return workflow.compile()

# --- 7. 메인 실행 부분 ---
if __name__ == "__main__":
    print("\n---농작물 챗봇 에이전트 시작 (종료하려면 'exit' 또는 'quit' 입력)---")

    try:
        loaded_vectorstore = load_vector_db(VECTOR_DB_PATH)
    except FileNotFoundError as e:
        print(f"오류: {e}")
        print("프로그램을 종료합니다.")
        exit()

    app = build_rag_workflow()

    # 스트리밍을 위한 체인을 별도로 생성합니다.
    rag_chain_stream = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    while True:
        user_question = input("\n질문을 입력하세요: ")
        if user_question.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다.")
            break
        if not user_question.strip():
            print("질문을 입력해주세요.")
            continue

        try:
            # 1. LangGraph를 실행하여 검색(retrieve) 단계를 처리합니다.
            final_state = app.invoke({"question": user_question, "vectorstore": loaded_vectorstore})
            retrieved_context = final_state.get('context')
            retrieved_sources = final_state.get('sources', []) # 새로운 sources 키를 가져옵니다.
            
            if not retrieved_context:
                print("주어진 정보로는 답변할 수 없습니다.")
                continue

            # 2. 검색된 문맥을 바탕으로 스트리밍 체인을 실행합니다.
            print("\n---챗봇 답변---")
            for chunk in rag_chain_stream.stream({"context": retrieved_context, "question": user_question}):
                print(chunk, end="", flush=True)
            print("\n")

            # 3. 답변 후 사용된 소스 파일을 출력합니다.
            if retrieved_sources:
                print("---참고 자료 출처---")
                for source in retrieved_sources:
                    print(f"- {source}")

        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")

    print("\n---농작물 챗봇 에이전트 종료---")