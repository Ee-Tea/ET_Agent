import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus as LangChainMilvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
# EnsembleRetriever를 임포트합니다.
from langchain.retrievers import EnsembleRetriever

# --- 1. 환경 설정 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# 두 개의 Milvus 컬렉션 이름 정의
COLLECTION_NAME_INFO = "crop_info"
COLLECTION_NAME_GROW = "crop_grow" # crop_prices -> crop_grow로 변경

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# --- 2. LLM 및 임베딩 모델 설정 ---
llm = ChatGroq(model_name="llama3-70b-8192", 
               temperature=0.7, 
               api_key=OPENAI_API_KEY)

# 문서 임베딩을 위한 모델
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

RAG_PROMPT_TEMPLATE = """
당신은 대한민국 농작물 정보 전문가입니다. 

다음은 작물 재배 및 생육에 대한 정보를 담고 있습니다. 제공된 다음의 정보를 사용하여 사용자의 질문에 답변해 주세요.
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
    이제 'retriever' 키가 EnsembleRetriever 객체를 담습니다.
    """
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    retriever: Optional[EnsembleRetriever] # Milvus 객체 대신 EnsembleRetriever 객체 사용
    sources: Optional[List[str]]

# --- 4. 핵심 기능 함수 정의 ---
def create_retriever() -> EnsembleRetriever:
    """두 개의 Milvus 컬렉션에 연결하여 EnsembleRetriever를 생성합니다."""
    print("---기능: Milvus 컬렉션 연결 및 EnsembleRetriever 생성 시작---")
    try:
        # 첫 번째 컬렉션(crop_info)에 대한 Milvus 객체 생성
        vectorstore_info = LangChainMilvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME_INFO,
            connection_args={
                "host": MILVUS_HOST,
                "port": MILVUS_PORT
            },
            consistency_level="Bounded"
        )
        print(f"✅ '{COLLECTION_NAME_INFO}' 컬렉션에 연결했습니다.")

        # 두 번째 컬렉션(crop_grow)에 대한 Milvus 객체 생성
        vectorstore_grow = LangChainMilvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME_GROW, # collection_name 변경
            connection_args={
                "host": MILVUS_HOST,
                "port": MILVUS_PORT
            },
            consistency_level="Bounded"
        )
        print(f"✅ '{COLLECTION_NAME_GROW}' 컬렉션에 연결했습니다.")

        # 각 컬렉션에서 검색기(retriever)를 생성
        retriever_info = vectorstore_info.as_retriever(search_kwargs={"k": 3})
        retriever_grow = vectorstore_grow.as_retriever(search_kwargs={"k": 3})

        # 두 검색기를 EnsembleRetriever로 결합
        # weights는 각 검색기의 결과를 합칠 때의 가중치입니다.
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_info, retriever_grow], # retriever_prices -> retriever_grow로 변경
            weights=[0.5, 0.5] # 50%씩 동일한 중요도 부여
        )
        
        print("✅ EnsembleRetriever가 성공적으로 생성되었습니다.")
        return ensemble_retriever
    except Exception as e:
        print(f"Milvus 연결 또는 EnsembleRetriever 생성 중 오류 발생: {e}")
        raise

def retrieve_relevant_chunks(retriever: EnsembleRetriever, question: str) -> Dict[str, Any]:
    """
    EnsembleRetriever를 사용하여 두 컬렉션에서 관련 문서를 검색합니다.
    """
    print("---기능: 관련 청크 유사도 검색 시작---")
    docs = retriever.invoke(question)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get('source') for doc in docs if doc.metadata.get('source')]))
    
    print(f"검색된 총 청크 수: {len(docs)}개")
    return {"context": context, "sources": sources}

# --- 5. LangGraph 노드 함수 정의 ---
def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph 노드: 검색
    """
    print("\n---노드: 검색 (Retrieve) 실행---")
    if "retriever" not in state or not state["retriever"]:
        raise ValueError("검색을 위해 EnsembleRetriever가 필요합니다.")
    question = state["question"]
    retriever = state["retriever"]
    
    retrieval_result = retrieve_relevant_chunks(retriever, question)
    
    return {
        "context": retrieval_result["context"],
        "sources": retrieval_result["sources"],
        "question": question
    }

def generate_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph 노드: 답변 생성
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
        ensemble_retriever = create_retriever()
    except Exception as e:
        print(f"오류: {e}")
        print("프로그램을 종료합니다.")
        sys.exit()

    app = build_rag_workflow()

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
            # LangGraph를 실행하여 검색(retrieve) 단계를 처리합니다.
            final_state = app.invoke({"question": user_question, "retriever": ensemble_retriever})
            retrieved_context = final_state.get('context')
            retrieved_sources = final_state.get('sources', [])
            
            if not retrieved_context:
                print("주어진 정보로는 답변할 수 없습니다.")
                continue

            print("\n---챗봇 답변---")
            for chunk in rag_chain_stream.stream({"context": retrieved_context, "question": user_question}):
                print(chunk, end="", flush=True)
            print("\n")

            if retrieved_sources:
                print("---참고 자료 출처---")
                for source in retrieved_sources:
                    print(f"- {source}")

        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")

    print("\n---농작물 챗봇 에이전트 종료---")