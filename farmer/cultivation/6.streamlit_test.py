import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnablePassthrough

# --- 1. 환경 설정 및 초기화 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # 환경 변수가 설정되지 않았을 경우 오류 메시지를 표시하고 종료합니다.
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    st.stop()

st.title("🌱 농작물 재배 정보 챗봇")

# FAISS 벡터 DB가 저장된 경로를 지정합니다.
VECTOR_DB_PATH = 'faiss_pdf_db' 

# --- 2. LLM 및 프롬프트 설정 ---
# ChatGroq LLM 모델을 초기화합니다.
llm = ChatGroq(model_name="llama3-70b-8192", 
               temperature=0.7, 
               api_key=OPENAI_API_KEY)

# RAG(Retrieval-Augmented Generation) 답변 생성을 위한 프롬프트입니다.
# chat_history, context, question 변수를 받아 답변을 생성합니다.
RAG_PROMPT_TEMPLATE = """
당신은 대한민국 농작물 정보 전문가입니다. 

이전 대화:
{chat_history}

제공된 다음의 정보를 사용하여 사용자의 질문에 답변해 주세요.
{context}

질문: {question}

당신이 지킬 규칙들은 다음과 같습니다:
- 이전 대화 내용과 현재 질문에 모두 부합하는 답변을 제공해야 합니다.
- 제공된 정보에 없는 정보, 저의 상식, 추측, 거짓 정보, 한자 등은 절대 답변에 넣지 마세요.
- 모든 답변은 한글로만 답해야 합니다. 한글이 아니면 절대 출력하지 말고, 한글로만 답변을 완성해야 합니다.
- 각 재배 단계나 설명은 반드시 "한 문장씩 줄바꿈"해서 써주세요.
- 답변이 불가능하거나 제공된 정보에 관련 정보가 없으면 "주어진 정보로는 답변할 수 없습니다."라고만 답해야 합니다.
- 친근하게 대화하듯, 자연스럽게 설명해주세요.
"""
rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# 검색 질문을 생성하는 프롬프트입니다.
# LLM이 불필요한 문장이나 다른 언어를 덧붙이지 않도록 매우 강력하게 지시합니다.
QUERY_TRANSFORM_PROMPT = """
이전 대화와 현재 질문을 분석하여 문서 검색에 사용될 가장 적절한 질문을 **단 하나의 문장으로** 생성하세요. 이 질문은 반드시 한국어여야 합니다.

- 만약 현재 질문이 이전 대화의 맥락(context)에 의존하는 경우에만, 이전 대화 내용을 참고하여 질문을 변환하세요.
- 만약 현재 질문이 이전 대화와 전혀 관련이 없는 새로운 내용이라면, 오직 현재 질문만을 사용하여 검색 질문을 만드세요.
- 절대로 불필요한 설명, 인사말, 또는 다른 언어를 추가하지 마세요.

이전 대화:
{chat_history}

질문: {question}
"""

query_transform_prompt = PromptTemplate.from_template(QUERY_TRANSFORM_PROMPT)

# --- 3. LangGraph 상태 정의 ---
# LangGraph 워크플로우를 위한 상태를 정의합니다.
class GraphState(TypedDict):
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    vectorstore: Optional[FAISS]
    sources: Optional[List[str]]
    chat_history: Optional[str]

# --- 4. 핵심 기능 함수 정의 ---
@st.cache_resource
def load_vector_db(db_path: str) -> FAISS:
    """FAISS 벡터 DB를 로드하고 캐시합니다."""
    # 임베딩 모델을 로드합니다. (캐싱되어 재사용됨)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    except Exception as e:
        st.error(f"❌ 임베딩 모델을 다운로드할 수 없습니다. 네트워크 연결 상태를 확인하고 VPN/방화벽 설정을 점검해 주세요. \n\n오류: {e}")
        st.stop()

    if not os.path.exists(db_path):
        st.error(f"'{db_path}' 경로에 벡터 DB 파일이 없습니다. 먼저 DB를 생성해야 합니다.")
        st.stop()
    
    # 로컬 경로에서 FAISS DB를 로드합니다.
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def retrieve_relevant_chunks(vectorstore: FAISS, question: str) -> Dict[str, Any]:
    """벡터 DB에서 관련 문서를 검색합니다."""
    # 유사도 검색을 위한 리트리버를 설정합니다.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    # 질문에 대한 문서 덩어리(chunk)를 검색합니다.
    docs = retriever.invoke(question)
    # 검색된 문서의 내용을 하나의 문자열로 결합합니다.
    context = "\n\n".join([doc.page_content for doc in docs])
    # 문서 출처를 추출합니다.
    sources = list(set([doc.metadata.get('source') for doc in docs if doc.metadata.get('source')]))
    return {"context": context, "sources": sources}

# --- 5. LangGraph 노드 함수 정의 ---
def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """대화 기록을 바탕으로 검색 질문을 변환하고, 문서를 검색합니다."""
    question = state["question"]
    chat_history = state["chat_history"]
    vectorstore = state["vectorstore"]
    
    # 질문 변환을 위한 체인을 정의합니다.
    query_transform_chain = query_transform_prompt | llm | StrOutputParser()
    
    # LLM을 호출하여 대화 기록을 기반으로 새로운 검색 질문을 생성합니다.
    transformed_question = query_transform_chain.invoke({
        "question": question, 
        "chat_history": chat_history
    })
    
    st.info(f"🔍 검색을 위한 질문을 변환 중... (변환된 질문: '{transformed_question}')")
    
    # 벡터 DB가 없을 경우를 처리합니다.
    if "vectorstore" not in state or not state["vectorstore"]:
        return {"context": None, "sources": [], "question": transformed_question, "chat_history": chat_history}
    
    # 변환된 질문으로 벡터 DB에서 문서를 검색합니다.
    retrieval_result = retrieve_relevant_chunks(vectorstore, transformed_question)
    
    return {
        "context": retrieval_result["context"],
        "sources": retrieval_result["sources"],
        "question": question,
        "chat_history": chat_history
    }

def generate_node(state: GraphState) -> Dict[str, Any]:
    """답변을 생성하는 노드입니다. 현재는 더미 함수입니다."""
    return {}

# --- 6. LangGraph 워크플로우 빌드 및 컴파일 ---
@st.cache_resource
def build_rag_workflow():
    """LangGraph 워크플로우를 구축하고 컴파일합니다."""
    workflow = StateGraph(GraphState) 
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    workflow.set_entry_point("retrieve")
    return workflow.compile()

# --- 7. 메인 실행 부분 ---
# 세션 상태에 'messages'가 없으면 초기화합니다.
if "messages" not in st.session_state:
    st.session_state.messages = []
    
try:
    # 벡터 DB를 로드합니다.
    loaded_vectorstore = load_vector_db(VECTOR_DB_PATH)
except FileNotFoundError as e:
    st.error(f"오류: {e}")
    st.stop()

# LangGraph 워크플로우를 빌드합니다.
app = build_rag_workflow()

# RAG 체인을 정의합니다.
rag_chain = rag_prompt | llm | StrOutputParser()

# 기존 메시지를 화면에 표시합니다.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    if message.get("sources"):
        with st.expander("참고 자료 출처"):
            for source in message["sources"]:
                st.markdown(f"- {source}")

# 이전 대화 기록을 문자열로 변환하는 헬퍼 함수
def format_chat_history(messages):
    """스트림릿 메시지 딕셔너리를 문자열로 변환합니다."""
    formatted_history = ""
    for msg in messages:
        if msg["role"] == "user":
            formatted_history += f"사용자: {msg['content']}\n"
        else:
            formatted_history += f"챗봇: {msg['content']}\n"
    return formatted_history

# 사용자 질문 입력 처리
if user_question := st.chat_input("질문을 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        response_container = st.empty()
        try:
            # 이전 대화 기록을 LLM이 이해하기 쉬운 문자열로 변환합니다.
            formatted_history = format_chat_history(st.session_state.messages)

            # LangGraph 워크플로우를 호출하여 검색 단계와 생성 단계를 처리합니다.
            final_state = app.invoke({
                "question": user_question,
                "vectorstore": loaded_vectorstore,
                "chat_history": formatted_history
            })
            retrieved_context = final_state.get('context')
            retrieved_sources = final_state.get('sources', [])
            
            if not retrieved_context:
                response_text = "주어진 정보로는 답변할 수 없습니다."
                response_container.markdown(response_text)
            else:
                input_for_prompt = {
                    "context": retrieved_context,
                    "question": user_question,
                    "chat_history": formatted_history
                }
                
                # RAG 체인으로 답변을 스트리밍합니다.
                response_generator = rag_chain.stream(input_for_prompt)
                response_text = response_container.write_stream(response_generator)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text, "sources": retrieved_sources})
            
            if retrieved_sources:
                with st.expander("참고 자료 출처"):
                    for source in retrieved_sources:
                        st.markdown(f"- {source}")
                        
        except Exception as e:
            error_message = f"오류가 발생했습니다: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.stop()