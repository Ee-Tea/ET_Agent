import streamlit as st
import os
import re
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict

# Langchain 및 LangGraph 관련 라이브러리
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 1. 환경 설정 ---
# .env 파일에서 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PESTICIDE_API_KEY = os.getenv("PESTICIDE_API_KEY")

# 통합할 개별 벡터 DB 저장 경로를 정의합니다.
# 이 폴더들이 존재해야 합니다.
DATA_DB_CONFIG = {
    "cultivation": "faiss_crop_guide_db",
    "fertilizer": "faiss_crop_fer_db",
    "pest_disease": "faiss_crop_pest_db",
}

# --- 2. LLM 및 프롬프트 설정 ---
# 답변 생성을 위한 LLM
llm = ChatGroq(model_name="llama3-70b-8192",
               temperature=0.7,
               api_key=OPENAI_API_KEY)

# 키워드 추출을 위한 LLM (더 가볍고 정확한 모델 사용)
llm_keyword = ChatGroq(model_name="llama3-8b-8192",
                       temperature=0.0,
                       api_key=OPENAI_API_KEY)

# 질문 분류를 위한 프롬프트
CLASSIFY_PROMPT_TEMPLATE = """
당신은 사용자의 질문이 어떤 주제에 관한 것인지 분류하는 전문가입니다.

질문을 분석하여 다음 규칙에 따라 답변하세요.
- 질문이 '농약'에 관련이 있다면, 'pesticide'라고만 답변하세요.
- 질문이 농업(농작물 재배, 비료, 병해충 등)에 관련이 있다면, 'agriculture'라고만 답변하세요.
- 그 외의 모든 일반적인 질문이라면, 'other'라고만 답변하세요.
- 답변은 한 단어만 출력해야 합니다.

질문: {question}
답변:
"""
classify_prompt = ChatPromptTemplate.from_template(CLASSIFY_PROMPT_TEMPLATE)

# RAG 프롬프트 (내부 DB 사용)
RAG_PROMPT_TEMPLATE = """
당신은 대한민국 농업 전문가입니다.

다음은 농업에 대한 정보를 담고 있습니다. 제공된 다음의 정보를 사용하여 사용자의 질문에 답변해 주세요.
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

# 농약 전문가 프롬프트 템플릿 (API 정보만 사용)
PESTICIDE_RAG_PROMPT_TEMPLATE = """
당신은 대한민국 농약 전문가입니다.

다음은 외부 API에서 얻은 농약 관련 정보입니다. 이 정보를 활용하여 사용자의 질문에 답변해 주세요.

# 외부 API에서 얻은 정보:
{api_result}

당신이 지킬 규칙들은 다음과 같습니다:
- 제공된 정보에 없는 정보, 저의 상식, 추측, 거짓 정보 등은 절대 답변에 넣지 마세요.
- 모든 답변은 한글로만 답해야 합니다. 한글이 아니면 절대 출력하지 말고, 한글로만 답변을 완성해야 합니다.
- 각 설명은 반드시 "한 문장씩 줄바꿈"해서 써주세요.
- 답변이 불가능하거나 제공된 정보에 관련 정보가 없으면 "주어진 정보로는 답변할 수 없습니다."라고만 답해야 합니다.
- 친근하고 전문적인 태도로, 자연스럽게 설명해주세요.

질문: {question}
답변:
"""
pesticide_rag_prompt = ChatPromptTemplate.from_template(PESTICIDE_RAG_PROMPT_TEMPLATE)

# 웹 검색 결과를 요약하기 위한 프롬프트
WEB_SEARCH_PROMPT_TEMPLATE = """
당신은 검색 전문가입니다.
다음 검색 결과들을 활용하여 사용자의 질문에 가장 정확하고 완전한 답변을 제공해 주세요.

답변 규칙
1. **친절하고 자연스럽게**: 친근하고 명확한 문체로 작성해 주세요.
2. **정보의 출처 명시**: 검색 결과에 제시된 정보만을 사용하세요. 만약 질문에 대한 답변이 검색 결과에 없다면, '검색 결과에 해당 정보가 없습니다.'라고 명확하게 말해야 합니다.
3. **핵심 요약 및 정리**: 여러 검색 결과에서 중복되는 핵심 내용들을 종합하여 간결하게 요약해 주세요.
4. **구체적이고 상세하게**: 답변은 가능한 한 구체적인 정보(예: 날짜, 숫자, 기관명 등)를 포함하여 작성해 주세요.
5. **한글로만 답변**: 모든 답변은 한글로만 제공해야 합니다.

질문: {question}
답변:
"""
web_search_prompt = ChatPromptTemplate.from_template(WEB_SEARCH_PROMPT_TEMPLATE)

# 검색 결과가 질문에 충분한지 판단하기 위한 프롬프트
RETRIEVAL_CHECK_PROMPT_TEMPLATE = """
다음은 사용자의 질문과 내부 데이터베이스에서 검색된 정보입니다.
이 검색된 정보만으로 사용자의 질문에 충분히 답변할 수 있는지 판단해 주세요.

- 정보가 질문에 대한 답변을 제공할 수 있는 관련 내용이라면 'sufficient'라고만 답변하세요.
- 정보가 질문과 전혀 관련이 없다면 'insufficient'라고만 답변하세요.
- 답변은 한 단어만 출력해야 합니다.

질문: {question}
검색된 정보: {context}
답변:
"""
retrieval_check_prompt = ChatPromptTemplate.from_template(RETRIEVAL_CHECK_PROMPT_TEMPLATE)

# 키워드 추출을 위한 프롬프트
KEYWORD_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    "사용자의 질문에서 작물명, 병해충명을 쉼표로 구분하여 추출해. 다른 말은 하지마.\n질문: {question}\n키워드:"
)

tavily_tool = TavilySearchResults(max_results=5, api_key=TAVILY_API_KEY)

# --- 3. LangGraph 상태 정의 ---
class GraphState(TypedDict):
    question: Optional[str]
    vectorstore: Optional[FAISS]
    context: Optional[str]
    answer: Optional[str]
    classification: Optional[str]
    retrieval_sufficiency: Optional[str]
    keywords: Optional[str]
    api_result: Optional[str]
    # 답변의 출처를 기록하기 위한 새로운 필드 추가
    answer_source: Optional[str] 

# --- 4. 핵심 기능 함수 정의 ---
def retrieve_relevant_chunks(vectorstore: FAISS, question: str) -> str:
    """벡터 DB에서 질문과 관련된 문맥을 검색합니다."""
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def generate_answer_with_llm(context: str, question: str, llm: ChatGroq, prompt_template: ChatPromptTemplate) -> str:
    """LLM을 사용하여 최종 답변을 생성합니다."""
    rag_chain_internal = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    inputs = {"context": context, "question": question}
    answer = rag_chain_internal.invoke(inputs)
    return answer

# 농약 API 호출 함수
def call_pesticide_api(crop_name: str = "", disease_name: str = ""):
    """농약 API를 호출하여 농약 정보를 DataFrame으로 반환합니다."""
    BASE_URL = "https://psis.rda.go.kr/openApi/service.do"
    if not PESTICIDE_API_KEY:
        return pd.DataFrame()
    params = {
        "apiKey": PESTICIDE_API_KEY,
        "serviceCode": "SVC01",
        "serviceType": "AA001",
        "displayCount": 10,
        "startPoint": 1,
        "cropName": crop_name,
        "diseaseWeedName": disease_name,
        "similarFlag" : "Y",
    }
    try:
        res = requests.get(BASE_URL, params=params)
        res.raise_for_status()
        root = ET.fromstring(res.text)
        rows = []
        for item in root.findall(".//item"):
            rows.append({
                "작물명": item.findtext("cropName"),
                "병해충": item.findtext("diseaseWeedName"),
                "용도": item.findtext("useName"),
                "상표명": item.findtext("pestiBrandName"),
                "사용방법": item.findtext("pestiUse"),
                "희석배수": item.findtext("dilutUnit"),
                "안전사용기준(수확 일 전)": item.findtext("useSuittime"),
                "안전사용기준(회 이내)": item.findtext("useNum"),
            })
        df = pd.DataFrame(rows)
        return df
    except (requests.exceptions.RequestException, ET.ParseError) as e:
        print(f"API 호출 또는 XML 파싱 오류: {e}")
        return pd.DataFrame()

# --- 5. LangGraph 노드 함수 정의 ---
def load_and_merge_dbs_node(state: GraphState) -> Dict[str, Any]:
    """이미 존재하는 개별 벡터 DB를 로드하여 하나로 통합합니다."""
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    all_dbs_exist = all(os.path.exists(path) for path in DATA_DB_CONFIG.values())

    if not all_dbs_exist:
        st.error("필요한 벡터 DB 폴더 중 하나 이상이 존재하지 않습니다. 먼저 개별 DB를 생성해야 합니다.")
        st.stop()
    
    first_db_path = list(DATA_DB_CONFIG.values())[0]
    vectorstore = FAISS.load_local(first_db_path, embeddings, allow_dangerous_deserialization=True)

    for key, db_path in list(DATA_DB_CONFIG.items())[1:]:
        other_vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.merge_from(other_vectorstore)
    
    return {**state, "vectorstore": vectorstore}

def classify_question_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    chain = classify_prompt | llm_keyword | StrOutputParser()
    classification = chain.invoke({"question": question})
    classification_str = classification.strip()
    return {**state, "classification": classification_str}

def retrieve_and_check_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    vectorstore = state["vectorstore"]
    context = retrieve_relevant_chunks(vectorstore, question)
    
    chain = retrieval_check_prompt | llm_keyword | StrOutputParser()
    retrieval_sufficiency = chain.invoke({"question": question, "context": context})
    
    return {**state, "context": context, "retrieval_sufficiency": retrieval_sufficiency.strip()}

def generate_node(state: GraphState) -> Dict[str, Any]:
    context = state["context"]
    question = state["question"]
    answer = generate_answer_with_llm(context, question, llm, rag_prompt)
    return {**state, "answer": answer, "answer_source": "internal_db"}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    search_results = tavily_tool.invoke({"query": question})
    search_results_str = "\n".join([str(res) for res in search_results])
    truncated_search_results = search_results_str[:2000]

    web_answer = generate_answer_with_llm(truncated_search_results, question, llm, web_search_prompt)
    
    return {**state, "answer": web_answer, "answer_source": "web_search"}

# --- 농약 API 관련 노드 ---
def extract_keywords_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    keyword_chain = KEYWORD_EXTRACT_PROMPT | llm_keyword | StrOutputParser()
    keywords = keyword_chain.invoke({"question": question})
    return {**state, "keywords": keywords}

def call_api_node(state: GraphState) -> Dict[str, Any]:
    keywords = state.get("keywords")
    api_result = "외부 API에서 얻은 추가 정보가 없습니다."
    
    if keywords:
        parsed_keywords = [k.strip() for k in keywords.split(',') if k.strip()]
        crop_name = parsed_keywords[0] if len(parsed_keywords) > 0 else ""
        disease_name = parsed_keywords[1] if len(parsed_keywords) > 1 else ""

        df = call_pesticide_api(crop_name=crop_name, disease_name=disease_name)
        
        if not df.empty:
            api_result = "외부 API 결과:\n" + df.to_string(index=False)
        else:
            api_result = "외부 API 결과: 관련 정보를 찾을 수 없습니다."
    
    return {**state, "api_result": api_result}

def pesticide_generate_or_websearch_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    api_result = state["api_result"]

    if "찾을 수 없습니다" in api_result:
        # API 결과가 불충분하면 웹 검색 노드로 전환
        # web_search_node가 새로운 상태를 반환하므로, 그대로 반환
        return web_search_node(state)
    else:
        # API 결과가 충분하면 답변 생성
        pesticide_rag_chain = (
            {"question": RunnablePassthrough(), "api_result": RunnablePassthrough()}
            | pesticide_rag_prompt
            | llm
            | StrOutputParser()
        )
        answer = pesticide_rag_chain.invoke({"question": question, "api_result": api_result})
        return {**state, "answer": answer, "answer_source": "pesticide_api"}

# --- 6. LangGraph 워크플로우 빌드 및 컴파일 ---
@st.cache_resource
def build_initial_setup_graph():
    """초기 문서 로딩 및 벡터스토어 구축을 위한 그래프를 빌드합니다."""
    initial_builder = StateGraph(GraphState)
    initial_builder.add_node("load_and_merge_dbs", load_and_merge_dbs_node)
    initial_builder.set_entry_point("load_and_merge_dbs")
    initial_builder.add_edge("load_and_merge_dbs", END)
    return initial_builder.compile()

@st.cache_resource
def build_query_graph():
    """질문 분류, RAG, 웹 검색을 통합한 메인 질의 그래프를 빌드합니다."""
    query_builder = StateGraph(GraphState)
    
    query_builder.add_node("classify_question", classify_question_node)
    query_builder.add_node("retrieve_and_check", retrieve_and_check_node)
    query_builder.add_node("generate", generate_node)
    query_builder.add_node("extract_keywords", extract_keywords_node)
    query_builder.add_node("call_api", call_api_node)
    query_builder.add_node("pesticide_generate_or_websearch", pesticide_generate_or_websearch_node)
    query_builder.add_node("web_search", web_search_node)
    
    query_builder.set_entry_point("classify_question")
    
    def route_classification(state: GraphState):
        classification = state.get("classification")
        if classification == "pesticide":
            return "extract_keywords"
        elif classification == "agriculture":
            return "retrieve_and_check"
        else:
            return "web_search"
            
    query_builder.add_conditional_edges(
        "classify_question",
        route_classification,
        {
            "extract_keywords": "extract_keywords",
            "retrieve_and_check": "retrieve_and_check",
            "web_search": "web_search"
        }
    )
    
    def route_retrieval_sufficiency(state: GraphState):
        retrieval_sufficiency = state.get("retrieval_sufficiency")
        return "generate" if retrieval_sufficiency == "sufficient" else "web_search"
            
    query_builder.add_conditional_edges(
        "retrieve_and_check",
        route_retrieval_sufficiency,
        {
            "generate": "generate",
            "web_search": "web_search"
        }
    )
    
    query_builder.add_edge("extract_keywords", "call_api")
    query_builder.add_edge("call_api", "pesticide_generate_or_websearch")
    
    query_builder.add_edge("generate", END)
    query_builder.add_edge("web_search", END)
    query_builder.add_edge("pesticide_generate_or_websearch", END)
    
    return query_builder.compile()

# --- 7. Streamlit 앱 실행 로직 ---
st.set_page_config(page_title="🌱 농작물 챗봇 에이전트", layout="wide")
st.title("🌱 농작물 챗봇 에이전트")
st.markdown("궁금한 농업 관련 질문을 입력해주세요.")

# API 키 및 DB 존재 여부 확인
if not OPENAI_API_KEY or not TAVILY_API_KEY or not PESTICIDE_API_KEY:
    st.error("API 키가 설정되지 않았습니다. `.env` 파일을 확인해주세요.")
    st.stop()
    
# LangGraph 앱과 VectorStore를 한 번만 로드하도록 세션 상태 사용
if "vectorstore" not in st.session_state or "rag_app" not in st.session_state:
    with st.spinner("챗봇 시스템을 준비하는 중입니다. 잠시만 기다려주세요..."):
        try:
            setup_graph = build_initial_setup_graph()
            setup_result = setup_graph.invoke({"question": "setup"})
            st.session_state.vectorstore = setup_result.get("vectorstore")
            st.session_state.rag_app = build_query_graph()
        except Exception as e:
            st.error(f"초기화 중 오류가 발생했습니다: {e}")
            st.stop()

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중..."):
            try:
                final_state = st.session_state.rag_app.invoke({
                    "question": prompt,
                    "vectorstore": st.session_state.vectorstore
                })
                
                response = final_state.get('answer', "죄송합니다. 답변을 생성하지 못했습니다.")
                source = final_state.get('answer_source')

                # 답변 출처에 따라 메시지 추가
                if source == "web_search":
                    response = "💡 웹 검색 결과를 활용하여 답변합니다.\n\n" + response
                elif source == "pesticide_api":
                    response = "💡 농약 API 정보를 활용하여 답변합니다.\n\n" + response
                else:
                    response = "💡 내부 데이터베이스 정보를 활용하여 답변합니다.\n\n" + response

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다: {e}")