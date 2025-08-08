import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables.graph import MermaidDrawMethod

# --- 1. 환경 설정 ---
load_dotenv()
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY=REDACTED ValueError("OPENAI_API_KEY=REDACTED 파일을 확인해주세요.")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

print(f"OPENAI_API_KEY=REDACTED[:5]}...")
print(f"TAVILY_API_KEY 로드 완료: {TAVILY_API_KEY[:5]}...")

# 데이터 파일 경로 및 벡터 DB 저장 경로
CSV_FILE_PATH = 'data/농림수산식품교육문화정보원_영농가이드 재배정보_20230911.csv'
VECTOR_DB_PATH = 'faiss_crop_guide_db'

if not os.path.exists('data'):
    os.makedirs('data')

# --- 2. LLM 및 프롬프트 설정 ---
llm = ChatGroq(model_name="llama3-70b-8192",
               temperature=0.3,
               api_key=OPENAI_API_KEY=REDACTED = """
당신은 사용자의 질문이 '농업 및 작물 재배 방법'에 관련이 있는지 없는지 분류하는 전문가입니다.

질문을 분석하여 다음 규칙에 따라 답변하세요.
- 질문이 농업 또는 작물 재배에 관련이 있다면, 'agriculture'라고만 답변하세요.
- 그 외의 모든 일반적인 질문이라면, 'other'라고만 답변하세요.
- 답변은 한 단어만 출력해야 합니다.

질문: {question}
답변:
"""
classify_prompt = ChatPromptTemplate.from_template(CLASSIFY_PROMPT_TEMPLATE)

# RAG 프롬프트 (기존과 동일)
RAG_PROMPT_TEMPLATE = """
당신은 대한민국 농업 작물 재배 방법 전문가입니다. 

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

# 웹 검색 결과를 요약하기 위한 새로운 프롬프트
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

***REMOVED*** 웹 검색 도구 설정
tavily_tool = TavilySearchResults(max_results=5, api_key = TAVILY_API_KEY)

# --- 3. LangGraph 상태 정의 ---
class GraphState(TypedDict):
    question: Optional[str]
    documents: Optional[List[Dict[str, Any]]]
    splits: Optional[List[Dict[str, Any]]]
    vectorstore: Optional[FAISS]
    context: Optional[str]
    answer: Optional[str]
    search_results: Optional[str]
    classification: Optional[str]

# --- 4. 핵심 기능 함수 정의 (기존과 동일) ---
def load_csv_data(file_path: str) -> List[Dict[str, Any]]:
    print("---기능: 데이터 로드 시작---")
    loader = CSVLoader(file_path=file_path, encoding='utf-8')
    documents = loader.load()
    print(f"로드된 문서 개수: {len(documents)}개")
    return documents

def split_documents(documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
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

def retrieve_relevant_chunks(vectorstore: FAISS, question: str) -> str:
    print("---기능: 관련 청크 유사도 검색 시작---")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"검색된 청크 수: {len(docs)}개")
    return context

def generate_answer_with_llm(context: str, question: str, llm: ChatGroq, prompt_template: ChatPromptTemplate) -> str:
    print("---기능: 답변 생성 시작---")
    rag_chain_internal = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    inputs = {"context": context, "question": question}
    answer = rag_chain_internal.invoke(inputs)
    return answer

# --- 5. LangGraph 노드 함수 정의 ---
def load_data_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 데이터 로드 (Load Data) 실행---")
    documents = load_csv_data(CSV_FILE_PATH)
    return {**state, "documents": documents}

def split_documents_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 문서 분할 (Split Documents) 실행---")
    if "documents" not in state or not state["documents"]: 
        raise ValueError("문서 분할을 위해 로드된 문서가 필요합니다.")
    splits = split_documents(state["documents"])
    return {**state, "splits": splits} 

def embed_and_store_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 임베딩 및 벡터 DB 저장 (Embed & Store) 실행---")
    if "splits" not in state or not state["splits"]:
        raise ValueError("임베딩을 위해 분할된 청크가 필요합니다.")
    vectorstore = embed_and_store_vector_db(state["splits"], VECTOR_DB_PATH)
    return {**state, "vectorstore": vectorstore}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 검색 (Retrieve) 실행---")
    if "vectorstore" not in state or not state["vectorstore"]:
        raise ValueError("검색을 위해 벡터 DB가 필요합니다.")
    question = state["question"] 
    vectorstore = state["vectorstore"]
    context = retrieve_relevant_chunks(vectorstore, question)
    return {**state, "context": context}

def generate_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 생성 (Generate) 실행---")
    if ("context" not in state or not state["context"] or 
        "question" not in state or not state["question"]):
        raise ValueError("답변 생성을 위해 문맥과 질문이 필요합니다.")
    context = state["context"]
    question = state["question"]
    answer = generate_answer_with_llm(context, question, llm, rag_prompt)
    return {**state, "answer": answer}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 웹 검색 (Web Search) 실행---")
    question = state["question"]
    
    # 1. Tavily 검색 실행
    search_results = tavily_tool.invoke({"query": question})
    search_results_str = "\n".join([str(res) for res in search_results])

    # 2. 검색 결과를 일정 크기로 잘라냄 (토큰 제한 회피)
    # 2000자는 약 500~700 토큰에 해당하여 안전하게 처리 가능
    truncated_search_results = search_results_str[:2000]
    print(f"검색 결과를 {len(truncated_search_results)}자로 잘라냄.")
    print(f"잘린 검색 결과: {truncated_search_results[:100]}...")
    
    # 3. 잘린 검색 결과로 답변 생성
    web_answer = generate_answer_with_llm(truncated_search_results, question, llm, web_search_prompt)
    
    return {**state, "answer": web_answer}

def classify_question_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 질문 분류 (Classify Question) 실행---")
    question = state["question"]
    chain = classify_prompt | llm | StrOutputParser()
    classification = chain.invoke({"question": question})
    print(f"분류 결과: '{classification.strip()}'")
    return {**state, "classification": classification.strip()}

# --- 6. LangGraph 워크플로우 빌드 및 컴파일 ---
def build_initial_setup_graph():
    """초기 문서 로딩 및 벡터스토어 구축을 위한 그래프를 빌드합니다."""
    print("📚 초기 설정(문서 로딩 및 벡터스토어 구축) 흐름 구성 중...")
    initial_builder = StateGraph(GraphState)
    initial_builder.add_node("load_data", load_data_node)
    initial_builder.add_node("split_documents", split_documents_node)
    initial_builder.add_node("embed_and_store", embed_and_store_node)
    initial_builder.set_entry_point("load_data")
    initial_builder.add_edge("load_data", "split_documents")
    initial_builder.add_edge("split_documents", "embed_and_store")
    initial_builder.add_edge("embed_and_store", END)
    return initial_builder.compile()

def build_query_graph():
    """질문 분류, RAG, 웹 검색을 통합한 메인 질의 그래프를 빌드합니다."""
    print("📚 메인 질의 흐름 구성 중...")
    query_builder = StateGraph(GraphState)
    
    # 노드 추가
    query_builder.add_node("classify_question", classify_question_node)
    query_builder.add_node("retrieve", retrieve_node)
    query_builder.add_node("generate", generate_node)
    query_builder.add_node("web_search", web_search_node)
    
    # 시작점 설정
    query_builder.set_entry_point("classify_question")
    
    # **변경됨**: 분류 결과에 따라 라우팅 로직을 변경합니다.
    def route_classification(state: GraphState):
        if state.get("classification") == "agriculture":
            # 농업 관련 질문 -> RAG로 바로 이동
            return "agriculture"
        else:
            # 기타 질문 -> 웹 검색으로 바로 이동
            return "other"
            
    query_builder.add_conditional_edges(
        "classify_question",
        route_classification,
        {
            "agriculture": "retrieve",
            "other": "web_search"
        }
    )
    
    # **변경됨**: RAG 파이프라인 이후 무조건 종료하도록 변경 (웹 검색으로 전환하지 않음)
    query_builder.add_edge("retrieve", "generate")
    query_builder.add_edge("generate", END)
    
    # 웹 검색 후 종료
    query_builder.add_edge("web_search", END)
    
    return query_builder.compile()

# --- 7. 메인 실행 부분 ---
if __name__ == "__main__":
    print("\n---농작물 챗봇 에이전트 시작 (종료하려면 'exit' 또는 'quit' 입력)---")

    # 1. 초기 설정: 데이터 로드 및 벡터스토어 구축 (한 번만 실행)
    setup_graph = build_initial_setup_graph()
    initial_state = {"question": "setup"}
    setup_result = setup_graph.invoke(initial_state)
    vectorstore = setup_result.get("vectorstore")

    # 2. 메인 질의 그래프 준비 (벡터스토어 주입)
    app = build_query_graph()

    # 그래프 시각화 및 저장
    graph_image_path = "agent_workflow.png"
    try:
        graph_data = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        with open(graph_image_path, "wb") as f:
            f.write(graph_data)
        print(f"\n✅ LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")
        print("Graphviz 설치 및 시스템 PATH 설정을 확인하거나, 인터넷 연결 상태를 점검해주세요.")

    while True:
        user_question = input("\n질문을 입력하세요: ")
        if user_question.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다.")
            break

        if not user_question.strip():
            print("질문을 입력해주세요.")
            continue

        try:
            # 벡터스토어를 state에 함께 전달합니다.
            final_state = app.invoke({"question": user_question, "vectorstore": vectorstore})
            print("\n---챗봇 답변---")
            print(final_state['answer'])

        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")

    print("\n---농작물 챗봇 에이전트 종료---")