import os
import json
from typing import TypedDict, Optional, Any, Dict, List
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from langchain_core.runnables.graph import MermaidDrawMethod 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# --- 웹 검색에 필요한 라이브러리 추가 ---
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
# --- 채팅 기록 관련 라이브러리 추가 ---
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# ----------------------------------------
load_dotenv(find_dotenv()) 

# 실행 중인 .py 파일이 있는 폴더
BASE_DIR = Path(__file__).resolve().parent

# 상대경로로 벡터DB 지정
# 1) 경로 지정 (forward slash)
VECTOR_DB_PATH = Path("faiss_pdf_db")
CHAT_HISTORY_PATH = Path("chat_history.json") # 대화 기록 파일 경로 추가

print("CWD:", Path.cwd())
print("VECTOR_DB_PATH (relative):", VECTOR_DB_PATH.as_posix())
print("index.faiss 존재:", (VECTOR_DB_PATH / "index.faiss").exists())
print("index.pkl 존재:", (VECTOR_DB_PATH / "index.pkl").exists())

# 2) 임베딩 + 로드
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"
embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask"))

try:
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH.as_posix(),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("✅ FAISS 벡터스토어 로드 완료")
except Exception as e:
    print(f"❌ FAISS 벡터스토어 로드 실패: {e}")
    vectorstore = None

# === 설정 ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
# --- Tavily API 키 설정 ---
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY가 .env에 설정되어야 합니다.")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY가 .env에 설정되어야 합니다.")

# === LangChain / LangGraph ===

# --- 프롬프트 (채팅 기록 추가) ---
RAG_PROMPT_TMPL = """
당신은 대한민국 농업 작물 재배 방법 전문가입니다.
아래 '문맥'과 '대화 기록'만 사용해 질문에 답하세요.

[대화 기록]
{chat_history}

[문맥]
{context}

규칙:
- 문맥에 없는 정보/추측/한자 금지.
- 한글로만 작성.
- 단계/설명은 "한 문장씩 줄바꿈".
- 문맥에 근거 없으면: "주어진 정보로는 답변할 수 없습니다."

질문: {question}
답변:
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL)

WEB_SEARCH_PROMPT_TMPL = """
당신은 대한민국 농업 관련 지식과 일반 상식에 해박한 전문가입니다.
아래 '대화 기록'과 '검색 결과'를 바탕으로 '질문'에 대해 한국어로 명확하고 간결하게 답변하세요.

[대화 기록]
{chat_history}

[검색 결과]
{search_results}

질문: {question}
답변:
"""
web_search_prompt = PromptTemplate.from_template(WEB_SEARCH_PROMPT_TMPL)

QUESTION_TYPE_PROMPT_TMPL = """
주어진 질문이 "대한민국 농업 작물 재배 방법"에 관련된 질문이면 'true'를, 그 외의 일반적인 질문이면 'false'를 반환하세요.
'true' 또는 'false' 두 단어 중 하나만 반환해야 합니다. 다른 설명은 금지합니다.

[대화 기록]
{chat_history}

질문: {question}
답변:
"""
question_type_prompt = ChatPromptTemplate.from_template(QUESTION_TYPE_PROMPT_TMPL)
# -----------------------------

# --- 상태 정의 (chat_history 추가) ---
class GraphState(TypedDict):
    question: Optional[str]
    chat_history: List[BaseMessage]
    vectorstore: Optional[Any]
    context: Optional[str]
    answer: Optional[str]
    is_context_sufficient: Optional[bool]
    is_crop_question: Optional[bool]

# --- 공통 함수 ---
def load_vectorstore(db_path: str) -> Any:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    try:
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"❌ 벡터스토어 로드 실패: {e}")
        return None

def retrieve(vs: Any, question: str, k: int = 5) -> str:
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return "\n\n".join([d.page_content for d in docs])

def make_llm() -> ChatGroq:
    return ChatGroq(model_name=GROQ_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY)

# --- 채팅 기록을 LLM이 이해하기 쉬운 문자열로 변환하는 헬퍼 함수 ---
def format_chat_history(history: List[BaseMessage]) -> str:
    formatted_history = ""
    for message in history:
        if isinstance(message, HumanMessage):
            formatted_history += f"사용자: {message.content}\n"
        elif isinstance(message, AIMessage):
            formatted_history += f"AI: {message.content}\n"
    return formatted_history

# --- LangGraph 노드 ---
def load_vs_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 벡터스토어 로드")
    vs = load_vectorstore(VECTOR_DB_PATH)
    return {**state, "vectorstore": vs}

def is_crop_question_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 질문 유형 분류")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    formatted_chat_history = format_chat_history(chat_history)
    
    chain = question_type_prompt | make_llm() | StrOutputParser()
    try:
        result = chain.invoke({"question": question, "chat_history": formatted_chat_history}).strip().lower()
        is_crop_q = result == "true"
        print(f"LLM이 판단한 질문 유형: {'작물 관련' if is_crop_q else '일반 질문'}")
    except Exception as e:
        print(f"❌ LLM 질문 분류 실패, 기본값(일반 질문)으로 설정: {e}")
        is_crop_q = False
        
    return {**state, "is_crop_question": is_crop_q}

def retrieve_and_generate_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 검색 및 답변 생성")
    vs = state.get("vectorstore")
    if not vs:
        return {**state, "answer": "벡터스토어를 찾을 수 없습니다."}

    q = state["question"] or ""
    chat_history = state.get("chat_history", [])
    
    formatted_chat_history = format_chat_history(chat_history)
    
    # 여기서 대화 기록을 기반으로 질문을 확장하는 로직을 추가할 수 있습니다.
    # 예: "그거" -> "가지"로 변환
    # 현재는 단순히 전체 대화 기록을 retriever에 던집니다.
    retrieval_query = formatted_chat_history + "\n" + q if formatted_chat_history else q
    ctx = retrieve(vs, retrieval_query, k=5)
    
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
        | rag_prompt
        | make_llm()
        | StrOutputParser()
    )
    ans = chain.invoke({"context": ctx, "question": q, "chat_history": formatted_chat_history})
    
    return {**state, "context": ctx, "answer": ans}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 웹 검색")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    tavily_tool = TavilySearchResults(max_results=5)
    
    formatted_chat_history = format_chat_history(chat_history)
    
    print("🔍 웹 검색 수행 중...")
    try:
        search_results = tavily_tool.invoke({"query": question})
        search_results_str = "\n".join([str(res) for res in search_results])
    except Exception as e:
        print(f"❌ 웹 검색 오류 발생: {e}")
        return {**state, "answer": "웹 검색 중 오류가 발생했습니다."}
    
    print("💡 웹 검색 결과를 바탕으로 답변 생성 중...")
    chain = (
        {"search_results": RunnablePassthrough(), "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
        | web_search_prompt
        | make_llm()
        | StrOutputParser()
    )
    web_answer = chain.invoke({"search_results": search_results_str, "question": question, "chat_history": formatted_chat_history})
    
    return {**state, "answer": web_answer}

# --- 조건부 라우팅 함수 ---
def decide_route_from_type(state: GraphState) -> str:
    print("🧩 노드: 경로 결정 (질문 유형)")
    is_crop_question = state.get("is_crop_question")
    
    if is_crop_question:
        print("결정: 작물 관련 질문이므로 내부 DB 검색 및 답변 생성을 시도합니다.")
        return "retrieve_and_generate"
    else:
        print("결정: 일반 질문이므로 웹 검색으로 진행합니다.")
        return "web_search"

# --- 수정된 decide_route_from_retrieve 함수 ---
def decide_route_from_retrieve(state: GraphState) -> str:
    print("🧩 노드: 경로 결정 (DB 검색 결과)")
    answer = state.get("answer")
    
    if "주어진 정보로는 답변할 수 없습니다." not in answer:
        print("결정: 내부 DB 정보로 답변을 생성했으므로 종료합니다.")
        return "end"
    else:
        print("결정: 내부 DB에 정보가 부족하므로 웹 검색으로 진행합니다.")
        return "web_search"

# --- 그래프 빌드 ---
def build_graph():
    g = StateGraph(GraphState)
    
    # 노드 추가
    g.add_node("load_vs", load_vs_node)
    g.add_node("is_crop_question", is_crop_question_node)
    g.add_node("retrieve_and_generate", retrieve_and_generate_node)
    g.add_node("web_search", web_search_node)

    # 엣지 연결
    g.set_entry_point("load_vs")
    
    g.add_edge("load_vs", "is_crop_question")
    
    g.add_conditional_edges(
        "is_crop_question",
        decide_route_from_type,
        {
            "retrieve_and_generate": "retrieve_and_generate",
            "web_search": "web_search"
        }
    )
    
    g.add_conditional_edges(
        "retrieve_and_generate",
        decide_route_from_retrieve,
        {
            "end": END,
            "web_search": "web_search"
        }
    )
    
    g.add_edge("web_search", END)

    return g.compile()

# --- JSON 파일 관련 함수 ---
def load_chat_history() -> List[BaseMessage]:
    """JSON 파일에서 채팅 기록을 불러와 LangChain 메시지 객체로 변환합니다."""
    if CHAT_HISTORY_PATH.exists():
        try:
            with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            
            chat_history = []
            for item in history_data:
                if item["role"] == "user":
                    chat_history.append(HumanMessage(content=item["content"]))
                elif item["role"] == "assistant":
                    chat_history.append(AIMessage(content=item["content"]))
            print(f"✅ 기존 대화 기록 ({len(chat_history)}개)을 로드했습니다.")
            return chat_history
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"❌ 대화 기록 파일 로드 오류: {e}. 새 대화를 시작합니다.")
    return []

def save_chat_history(history: List[BaseMessage]):
    """LangChain 메시지 객체를 JSON 파일로 저장합니다."""
    try:
        history_data = []
        for message in history:
            if isinstance(message, HumanMessage):
                history_data.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history_data.append({"role": "assistant", "content": message.content})
        
        with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=4)
        print("✅ 대화 기록을 'chat_history.json' 파일에 저장했습니다.")
    except Exception as e:
        print(f"❌ 대화 기록 저장 오류: {e}")

if __name__ == "__main__":
    print("💬 LangGraph RAG + WebSearch 시작 (exit/quit 종료)")
    app = build_graph()

    # 기존 대화 기록을 로드합니다.
    chat_history: List[BaseMessage] = load_chat_history()

    # ── 그래프 시각화 ───────────────────────────────────────────
    try:
        graph_image_path = BASE_DIR / "agent_workflow_llm.png"
        png_bytes = app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
        )
        with open(graph_image_path, "wb") as f:
            f.write(png_bytes)
        print(f"\n✅ LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"⚠️ 그래프 시각화 중 오류 발생: {e}")
        try:
            ascii_map = app.get_graph().draw_ascii()
            print("\n[ASCII Graph]")
            print(ascii_map)
            mermaid_src = app.get_graph().draw_mermaid()
            mmd_path = BASE_DIR / "agent_workflow.mmd"
            with open(mmd_path, "w", encoding="utf-8") as f:
                f.write(mermaid_src)
            print(f"📝 Mermaid 소스를 '{mmd_path}'로 저장했습니다. (mermaid.live 등에서 렌더 가능)")
        except Exception as e2:
            print(f"추가 백업도 실패: {e2}")
    # ───────────────────────────────────────────────────────────

    while True:
        q = input("질문> ").strip()
        if q.lower() in ("exit", "quit"):
            save_chat_history(chat_history)
            break
        if not q:
            continue
        try:
            # 질문과 채팅 기록을 함께 전달
            final_state = app.invoke({"question": q, "chat_history": chat_history})
            answer = final_state["answer"]
            
            print("\n--- 답변 ---")
            print(answer)
            print("------------\n")
            
            # 현재 질문과 답변을 채팅 기록에 추가
            chat_history.append(HumanMessage(content=q))
            chat_history.append(AIMessage(content=answer))
            
        except Exception as e:
            print(f"❌ 오류: {e}\n")