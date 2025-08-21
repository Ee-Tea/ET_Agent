# run_graph.py
import os
from typing import TypedDict, Optional, Any, Dict, List
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from langchain_core.runnables.graph import MermaidDrawMethod 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv(find_dotenv()) 

# 실행 중인 .py 파일이 있는 폴더
BASE_DIR = Path(__file__).resolve().parent  

# 상대경로로 벡터DB 지정
# 1) 경로 지정 (forward slash)
VECTOR_DB_PATH = Path("faiss_pdf_db")

print("CWD:", Path.cwd())
print("VECTOR_DB_PATH (relative):", VECTOR_DB_PATH.as_posix())
print("index.faiss 존재:", (VECTOR_DB_PATH / "index.faiss").exists())
print("index.pkl   존재:", (VECTOR_DB_PATH / "index.pkl").exists())

# 2) 임베딩 + 로드
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"
embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask"))

vectorstore = FAISS.load_local(
    VECTOR_DB_PATH.as_posix(),
    embeddings,
    allow_dangerous_deserialization=True,
)
print("✅ FAISS 벡터스토어 로드 완료")

# === 설정 ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY가 .env에 설정되어야 합니다.")

# === LangChain / LangGraph ===
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- 프롬프트 ---
PROMPT_TMPL = """
당신은 대한민국 농업 작물 재배 방법 전문가입니다.
아래 '문맥'만 사용해 질문에 답하세요.

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
rag_prompt = ChatPromptTemplate.from_template(PROMPT_TMPL)

# --- 상태 정의 ---
class GraphState(TypedDict):
    question: Optional[str]
    vectorstore: Optional[Any]
    context: Optional[str]
    answer: Optional[str]

# --- 공통 함수 ---
def load_vectorstore(db_path: str) -> Any:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

def retrieve(vs: Any, question: str, k: int = 5) -> str:
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(question)
    # 원하면 출처 표시: "\n\n".join([f"(p{d.metadata.get('page')}:{d.metadata.get('source')}) {d.page_content}" for d in docs])
    return "\n\n".join([d.page_content for d in docs])

def make_llm() -> ChatGroq:
    return ChatGroq(model_name=GROQ_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY)

# --- LangGraph 노드 ---
def load_vs_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 벡터스토어 로드")
    vs = load_vectorstore(VECTOR_DB_PATH)
    return {**state, "vectorstore": vs}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 검색")
    if not state.get("vectorstore"):
        raise ValueError("vectorstore가 없습니다.")
    q = state["question"] or ""
    ctx = retrieve(state["vectorstore"], q, k=5)
    return {**state, "context": ctx}

def generate_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 생성")
    if not state.get("context") or not state.get("question"):
        raise ValueError("context/question 누락")
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | rag_prompt
        | make_llm()
        | StrOutputParser()
    )
    ans = chain.invoke({"context": state["context"], "question": state["question"]})
    return {**state, "answer": ans}

# --- 그래프 빌드 ---
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("load_vs", load_vs_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)

    g.add_edge("load_vs", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)

    g.set_entry_point("load_vs")
    return g.compile()

if __name__ == "__main__":
    print("💬 LangGraph RAG 시작 (exit/quit 종료)")
    app = build_graph()

    # ── 그래프 시각화 ───────────────────────────────────────────
    try:
        graph_image_path = BASE_DIR / "agent_workflow_llm.png"
        png_bytes = app.get_graph().draw_mermaid_png(
            # 기본값은 Mermaid.ink API 사용. 오프라인/방화벽 환경이면 PYPPETEER가 더 안전.
            draw_method=MermaidDrawMethod.API
            # draw_method=MermaidDrawMethod.PYPPETEER,  # pyppeteer 설치 시 대안
        )
        with open(graph_image_path, "wb") as f:
            f.write(png_bytes)
        print(f"\n✅ LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        # 실패 시: ASCII 다이어그램 출력 + Mermaid 소스 .mmd로 저장 (백업)
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
            break
        if not q:
            continue
        try:
            final_state = app.invoke({"question": q})
            print("\n--- 답변 ---")
            print(final_state["answer"])
            print("------------\n")
        except Exception as e:
            print(f"❌ 오류: {e}\n")