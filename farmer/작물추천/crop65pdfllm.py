# run_graph.py
import os
from typing import TypedDict, Optional, Any, Dict, List
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv()) 

# === 설정 ===
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "faiss_pdf_db")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))

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

def run(state: dict) -> dict:
    """
    오케스트레이터에서 호출 가능한 entrypoint 함수.
    입력: {"query": 질문}
    반환: {"pred_answer": 답변}
    """
    app = build_graph()
    question = state.get("query", "")
    if not question:
        return {"pred_answer": "질문이 비어 있습니다."}
    try:
        result = app.invoke({"question": question})
        answer = result.get("answer", "답변 생성 실패")
        return {"pred_answer": answer}
    except Exception as e:
        return {"pred_answer": f"crop65pdfllm 실행 중 오류: {e}"}

if __name__ == "__main__":
    print("💬 LangGraph RAG 시작 (exit/quit 종료)")
    app = build_graph()
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
