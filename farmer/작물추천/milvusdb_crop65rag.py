import os
import json
from typing import TypedDict, Optional, Any, Dict, List
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import hashlib

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document

# ──────────────────────────────────────────────────────────────────────────────
# 환경 변수 로드
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())

# Milvus / Embedding 설정
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "hongyoungjun")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")

# 입력 문서 폴더
DOCS_DIR = Path(os.getenv("DOCS_DIR", r"C:\Rookies_project\cropinfo"))
DOCS_DIR.mkdir(parents=True, exist_ok=True) 

# 청크/임베딩 파라미터
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# ──────────────────────────────────────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)
EMBEDDING_DIM = len(embedding_model.embed_query("test"))

# ──────────────────────────────────────────────────────────────────────────────
# 상태 정의
# ──────────────────────────────────────────────────────────────────────────────
class IngestState(TypedDict):
    docsPath: str
    files: List[str]
    rawDocs: List[Document]
    vectorstore: Optional[Milvus]
    inserted: int
    collectionName: str

# ──────────────────────────────────────────────────────────────────────────────
# LangGraph 노드들
# ──────────────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import MermaidDrawMethod
from pymilvus import connections, MilvusClient, DataType

def ensure_milvus_node(state: IngestState) -> IngestState:
    print("🧩 노드: ensure_milvus (컬렉션 확인)")
    
    if "default" in connections.list_connections():
        connections.disconnect("default")
    connections.connect(alias="default", host="localhost", port="19530")

    # 기존 컬렉션이 있다면 삭제 (테스트를 위해)
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    if client.has_collection(MILVUS_COLLECTION):
        print(f"  ↪ 기존 컬렉션 '{MILVUS_COLLECTION}' 삭제")
        client.drop_collection(MILVUS_COLLECTION)
    
    print(f"  ↪ Milvus 연결 및 컬렉션 준비 완료.")
    
    return state

def list_files_node(state: IngestState) -> IngestState:
    print("🧩 노드: list_files")
    docs_path = Path(state["docsPath"])
    allow_ext = {".txt", ".md", ".pdf"}
    files = [str(p) for p in sorted(docs_path.rglob("*")) if p.is_file() and p.suffix.lower() in allow_ext]
    print(f"  ↪ 대상 파일 {len(files)}개")
    if not files:
        print("  ⚠️ 'ingest_docs' 폴더에 .txt/.md/.pdf 파일을 넣어주세요.")
    return {**state, "files": files}

def load_and_ingest_node(state: IngestState) -> IngestState:
    print("🧩 노드: load_and_ingest_node (문서 로드 & Milvus에 인제스트)")
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    for fp in state["files"]:
        path = Path(fp)
        try:
            if path.suffix.lower() in {".txt", ".md"}:
                docs = TextLoader(str(path), autodetect_encoding=True).load()
            elif path.suffix.lower() == ".pdf":
                docs = PyPDFLoader(str(path)).load()
            else:
                continue

            for d in docs:
                d.metadata.setdefault("source", str(path.name))
                if "page" not in d.metadata:
                    d.metadata["page"] = 0
            
            # 여기서 바로 청킹
            chunks = text_splitter.split_documents(docs)
            all_docs.extend(chunks)
            print(f"  ↪ 로드 및 청크 완료: {path.name} ({len(chunks)}개 청크)")
            
        except Exception as e:
            print(f"  ⚠️ 처리 실패: {path.name} | {e}")
            
    if not all_docs:
        print("  ⚠️ 처리할 문서가 없습니다.")
        return {**state, "inserted": 0, "rawDocs": [], "vectorstore": None}

    print(f"🆕 총 {len(all_docs)}개 청크를 벡터스토어에 삽입 중...")
    
    try:
        vectorstore = Milvus.from_documents(
            documents=all_docs,
            embedding=embedding_model,
            collection_name=state["collectionName"],
            connection_args={"host": "localhost", "port": "19530"}
        )
        print("✅ 벡터스토어 생성 및 문서 삽입 완료.")
        
        inserted_count = len(all_docs)
        
    except Exception as e:
        print(f"❌ Milvus 삽입 오류: {e}")
        inserted_count = 0
        vectorstore = None

    return {**state, "inserted": inserted_count, "rawDocs": all_docs, "vectorstore": vectorstore}

# ──────────────────────────────────────────────────────────────────────────────
# 그래프 빌드
# ──────────────────────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(IngestState)
    g.add_node("ensure_milvus", ensure_milvus_node)
    g.add_node("list_files", list_files_node)
    g.add_node("load_and_ingest", load_and_ingest_node)

    g.set_entry_point("ensure_milvus")
    g.add_edge("ensure_milvus", "list_files")
    g.add_edge("list_files", "load_and_ingest")
    g.add_edge("load_and_ingest", END)
    
    return g.compile()

# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("🚀 LangGraph 기반 Milvus Ingest 파이프라인 시작")
    
    agent_app = build_graph()
    
    try:
        graph_image_path = "milvus_agent_workflow_rag.png"
        png_bytes = agent_app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        with open(graph_image_path, "wb") as f:
            f.write(png_bytes)
        print(f"\n✅ LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
        
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")

    initial_state: IngestState = {
        "docsPath": str(DOCS_DIR),
        "files": [],
        "rawDocs": [],
        "vectorstore": None,
        "inserted": 0,
        "collectionName": MILVUS_COLLECTION,
    }
    
    final_state = agent_app.invoke(initial_state)

    print("\n📦 결과 요약")
    print(f"  - 처리된 파일 수: {len(final_state['files'])}")
    print(f"  - Milvus 컬렉션: {final_state['collectionName']}")
    print(f"  - 삽입된 청크 수: {final_state['inserted']}")

if __name__ == "__main__":
    main()