# ──────────────────────────────────────────────────────────────────────────────
# 임포트 섹션: 필요한 라이브러리 및 모듈을 가져옵니다.
# ──────────────────────────────────────────────────────────────────────────────
import os  # 파일 경로, 환경 변수 등 운영체제 기능에 접근
import sys  # 진행률 출력용
import time  # ETA 계산용
import math  # 시간 포맷 등에 사용
from typing import TypedDict, Optional, List
from pathlib import Path  # 파일 시스템 경로를 객체지향적으로 다루기 위해 사용
from dotenv import load_dotenv, find_dotenv  # .env 로더
from collections import defaultdict  # ✨ 파일별로 문서를 그룹화하기 위해 사용합니다.

# LangChain과 관련된 클래스들을 임포트합니다.
from langchain_community.document_loaders import TextLoader, PyPDFLoader  # 파일 로더
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 텍스트 분할기
from langchain_huggingface import HuggingFaceEmbeddings  # 임베딩 모델
from langchain_community.vectorstores import Milvus  # Milvus 벡터스토어
from langchain_core.documents import Document  # 문서 구조

# ──────────────────────────────────────────────────────────────────────────────
# 환경 변수 로드 및 기본 설정
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())

# Milvus / Embedding 설정
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "test")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")

# 입력 문서 폴더
DOCS_DIR = Path(os.getenv("DOCS_DIR", r"C:\Rookies_project\cropinfo"))
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# 청크/임베딩 파라미터
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# ✨ 임베딩 진행률 설정(환경변수로 조정 가능)
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_PROGRESS_INTERVAL = float(os.getenv("EMBED_PROGRESS_INTERVAL", "0.2"))

# ──────────────────────────────────────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)
EMBEDDING_DIM = len(embedding_model.embed_query("test"))

# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────────────────────────────────────
def _format_eta(seconds: Optional[float]) -> str:
    if not seconds or seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--:--"
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _render_progress(prefix: str, done: int, total: int, start_ts: float, task_name: Optional[str] = None) -> None:
    done = min(done, total)
    percent = int((done / total) * 100) if total else 100
    elapsed = time.time() - start_ts
    rate = (done / elapsed) if elapsed > 0 else None
    remain = ((total - done) / rate) if rate else None
    eta = _format_eta(remain)
    bar_len = 24
    filled = int(bar_len * percent / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    task_str = f"| {task_name}" if task_name else ""
    full_line = f"{prefix} [{bar}] {percent:3d}%  ({done}/{total})  ETA {eta}{task_str}"
    padded_line = full_line.ljust(120)

    sys.stdout.write(f"\r{padded_line}")
    sys.stdout.flush()
    
    if done >= total:
        sys.stdout.write("\n")
        
class ProgressEmbeddings:
    # ✨ __init__ 생성자에서 total_texts와 task_name을 제거했습니다.
    def __init__(self, base: HuggingFaceEmbeddings, batch_size: int = 32, desc: str = "임베딩"):
        self.base = base
        self.batch_size = max(1, batch_size)
        self.desc = desc
        # 초기화 시에는 비워두거나 기본값으로 설정합니다.
        self.task_name = ""
        self.total_texts = 1
        self._last_print = 0.0

    # ✨ 파일 정보를 갱신하는 update_task 메서드입니다.
    def update_task(self, task_name: str, total_texts: int):
        """진행률 표시줄에 표시될 작업 이름과 전체 개수를 업데이트합니다."""
        self.task_name = task_name
        self.total_texts = max(total_texts, 1)
        self._last_print = 0.0

    def embed_query(self, text: str) -> List[float]:
        return self.base.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        n = len(texts)
        # ✨ __init__이 아닌 update_task에서 설정된 값으로 total을 사용합니다.
        total = self.total_texts
        start_ts = time.time()
        results: List[List[float]] = []
        processed = 0

        task_str = f" ({self.task_name})" if self.task_name else ""
        print(f"🧮 {self.desc}{task_str} 시작: 총 {total}개 청크 | 배치 {self.batch_size}")

        for i in range(0, n, self.batch_size):
            batch = texts[i:i + self.batch_size]
            emb = self.base.embed_documents(batch)
            results.extend(emb)
            processed = min(i + len(batch), total)

            now = time.time()
            if now - self._last_print >= EMBED_PROGRESS_INTERVAL or processed == total:
                _render_progress(f"🔄 {self.desc}", processed, total, start_ts, task_name=self.task_name)
                self._last_print = now

        _render_progress(f"✅ {self.desc}", total, total, start_ts, task_name=self.task_name)
        return results

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
    connections.connect(host="localhost", port="19530")

    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    if client.has_collection(MILVUS_COLLECTION):
        print(f"  ↪ 기존 컬렉션 '{MILVUS_COLLECTION}' 삭제")
        client.drop_collection(MILVUS_COLLECTION)
    print(f"  ↪ Milvus 연결 및 컬렉션 준비 완료.")
    return state  
    
def list_files_node(state: IngestState) -> IngestState:
    print("🧩 노드: list_files")
    docs_path = Path(state["docsPath"])
    allow_ext = {".txt", ".md", ".pdf"}
    files = [str(p) for p in sorted(docs_path.rglob("*")) if p.is_file() and p.suffix.lower() in allow_ext]
    print(f"  ↪ 대상 파일 {len(files)}개")
    if not files:
        print("  ⚠️ 'ingest_docs' 폴더에 .txt/.md/.pdf 파일을 넣어주세요.")
    return {**state, "files": files}


def load_and_ingest_node(state: IngestState) -> IngestState:
    print("🧩 노드: load_and_ingest_node (문서 로드 & Milvus에 인제스트)")
    all_docs: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    # --- 1단계: 문서 로딩 & 청크 분할 ---
    # (이 부분은 기존 코드와 동일합니다)
    files_to_process = state["files"]
    total_files = len(files_to_process)
    if total_files == 0:
        print("  ⚠️ 처리할 파일이 없습니다.")
        return {**state, "inserted": 0, "rawDocs": [], "vectorstore": None}

    print(f"\n--- [ 1단계: 문서 로딩 & 청크 분할 ] ---")
    load_start_ts = time.time()
    completion_logs = []
    
    for i, fp in enumerate(files_to_process):
        path = Path(fp)
        _render_progress("🔄 로드 & 청크", i, total_files, load_start_ts, task_name=path.name)
        try:
            if path.suffix.lower() in [".txt", ".md"]:
                docs = TextLoader(str(path), autodetect_encoding=True).load()
            elif path.suffix.lower() == ".pdf":
                docs = PyPDFLoader(str(path)).load()
            else:
                continue

            chunks = text_splitter.split_documents(docs)
            all_docs.extend(chunks)
            completion_logs.append(f"  - {path.name} 로드 완료 ({len(chunks)}개 청크)")
        except Exception as e:
            completion_logs.append(f"  - {path.name} ❌ 로드 오류")
            sys.stdout.write("\n")
            print(f"  └─ ❌ 오류 발생: {path.name} | {e}")
            
    _render_progress("✅ 로드 & 청크", total_files, total_files, load_start_ts, task_name="완료")
    
    print("\n--- 개별 파일 로드 결과 ---")
    for log_entry in completion_logs:
        print(log_entry)
    print("--------------------------\n")

    if not all_docs:
        print("  ⚠️ 처리할 문서가 없습니다.")
        return {**state, "inserted": 0, "rawDocs": [], "vectorstore": None}

    # --- 2단계: 임베딩 & DB 삽입 ---
    print(f"--- [ 2단계: 임베딩 & DB 삽입 ] ---")
    
    # 모든 청크를 원본 파일 이름 기준으로 그룹화합니다. (이 부분은 수정할 필요 없습니다)
    docs_by_source = defaultdict(list)
    for doc in all_docs:
        source_name = os.path.basename(doc.metadata.get("source", "unknown"))
        docs_by_source[source_name].append(doc)
    
    # ✨ 1. 루프 시작 전에 ProgressEmbeddings 객체를 한 번만 생성합니다.
    progress_embedder = ProgressEmbeddings(
        base=embedding_model,
        batch_size=EMBED_BATCH_SIZE,
        desc="임베딩"
    )

    vectorstore = None
    inserted_count = 0
    
    total_source_files = len(docs_by_source)
    processed_source_files = 0
    # 파일 그룹별로 순회하며 임베딩 및 삽입 진행
    for source_name, doc_list in docs_by_source.items():
        processed_source_files += 1
        print(f"\n[{processed_source_files}/{total_source_files}] '{source_name}' 파일 처리 시작...")
        try:
            # ✨ 2. 루프 안에서는 객체의 정보를 업데이트합니다.
            progress_embedder.update_task(
                task_name=source_name,
                total_texts=len(doc_list)
            )

            if vectorstore is None:
                # 처음에는 vectorstore를 생성하면서 progress_embedder를 등록합니다.
                vectorstore = Milvus.from_documents(
                    documents=doc_list,
                    embedding=progress_embedder,
                    collection_name=state["collectionName"],
                    connection_args={"host": "localhost", "port": "19530"}
                )
            else:
                # ✨ 3. 두 번째 파일부터는 embedding 인자를 넘기지 않습니다.
                # vectorstore는 내부에 저장된 progress_embedder를 자동으로 사용합니다.
                vectorstore.add_documents(doc_list)
            
            inserted_count += len(doc_list)

        except Exception as e:
            print(f"❌ '{source_name}' 파일 임베딩/삽입 중 오류 발생: {e}")

    print("\n✅ 모든 파일의 임베딩 및 벡터스토어 삽입 완료.")

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
# 메인 실행 함수
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
    print(f"  - 처리된 파일 수: {len(final_state['files'])}")
    print(f"  - Milvus 컬렉션: {final_state['collectionName']}")
    print(f"  - 삽입된 청크 수: {final_state['inserted']}")

if __name__ == "__main__":
    main()
