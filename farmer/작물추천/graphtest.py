import os
from glob import glob
from typing import List, Any, TypedDict
import json
import hashlib
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# === 설정 ===
PDF_DIR = os.getenv("PDF_DIR", "./cropinfo")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "faiss_pdf_db")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# === LangChain ===
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# === LangGraph ===
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

# === 신규/변경 파일만 처리하기 위한 유틸 ===
MANIFEST_FILE = "manifest.json"

def _abs(p: str) -> str:
    return str(Path(p).resolve())

def _sha256_file(path: str, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _manifest_path(db_path: str) -> str:
    return os.path.join(db_path, MANIFEST_FILE)

def _load_manifest(db_path: str) -> dict:
    try:
        with open(_manifest_path(db_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def _save_manifest(db_path: str, manifest: dict) -> None:
    os.makedirs(db_path, exist_ok=True)
    with open(_manifest_path(db_path), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

# === Graph State 정의 ===
class GraphState(TypedDict):
    """
    RAG 인덱스 빌드 그래프의 상태를 나타내는 클래스.
    """
    documents: List[Document]
    splits: List[Document]
    error: str

# === 단계별 노드 함수 ===
def load_all_pdfs_node(state: GraphState) -> GraphState:
    """
    PDF 파일을 로드하고, 변경된 파일만 식별하여 상태에 저장하는 노드.
    """
    print("📥 [1/4] PDF 로드 단계 시작")
    paths = sorted(glob(os.path.join(PDF_DIR, "*.pdf")))
    if not paths:
        return {"error": f"PDF가 없습니다: {PDF_DIR}"}
    print(f"📂 총 파일 수: {len(paths)}")

    manifest = _load_manifest(VECTOR_DB_PATH)
    all_docs = []
    skipped, to_process = 0, 0

    for idx, p in enumerate(paths, start=1):
        abs_path = _abs(p)
        try:
            file_hash = _sha256_file(abs_path)

            if manifest.get(abs_path, {}).get("sha256") == file_hash:
                skipped += 1
                print(f"[{idx}/{len(paths)}] ⏭️ 변경 없음: {os.path.basename(p)} (skip)")
                continue

            print(f"[{idx}/{len(paths)}] 📥 파일 로드 중: {os.path.basename(p)}")
            docs = PyPDFLoader(abs_path).load()

            for d in docs:
                d.metadata["source"] = abs_path
                d.metadata["file_sha256"] = file_hash

            all_docs.extend(docs)
            to_process += 1
            print(f"    ✅ 로드 완료 (pages={len(docs)})")
        except Exception as e:
            print(f"[{idx}/{len(paths)}] ❗ 로드 실패: {p} -> {e}")
            return {"error": f"파일 로드 실패: {p} -> {e}"}

    print(f"📚 총 페이지 문서 수: {len(all_docs)}")
    print(f"⏭️ 스킵: {skipped}개, ⏳ 신규/변경: {to_process}개")
    print("📥 [1/4] PDF 로드 단계 완료")
    return {"documents": all_docs, "splits": []}

def split_documents_node(state: GraphState) -> GraphState:
    """
    로드된 문서를 청크로 분할하여 상태에 저장하는 노드.
    """
    print("✂️ [2/4] 문서 분할 단계 시작")
    documents = state.get("documents", [])
    if not documents:
        print("⏭️ 분할할 문서가 없습니다.")
        return {"splits": []}
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    splits = splitter.split_documents(documents)
    print(f"✂️ 청크 수: {len(splits)}")
    print("✂️ [2/4] 문서 분할 단계 완료")
    return {"splits": splits}

def build_or_update_faiss_node(state: GraphState) -> GraphState:
    """
    분할된 청크를 임베딩하고 FAISS 인덱스를 생성/갱신하는 노드.
    """
    print("🧮 [3/4] 임베딩 & 인덱스 생성 단계 시작")
    splits = state.get("splits", [])
    if not splits:
        print("⏭️ 임베딩할 청크가 없습니다.")
        return {}

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    
    if os.path.exists(VECTOR_DB_PATH):
        print(f"📦 기존 인덱스 로드: {VECTOR_DB_PATH}")
        vs = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("🧱 새 인덱스 생성 중…")
        vs = None

    file_map = {}
    for s in splits:
        fname = s.metadata.get("source", "unknown.pdf")
        file_map.setdefault(fname, []).append(s)

    file_list = list(file_map.items())
    total_files = len(file_list)

    for idx, (fname, file_splits) in enumerate(file_list, start=1):
        chunks = len(file_splits)
        pages = len({sp.metadata.get("page") for sp in file_splits if "page" in sp.metadata})

        if vs:
            vs.add_documents(file_splits)
        else:
            vs = FAISS.from_documents(file_splits, embeddings)

        print(f"[{idx}/{total_files}] ✅ {os.path.basename(fname)} 임베딩 완료 "
              f"(pages={pages if pages else 'N/A'}, chunks={chunks})")

    if vs:
        vs.save_local(VECTOR_DB_PATH)
    print(f"💾 최종 인덱스 저장 완료: {VECTOR_DB_PATH}")

    manifest = _load_manifest(VECTOR_DB_PATH)
    for fname, file_splits in file_list:
        sha = next((sp.metadata.get("file_sha256") for sp in file_splits if "file_sha256" in sp.metadata), None)
        if sha:
            manifest[_abs(fname)] = {"sha256": sha}
    _save_manifest(VECTOR_DB_PATH, manifest)

    total_pages = sum(len({sp.metadata.get("page") for sp in fs if "page" in sp.metadata})
                      for _, fs in file_list)
    total_chunks = sum(len(fs) for _, fs in file_list)
    print(f"📊 전체 결과: 파일={total_files}, 페이지={total_pages}, 청크={total_chunks}")
    print("🧮 [3/4] 임베딩 & 인덱스 생성 단계 완료")
    return {}

# === 그래프 빌드 함수 ===
def build_graph():
    """
    LangGraph를 사용하여 인덱스 빌드 워크플로우를 정의하고 반환합니다.
    """
    graph = StateGraph(GraphState)
    graph.add_node("load_all_pdfs", load_all_pdfs_node)
    graph.add_node("split_documents", split_documents_node)
    graph.add_node("build_or_update_faiss", build_or_update_faiss_node)

    graph.add_edge("load_all_pdfs", "split_documents")
    graph.add_edge("split_documents", "build_or_update_faiss")
    graph.add_edge("build_or_update_faiss", END)

    graph.set_entry_point("load_all_pdfs")
    return graph.compile()

# === 메인 실행 로직 ===
if __name__ == "__main__":
    app = build_graph()
    
    # ── 그래프 시각화 ───────────────────────────────────────────
    try:
        from langgraph.graph import MermaidDrawMethod
        graph_image_path = Path(".") / "index_build_graph.png"
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
            mmd_path = Path(".") / "index_build_graph.mmd"
            with open(mmd_path, "w", encoding="utf-8") as f:
                f.write(mermaid_src)
            print(f"📝 Mermaid 소스를 '{mmd_path}'로 저장했습니다. (mermaid.live 등에서 렌더 가능)")
        except Exception as e2:
            print(f"추가 백업도 실패: {e2}")
    # ───────────────────────────────────────────────────────────

    initial_state = {"documents": [], "splits": [], "error": None}
    print("\n🚀 인덱스 빌드 프로세스 시작")
    final_state = app.invoke(initial_state)

    if final_state.get("error"):
        print(f"❗ 그래프 실행 중 오류 발생: {final_state['error']}")
    else:
        print(f"🎉 [4/4] 전체 인덱스 빌드 완료 (저장 경로: {VECTOR_DB_PATH})")
