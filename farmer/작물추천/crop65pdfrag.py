import os
from glob import glob
from typing import List, Any

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

# === 신규/변경 파일만 처리하기 위한 유틸 ===
import json
import hashlib
from pathlib import Path

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

# === 단계별 함수 ===
def load_all_pdfs(pdf_dir: str) -> List[Any]:
    print("📥 [1/4] PDF 로드 단계 시작")
    paths = sorted(glob(os.path.join(pdf_dir, "*.pdf")))
    if not paths:
        raise FileNotFoundError(f"PDF가 없습니다: {pdf_dir}")
    print(f"📂 총 파일 수: {len(paths)}")

    manifest = _load_manifest(VECTOR_DB_PATH)  # { abs_path: {"sha256": "..."} }
    all_docs = []
    skipped, to_process = 0, 0

    for idx, p in enumerate(paths, start=1):
        abs_path = _abs(p)
        try:
            file_hash = _sha256_file(abs_path)

            # 변경 없으면 스킵
            if manifest.get(abs_path, {}).get("sha256") == file_hash:
                skipped += 1
                print(f"[{idx}/{len(paths)}] ⏭️ 변경 없음: {os.path.basename(p)} (skip)")
                continue

            print(f"[{idx}/{len(paths)}] 📥 파일 로드 중: {os.path.basename(p)}")
            docs = PyPDFLoader(abs_path).load()

            # 이후 단계에서 사용할 메타데이터 주입
            for d in docs:
                d.metadata["source"] = abs_path       # 절대 경로
                d.metadata["file_sha256"] = file_hash # 파일 해시

            all_docs.extend(docs)
            to_process += 1
            print(f"    ✅ 로드 완료 (pages={len(docs)})")
        except Exception as e:
            print(f"[{idx}/{len(paths)}] ❗ 로드 실패: {p} -> {e}")

    print(f"📚 총 페이지 문서 수: {len(all_docs)}")
    print(f"⏭️ 스킵: {skipped}개, ⏳ 신규/변경: {to_process}개")
    print("📥 [1/4] PDF 로드 단계 완료")
    return all_docs


def split_documents(documents: List[Any]) -> List[Any]:
    print("✂️ [2/4] 문서 분할 단계 시작")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    splits = splitter.split_documents(documents)
    print(f"✂️ 청크 수: {len(splits)}")
    print("✂️ [2/4] 문서 분할 단계 완료")
    return splits


def build_or_update_faiss(splits: List[Any], db_path: str) -> None:
    print("🧮 [3/4] 임베딩 & 인덱스 생성 단계 시작")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # 인덱스 로드/생성
    if os.path.exists(db_path):
        print(f"📦 기존 인덱스 로드: {db_path}")
        vs = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("🧱 새 인덱스 생성 중…")
        vs = None

    # === splits를 파일별로 묶기 (source 기준) ===
    file_map = {}
    for s in splits:
        fname = s.metadata.get("source", "unknown.pdf")
        file_map.setdefault(fname, []).append(s)

    file_list = list(file_map.items())
    total_files = len(file_list)

    # 파일별 임베딩
    for idx, (fname, file_splits) in enumerate(file_list, start=1):
        chunks = len(file_splits)
        # 페이지 수 추정: split 메타데이터의 page 값을 unique count
        pages = len({sp.metadata.get("page") for sp in file_splits if "page" in sp.metadata})

        if vs:
            vs.add_documents(file_splits)
        else:
            vs = FAISS.from_documents(file_splits, embeddings)

        # ✅ 파일별 임베딩 완료 로그
        print(f"[{idx}/{total_files}] ✅ {os.path.basename(fname)} 임베딩 완료 "
              f"(pages={pages if pages else 'N/A'}, chunks={chunks})")

    # 저장
    if vs:
        vs.save_local(db_path)
    print(f"💾 최종 인덱스 저장 완료: {db_path}")

    # === 매니페스트 갱신: 이번에 처리된 파일의 sha256 기록 ===
    manifest = _load_manifest(db_path)
    for fname, file_splits in file_list:
        sha = None
        for sp in file_splits:
            sha = sp.metadata.get("file_sha256")
            if sha:
                break
        if sha:
            manifest[_abs(fname)] = {"sha256": sha}
    _save_manifest(db_path, manifest)

    # 총합 로그
    total_pages = sum(len({sp.metadata.get("page") for sp in fs if "page" in sp.metadata})
                      for _, fs in file_list)
    total_chunks = sum(len(fs) for _, fs in file_list)
    print(f"📊 전체 결과: 파일={total_files}, 페이지={total_pages}, 청크={total_chunks}")
    print("🧮 [3/4] 임베딩 & 인덱스 생성 단계 완료")
    

if __name__ == "__main__":
    print("🚀 인덱스 빌드 프로세스 시작")
    docs = load_all_pdfs(PDF_DIR)             # ✅ 신규/변경 파일만 반환 (스킵 로그 포함)
    splits = split_documents(docs)             # ✂️ [2/4] 로그 그대로 출력
    build_or_update_faiss(splits, VECTOR_DB_PATH)  # 파일별 임베딩 완료 로그 + manifest 갱신
    print(f"🎉 [4/4] 전체 인덱스 빌드 완료 (저장 경로: {VECTOR_DB_PATH})")
