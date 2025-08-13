# build_index.py
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

def load_all_pdfs(pdf_dir: str) -> List[Any]:
    paths = sorted(glob(os.path.join(pdf_dir, "*.pdf")))
    if not paths:
        raise FileNotFoundError(f"PDF가 없습니다: {pdf_dir}")
    all_docs = []
    for p in paths:
        try:
            docs = PyPDFLoader(p).load()  # page 단위 Document 리스트
            all_docs.extend(docs)
            print(f"✅ 로드: {os.path.basename(p)} (pages={len(docs)})")
        except Exception as e:
            print(f"❗ 로드 실패: {p} -> {e}")
    print(f"📚 총 페이지 문서 수: {len(all_docs)}")
    return all_docs

def split_documents(documents: List[Any]) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    splits = splitter.split_documents(documents)
    print(f"✂️ 청크 수: {len(splits)}")
    return splits

def build_or_update_faiss(splits: List[Any], db_path: str) -> None:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    if os.path.exists(db_path):
        print(f"📦 기존 인덱스 로드: {db_path}")
        vs = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print("➕ 새 청크 추가 후 인덱스 저장…")
        vs.add_documents(splits)
        vs.save_local(db_path)
    else:
        print("🧱 새 인덱스 생성…")
        vs = FAISS.from_documents(splits, embeddings)
        vs.save_local(db_path)
    print(f"✅ 저장 완료: {db_path}")

if __name__ == "__main__":
    print("🚀 인덱스 빌드 시작")
    docs = load_all_pdfs(PDF_DIR)
    splits = split_documents(docs)
    build_or_update_faiss(splits, VECTOR_DB_PATH)
    print("🎉 인덱스 빌드 완료")
