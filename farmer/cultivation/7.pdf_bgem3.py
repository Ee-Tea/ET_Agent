import os
from glob import glob
from typing import List, Any
from dotenv import load_dotenv

# --- 1. 환경 변수 로드 및 설정 ---
load_dotenv()

PDF_CSV_DIR = os.getenv("PDF_CSV_DIR", "./data/cropinfo")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "faiss_db_bge_m3")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- 2. LangChain 라이브러리 임포트 ---
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 3. 핵심 함수 정의 ---

def load_all_documents(data_dir: str) -> List[Any]:
    """
    지정된 디렉토리에서 모든 PDF 및 CSV 파일을 로드합니다.
    """
    print("--- 📚 문서 로딩 시작 ---")
    all_docs = []
    
    # PDF 파일 로드
    pdf_paths = sorted(glob(os.path.join(data_dir, "*.pdf")))
    for p in pdf_paths:
        try:
            loader = PyPDFLoader(p)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"✅ PDF 로드: {os.path.basename(p)} (pages={len(docs)})")
        except Exception as e:
            print(f"❗ PDF 로드 실패: {p} -> {e}")

    # CSV 파일 로드
    csv_paths = sorted(glob(os.path.join(data_dir, "*.csv")))
    for p in csv_paths:
        try:
            loader = CSVLoader(p, encoding='utf-8')
            docs = loader.load()
            all_docs.extend(docs)
            print(f"✅ CSV 로드: {os.path.basename(p)} (rows={len(docs)})")
        except Exception as e:
            print(f"❗ CSV 로드 실패: {p} -> {e}")

    if not all_docs:
        raise FileNotFoundError(f"PDF 및 CSV 파일이 없습니다: {data_dir}")
        
    print(f"📚 총 문서 수: {len(all_docs)} 페이지/행")
    print("--- 문서 로딩 완료 ---")
    return all_docs

def split_documents(documents: List[Any]) -> List[Any]:
    """
    문서를 작은 덩어리(청크)로 분할합니다.
    """
    print("--- ✂️ 텍스트 분할 (청킹) 시작 ---")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    splits = splitter.split_documents(documents)
    print(f"✂️ 총 청크 수: {len(splits)}")
    print("--- 텍스트 분할 완료 ---")
    return splits

def build_or_update_faiss(splits: List[Any], db_path: str, model_name: str) -> None:
    """
    문서 청크를 사용하여 FAISS 벡터 데이터베이스를 생성하거나 업데이트합니다.
    """
    print(f"--- 🧱 임베딩 및 FAISS DB 생성/업데이트 시작 ({model_name}) ---")
    
    # HuggingFace 임베딩 모델 설정
    print(f"🔎 임베딩 모델 로딩: {model_name} (최초 실행 시 다운로드 필요)")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # GPU가 없다면 'cpu'를 사용하세요.
        encode_kwargs={'normalize_embeddings': True}
    )

    if os.path.exists(db_path):
        print(f"📦 기존 인덱스 로드: {db_path}")
        vs = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print("➕ 새 청크를 추가하여 인덱스 업데이트 중...")
        vs.add_documents(splits)
        vs.save_local(db_path)
    else:
        print("🧱 새 인덱스 생성 중...")
        vs = FAISS.from_documents(splits, embeddings)
        vs.save_local(db_path)
    
    print(f"✅ FAISS 벡터 DB가 '{db_path}' 경로에 성공적으로 저장되었습니다.")
    print("--- 임베딩 및 DB 생성/업데이트 완료 ---")

# --- 4. 메인 실행 부분 ---
if __name__ == "__main__":
    print("🚀 FAISS 인덱스 빌드 시작")
    try:
        docs = load_all_documents(PDF_CSV_DIR)
        splits = split_documents(docs)
        build_or_update_faiss(splits, VECTOR_DB_PATH, EMBED_MODEL_NAME)
        print("\n🎉 인덱스 빌드 완료 🎉")
    except FileNotFoundError as e:
        print(f"오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")