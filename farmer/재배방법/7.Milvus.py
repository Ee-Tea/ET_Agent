import os
import pandas as pd
from glob import glob
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import utility, connections, FieldSchema, CollectionSchema, DataType
from langchain_core.documents import Document

load_dotenv()

# === 설정 ===
DATA_DIR = "./data/cropinfo" 
COLLECTION_NAME = "crop_info"
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def load_documents() -> List[Document]:
    """
    지정된 디렉터리에서 모든 PDF 파일을 Document 객체로 로드합니다.
    """
    all_docs = []
    
    # 1. PDF 파일 로드
    pdf_paths = sorted(glob(os.path.join(DATA_DIR, "*.pdf")))
    for p in pdf_paths:
        try:
            docs = PyPDFLoader(p).load()
            # ❗ 수정된 부분: 경로 구분자를 '/'로 통일
            normalized_path = p.replace(os.sep, '/')
            for doc in docs:
                doc.metadata['source'] = normalized_path
            print(f"✅ PDF 파일 로드: {os.path.basename(p)} (총 페이지 수: {len(docs)})")
            all_docs.extend(docs)
        except Exception as e:
            print(f"❗ PDF 파일 로드 실패: {p} -> 오류: {e}")

    if not all_docs:
        raise FileNotFoundError(f"'{DATA_DIR}' 디렉터리에서 PDF 파일을 찾을 수 없습니다.")
        
    print(f"📚 총 로드된 문서 수: {len(all_docs)}")
    return all_docs

def split_documents(documents: List[Document]) -> List[Document]:
    """
    문서를 작은 청크로 분할합니다.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", ". ", "? ", "! ", "\n", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️ 총 생성된 청크 수: {len(chunks)}")
    return chunks

def build_vectorstore(documents: List[Document]) -> None:
    """
    Milvus에 벡터스토어를 구축합니다.
    """
    print("🧠 벡터스토어 준비 중...")

    # 🔄 기존 연결 확인 및 끊기 (중복 연결 방지)
    if "default" in connections.list_connections():
        print("🔄 기존 연결을 끊고 새 연결을 시도합니다.")
        connections.disconnect("default")

    # Milvus 연결 설정
    connections.connect(alias="default", host="localhost", port="19530")
    
    # 임베딩 모델 로드
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )
    
    # ✅ 기존 컬렉션 확인 및 데이터 삽입 로직
    if utility.has_collection(COLLECTION_NAME):
        print(f"🔄 '{COLLECTION_NAME}' 컬렉션이 이미 존재합니다. 데이터 삽입을 건너뜁니다.")
    else:
        print(f"🆕 새로운 컬렉션 '{COLLECTION_NAME}'을 생성하고 데이터를 삽입합니다...")
        
        # Milvus 벡터스토어 생성
        Milvus.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=COLLECTION_NAME,
            connection_args={"host": "localhost", "port": "19530"}
        )
    
    print("✅ 벡터스토어 구축이 완료되었습니다.")

if __name__ == "__main__":
    try:
        docs = load_documents()
        chunks = split_documents(docs)
        build_vectorstore(chunks)
        print("🎉 모든 문서가 처리되어 Milvus에 인덱싱되었습니다.")
    except Exception as e:
        print(f"❗ 처리 중 오류 발생: {e}")