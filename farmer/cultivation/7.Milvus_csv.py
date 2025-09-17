import os
import pandas as pd
from glob import glob
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import utility, connections
from langchain_core.documents import Document

load_dotenv()

# === 설정 ===
DATA_DIR = "./data/redis/cropinfo" 
COLLECTION_NAME = "crop_grow" # 새로운 컬렉션 이름
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def load_documents() -> List[Document]:
    """
    지정된 디렉터리에서 모든 CSV 파일을 Document 객체로 로드합니다.
    """
    all_docs = []
    
    # 1. CSV 파일 로드
    csv_paths = sorted(glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"'{DATA_DIR}' 디렉터리에서 CSV 파일을 찾을 수 없습니다.")

    for p in csv_paths:
        try:
            # CSVLoader를 사용하여 파일 로드
            # source_column을 지정하면 메타데이터에 소스 파일 경로가 저장됩니다.
            loader = CSVLoader(file_path=p, encoding='utf-8')
            docs = loader.load()
            
            # 각 문서의 메타데이터에 소스 파일 경로를 추가합니다.
            normalized_path = os.path.normpath(p).replace(os.sep, '/')
            for doc in docs:
                doc.metadata['source'] = normalized_path

            print(f"✅ CSV 파일 로드: {os.path.basename(p)} (총 문서 수: {len(docs)})")
            all_docs.extend(docs)
        except Exception as e:
            print(f"❗ CSV 파일 로드 실패: {p} -> 오류: {e}")

    if not all_docs:
        raise FileNotFoundError(f"'{DATA_DIR}' 디렉터리에서 유효한 데이터를 가진 CSV 파일을 찾을 수 없습니다.")
        
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

    # 기존 연결 확인 및 끊기 (중복 연결 방지)
    if "default" in connections.list_connections():
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
        print(f"🔄 '{COLLECTION_NAME}' 컬렉션이 이미 존재합니다. 데이터를 삽입합니다.")
        # 기존 컬렉션에 추가하는 로직
        vectorstore = Milvus(
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
            connection_args={"host": "localhost", "port": "19530"}
        )
        vectorstore.add_documents(documents=documents)
        print(f"✅ '{COLLECTION_NAME}' 컬렉션에 새 데이터가 추가되었습니다.")
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