# sales_rag.py

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from dotenv import load_dotenv
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# 임베딩 모델
embedder = SentenceTransformer("BAAI/bge-m3")

# 환경 변수 로드
load_dotenv()
milvus_host = os.getenv("MILVUS_HOST", "localhost")
milvus_port = os.getenv("MILVUS_PORT", "19530")
collection_name = "market_price_docs"
collection = None

# CSV 파일 임베딩 및 Milvus에 저장 함수
def embed_and_store_csv(csv_path="sales/info_20240812.csv"):
    global collection
    df = pd.read_csv(csv_path, encoding="euc-kr")
    df['품목'] = df['품목'].fillna("정보 없음")
    docs = []
    for _, row in df.iterrows():
        doc = f"{row['판매장 이름']} ({row['주소']} / 주요 품목: {row['품목']})"
        docs.append(doc)
    
    if docs:
        embeddings = embedder.encode(docs)
        
        # 데이터 형태 검증
        if embeddings.shape[1] != 1024:
            raise ValueError(f"임베딩 차원이 1024가 아닙니다: {embeddings.shape[1]}")
        
        # 올바른 형태로 데이터 준비
        embedding_data = embeddings.tolist()  # 2D 리스트 형태 유지
        text_data = docs
        
        # Milvus에 데이터 삽입
        collection.insert([embedding_data, text_data], fields=["embedding", "text"])

# 실행
if __name__ == "__main__":
    print("Milvus 컬렉션을 강제로 재생성합니다...")
    connections.connect("default", host=milvus_host, port=milvus_port)

    # 1. 컬렉션이 존재하면 삭제
    if utility.has_collection(collection_name):
        print(f"기존 컬렉션 '{collection_name}'을 삭제합니다.")
        utility.drop_collection(collection_name)
        print("삭제 완료.")

    # 2. 새 컬렉션 생성
    print(f"'{collection_name}' 컬렉션을 새로 생성합니다.")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
    ]
    schema = CollectionSchema(fields, "시장 가격 문서 컬렉션")
    collection = Collection(collection_name, schema)
    print("컬렉션 생성 완료.")

    # 3. 데이터 삽입
    print("데이터를 임베딩하고 저장합니다...")
    embed_and_store_csv()
    print("데이터 삽입 완료.")

    # 4. 인덱스 생성
    print("인덱스를 생성합니다...")
    index_params = {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    print("인덱스 생성 완료.")
    
    connections.disconnect("default")
    print("작업이 완료되었습니다.")