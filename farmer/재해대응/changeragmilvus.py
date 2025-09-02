# -*- coding: utf-8 -*-
"""
하이브리드 인덱서 (PDF 텍스트 + (옵션)OCR + (옵션)PDF 표 파싱) → LangChain Milvus 저장

핵심
- LangChain Milvus 벡터스토어 사용(스키마 자동)
- PDF 원본 메타 유입 차단(화이트리스트 키만 유지)
- 메타데이터 스칼라화 + 기본값 채움 → 모든 배치에서 필드 수/타입 일관
- 스키마 충돌 시 자동 드롭 후 재생성 (또는 .env의 RESET_COLLECTION=1 사용)

필요 패키지
  pip install langchain-community langchain-text-splitters langchain-huggingface
  pip install pypdf pymupdf easyocr pdfplumber python-dotenv pillow numpy sentence-transformers pymilvus
"""

import os
import re
import io
import json
from glob import glob
from typing import List, Any, Dict, Tuple, Optional, TypedDict
from itertools import chain
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


from dotenv import load_dotenv
load_dotenv()

# ===== (신규) 지식 추출 (NER) =====
import spacy
_NER_MODEL = None

def get_ner_model():
    """spaCy 모델을 로드합니다. 모델이 없으면 자동으로 다운로드합니다."""
    global _NER_MODEL
    if _NER_MODEL is None:
        try:
            _NER_MODEL = spacy.load("ko_core_news_sm")
        except OSError:
            print("\n⏳ spaCy 한국어 모델(ko_core_news_sm)을 찾을 수 없어 새로 다운로드합니다...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "ko_core_news_sm"], check=True, capture_output=True)
            _NER_MODEL = spacy.load("ko_core_news_sm")
            print("✅ spaCy 모델 다운로드 완료.")
    return _NER_MODEL

# ===== 설정 =====
PDF_DIR = os.getenv("PDF_DIR", "./pdfs")
IMAGE_DIR = os.getenv("IMAGE_DIR", "").strip()

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

USE_OCR = os.getenv("USE_OCR", "1") == "1"
OCR_LANGS = [s.strip() for s in os.getenv("OCR_LANGS", "ko,en").split(",") if s.strip()]
OCR_CONF = float(os.getenv("OCR_CONF", "0.7"))
OCR_TRIGGER_LEN = int(os.getenv("OCR_TRIGGER_LEN", "80"))
FORCE_OCR = os.getenv("FORCE_OCR", "0") == "1"

PDF_PARSE_TABLES = os.getenv("PDF_PARSE_TABLES", "1") == "1"

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "agri_disaster_docs")
RESET_COLLECTION = os.getenv("RESET_COLLECTION", "0") == "1"

# 허용할 메타데이터 키(이외는 드랍) - 간소화
ALLOWED_META_KEYS = {
    "source", "file_name", "page", "type",
    "year", "month", "ocr_applied", "regions",
    # [업그레이드] 지식 추출 필드 추가
    "crops", "disaster_types",
}

# 모든 문서에 동일한 메타 키 집합 보장을 위한 기본값 - 간소화
_META_DEFAULTS = {
    "source": "",
    "file_name": "",
    "page": 0,
    "type": "",
    "year": 0,
    "month": 0,
    "ocr_applied": False,
    "regions": "[]", # JSON
    # [업그레이드] 지식 추출 필드 기본값 추가
    "crops": "[]", # JSON
    "disaster_types": "[]", # JSON
}

# ===== LangChain / VectorStore =====
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ===== PDF / OCR =====
from langchain_community.document_loaders import PyPDFLoader
import fitz  # PyMuPDF
import easyocr
from PIL import Image
import numpy as np

# ===== pdfplumber (표 파싱) =====
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

# =====(선택) 컬렉션 드롭/체크 & 예외 타입: pymilvus 사용 =====
try:
    from pymilvus import (
        connections, utility, FieldSchema, CollectionSchema, DataType, Collection
    )
    from pymilvus.exceptions import DataNotMatchException, ParamError
    _HAS_PYMILVUS = True
except Exception:
    DataNotMatchException = tuple([Exception])
    ParamError = tuple([Exception])
    _HAS_PYMILVUS = False
    Collection = None # To avoid NameErrors

# ===== (전역) Langchain / Pymilvus / LLM 모델 =====
# 모델 객체는 메모리에 한 번만 로드되도록 전역으로 관리
embed_model = None
ocr_reader = None


# ===== Milvus 연결 관리 =====
_MILVUS_CONN_ALIAS = "default"

def _ensure_pymilvus_conn(alias: str = _MILVUS_CONN_ALIAS):
    """Milvus 연결을 보장합니다. 기존 연결 상태와 무관하게 항상 새로 연결을 시도합니다."""
    connections.connect(alias, host=MILVUS_HOST, port=MILVUS_PORT)
    

def _drop_collection(name: str, alias: str = _MILVUS_CONN_ALIAS):
    """Milvus 컬렉션이 존재하면 삭제합니다."""
    try:
        _ensure_pymilvus_conn(alias)
        if utility.has_collection(name, using=alias):
            print(f"🗑️ 기존 '{name}' 컬렉션이 있어 삭제합니다.")
            utility.drop_collection(name, using=alias)
    except Exception as e:
        print(f"⚠️ Milvus 컬렉션 확인/삭제 중 오류 발생. Milvus 서버 상태를 확인하세요. (오류: {e})")

def _create_milvus_collection_if_not_exists(collection_name: str, embedding_dim: int):
    """지정된 스키마로 Milvus 컬렉션을 생성합니다 (없는 경우)."""
    if not utility.has_collection(collection_name):
        print(f"Milvus 컬렉션 '{collection_name}'이(가) 없어 새로 생성합니다.")
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="regions", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, "재해 대응 데이터")
        collection = Collection(name=collection_name, schema=schema)
        
        index_params = {
            "metric_type": "IP",
            "index_type": "AUTOINDEX",
            "params": {}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        print(f"'{collection_name}' 컬렉션 및 벡터 인덱스 생성 완료.")
    else:
        print(f"Milvus 컬렉션 '{collection_name}'이(가) 이미 존재합니다.")


# ===== 메타데이터 스칼라화/정규화 =====
# ===== (신규) 메타데이터 타입 정의 =====
class MilvusMetadata(TypedDict):
    """Milvus에 저장될 메타데이터의 구조를 정의합니다."""
    file_name: str
    page: int
    type: str
    regions: List[str]


def _sanitize_metadata_for_milvus(metadata: Dict[str, Any]) -> MilvusMetadata:
    """
    Milvus에 적재하기 전에 메타데이터를 정리하고 MilvusMetadata 타입으로 변환합니다.
    - Milvus 스키마 필드에 맞는 키만 유지합니다.
    - 'regions'는 항상 리스트여야 합니다.
    """
    schema_keys = {"file_name", "page", "type", "regions"}
    sanitized: Dict[str, Any] = {}

    for key, value in metadata.items():
        if key not in schema_keys:
            continue

        if key == "regions":
            regions_list = value if isinstance(value, list) else [value] if value else []
            sanitized[key] = regions_list
        elif value is not None:
            sanitized[key] = value

    # 스키마에 정의된 모든 키가 존재하는지 확인하고, 없으면 기본값 설정
    if "file_name" not in sanitized: sanitized["file_name"] = "unknown"
    if "page" not in sanitized: sanitized["page"] = -1
    if "type" not in sanitized: sanitized["type"] = "text"
    if "regions" not in sanitized: sanitized["regions"] = []

    return sanitized  # type: ignore


def _get_embedding(
    texts: List[str],
    device: str = "cuda",
) -> np.ndarray:
    """주어진 텍스트 목록에 대한 임베딩을 계산합니다."""
    # embed_model은 전역으로 관리되거나 필요 시 로드
    global embed_model
    if embed_model is None:
        print(f"HuggingFace 임베딩 모델 로드: {EMBED_MODEL_NAME}")
        embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    
    # 모델의 device를 명시적으로 설정
    if hasattr(embed_model, '_client') and embed_model._client is not None:
        embed_model._client.to(device)
    
    print(f"임베딩 계산 중 ({len(texts)}개 텍스트, 장치: {device})...")
    return np.array(embed_model.embed_documents(texts), dtype="float32")


def upsert_documents_to_milvus(
    docs: List[Document],
    collection_name: str,
    batch_size: int = 100,
) -> Milvus:
    """
    Milvus에 문서를 업서트합니다. 기존 컬렉션은 삭제하고 새로 생성합니다.
    (LangChain Milvus wrapper 대신 pymilvus를 직접 사용하여 JSON 스키마를 적용)
    """
    _ensure_pymilvus_conn()
    texts = [doc.page_content for doc in docs]
    
    # 임베딩 차원 확인 및 컬렉션 생성
    temp_embeddings = _get_embedding(texts[:1], "cpu")
    embedding_dim = temp_embeddings.shape[1]
    _create_milvus_collection_if_not_exists(collection_name, embedding_dim)

    print(f"총 {len(docs)}개 문서를 Milvus에 업로드합니다...")

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        
        print(f"  - 배치 {i//batch_size + 1}: 문서 {i+1} ~ {i+len(batch_docs)} 처리 중...")
        
        embeddings = _get_embedding(batch_texts, "cuda")
        metadata: List[MilvusMetadata] = [_sanitize_metadata_for_milvus(doc.metadata) for doc in batch_docs]

        # LangChain Milvus.add_texts 대신 pymilvus 직접 사용
        collection = Collection(name=collection_name)
        entities = [
            {
                "text": text,
                "vector": emb,
                "file_name": meta["file_name"],
                "page": meta["page"],
                "type": meta["type"],
                "regions": meta["regions"],
            }
            for text, emb, meta in zip(batch_texts, embeddings, metadata)
        ]
        
        try:
            collection.insert(entities)
            print(f"    > 성공: {len(entities)}개 문서 삽입 완료.")
        except Exception as e:
            print(f"    > ⚠️ 오류: 배치 삽입 실패. {e}")

    # 데이터가 완전히 삽입되고 인덱싱되도록 flush 수행
    collection = Collection(name=collection_name)
    print("모든 문서 삽입 후 Milvus 데이터 flush를 수행합니다...")
    collection.flush()
    print(f"'{collection_name}' 컬렉션의 총 문서 수: {collection.num_entities}")


# ===== PDF/OCR 처리 및 텍스트 분할 =====

def clean_text(text: str) -> str:
    """텍스트에서 불필요한 문자 및 공백을 제거합니다."""
    text = re.sub(r"[^\w\s\.\,\(\)\/%\:\-\~가-힣]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_splitter() -> RecursiveCharacterTextSplitter:
    """LangChain의 텍스트 분할기를 생성합니다."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

REGION_CANON = {
    "강원":"강원도","강원도":"강원도", "경기":"경기도","경기도":"경기도", "충북":"충청북도","충청북도":"충청북도",
    "충남":"충청남도","충청남도":"충청남도", "전북":"전라북도","전라북도":"전라북도", "전남":"전라남도","전라남도":"전라남도",
    "경북":"경상북도","경상북도":"경상북도", "경남":"경상남도","경상남도":"경상남도", "제주":"제주특별자치도","제주특별자치도":"제주특별자치도",
    "세종":"세종","세종특별자치시":"세종", "인천":"인천광역시","서울":"서울특별시","부산":"부산광역시","대구":"대구광역시",
    "광주":"광주광역시","대전":"대전광역시","울산":"울산광역시"
}
REGION_KEYS = sorted(set(REGION_CANON.keys()), key=len, reverse=True)

def canon_region(tok: str) -> str:
    t = (tok or "").strip().replace(" ", "")
    if t in REGION_CANON: return REGION_CANON[t]
    if t.endswith("도"): return t
    if t in ("강원","경기","충북","충남","전북","전남","경북","경남","제주"): return t+"도"
    return t

def detect_regions(text: str) -> List[str]:
    t = (text or "").replace(" ", "")
    hits = []
    for k in REGION_KEYS:
        if k in t: hits.append(canon_region(k))
    return sorted(list(set(hits)))

def _get_ocr_reader() -> easyocr.Reader:
    """EasyOCR Reader 객체를 지연 로딩하여 반환합니다."""
    global ocr_reader
    if ocr_reader is None:
        print("EasyOCR Reader 로드 중...")
        ocr_reader = easyocr.Reader(OCR_LANGS, gpu=True)
    return ocr_reader

def process_single_pdf(file_path: str) -> List[Document]:
    """단일 PDF 파일을 처리하여 Document 청크 리스트를 반환합니다."""
    docs = []
    try:
        doc_fitz = fitz.open(file_path)
        for page_num, page in enumerate(doc_fitz, start=1):
            text = page.get_text()
            ocr_applied = False
            if USE_OCR and (FORCE_OCR or len(text) < OCR_TRIGGER_LEN):
                try:
                    reader = _get_ocr_reader()
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    ocr_results = reader.readtext(img_bytes)
                    ocr_text = " ".join([res[1] for res in ocr_results])
                    text = text + " " + ocr_text
                    ocr_applied = True
                except Exception as e:
                    print(f"OCR 실패: {file_path} p.{page_num} - {e}")

            cleaned_text = clean_text(text)
            if not cleaned_text:
                continue

            splitter = make_splitter()
            chunks = splitter.split_text(cleaned_text)
            page_regions = detect_regions(cleaned_text)

            for chunk in chunks:
                metadata = {
                    "file_name": os.path.basename(file_path),
                    "page": page_num,
                    "type": "pdf_text",
                    "regions": page_regions
                }
                docs.append(Document(page_content=chunk, metadata=metadata))
    except Exception as e:
        print(f"PDF 처리 실패 {file_path}: {e}")
    return docs

def process_single_image(file_path: str) -> List[Document]:
    """단일 이미지 파일을 처리하여 Document 청크 리스트를 반환합니다."""
    docs = []
    if not USE_OCR:
        return docs
    try:
        reader = _get_ocr_reader()
        ocr_results = reader.readtext(file_path)
        text = " ".join([res[1] for res in ocr_results])
        cleaned_text = clean_text(text)

        if not cleaned_text:
            return []

        splitter = make_splitter()
        chunks = splitter.split_text(cleaned_text)
        image_regions = detect_regions(cleaned_text)

        for chunk in chunks:
            metadata = {
                "file_name": os.path.basename(file_path),
                "page": 1,
                "type": "image_ocr",
                "regions": image_regions
            }
            docs.append(Document(page_content=chunk, metadata=metadata))
    except Exception as e:
        print(f"이미지 처리 실패 {file_path}: {e}")
    return docs

# ===== (옵션) 간단 검색 =====
def _json_or_list(val):
    """문자열로 저장된 JSON(list) → list 로 복원. 실패 시 빈 리스트."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            obj = json.loads(val)
            if isinstance(obj, list):
                return obj
        except Exception:
            return []
    return []

def vector_search(query: str, k=5, meta_filters: Dict[str, Any] | None = None):
    """
    서버측 expr 대신 결과 수집 후 클라이언트 사이드 메타 필터링
    (regions/metrics는 JSON 문자열 파싱)
    """
    embed = _get_embedding()
    vs = Milvus(
        embedding_function=embed,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )
    results = vs.similarity_search_with_score(query, k=k*3)  # 넉넉히 받아 필터
    filt = []
    for doc, score in results:
        md = doc.metadata or {}
        ok = True
        if meta_filters:
            if "year" in meta_filters and md.get("year") != meta_filters["year"]:
                ok = False
            if "region" in meta_filters:
                rg = meta_filters["region"]
                if md.get("region") and md.get("region") != rg:
                    ok = False
                regions = _json_or_list(md.get("regions", "[]"))
                if regions and rg not in regions:
                    ok = False
            if "metric" in meta_filters:
                mts = _json_or_list(md.get("metrics", "[]"))
                if mts and meta_filters["metric"] not in mts:
                    ok = False
        if ok:
            filt.append((doc, score))
        if len(filt) >= k:
            break
    return filt

# ===== 메인 =====
if __name__ == "__main__":
    print("🚀 하이브리드 인덱스 → Milvus 업서트 시작")
    
    # 스크립트 시작 시점에 항상 컬렉션을 강제로 삭제하여 이전 실행의 데이터를 남기지 않음
    _drop_collection(COLLECTION_NAME)

    if PDF_PARSE_TABLES and not _HAS_PDFPLUMBER:
        print("⚠️ PDF_PARSE_TABLES=1 이지만 pdfplumber 미설치 → 표 파싱 생략합니다. (pip install pdfplumber)")

    pdf_paths = sorted(glob(os.path.join(PDF_DIR, "*.pdf")))
    image_paths = []
    if IMAGE_DIR:
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            image_paths.extend(glob(os.path.join(IMAGE_DIR, ext)))
        image_paths = sorted(image_paths)
    
    source_filenames = [os.path.basename(p) for p in pdf_paths + image_paths]

    # 병렬 처리 대신 순차적으로 처리
    all_docs = []
    if pdf_paths:
        print(f"📄 PDF {len(pdf_paths)}개 순차 처리 시작")
        for p_path in tqdm(pdf_paths, desc="PDF 처리"):
            all_docs.extend(process_single_pdf(p_path))

    if image_paths:
        print(f"🖼️ 이미지 {len(image_paths)}개 순차 처리 시작")
        for i_path in tqdm(image_paths, desc="이미지 처리"):
            all_docs.extend(process_single_image(i_path))
            
    print(f"📚 전체 수집 청크 수: {len(all_docs)}")

    # 3) Milvus 업서트 (컬렉션 생성/추가)
    print(f"🧾 업서트 대상 문서 수: {len(all_docs)}")
    if not all_docs:
        print("ℹ️ 처리할 문서가 없어 종료합니다.")
    else:
        upsert_documents_to_milvus(all_docs, COLLECTION_NAME)

    # (옵션) 간단 검색 테스트
    # samples = vector_search("강원도 8월 농작물 피해", k=3, meta_filters={"year": 2024, "region": "강원도"})
    # for doc, score in samples:
    #     print("🔎", round(score, 4), doc.page_content[:120], doc.metadata)

    print("🎉 작업 완료")
