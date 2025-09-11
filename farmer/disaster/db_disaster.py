# -*- coding: utf-8 -*-
"""
하이브리드 인덱서 (PDF 텍스트 + OCR 항상 실행 + 표 파싱 항상 실행) → Milvus 저장
- 모든 청크에서 연도(years) 메타데이터 자동 추출 및 저장

필요 패키지:
  pip install langchain-community langchain-text-splitters langchain-huggingface
  pip install pypdf pymupdf easyocr pdfplumber python-dotenv pillow numpy sentence-transformers pymilvus
"""

import os
import re
from glob import glob
from typing import List, Any, Dict, Optional, TypedDict
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# ===== 설정 =====
PDF_DIR = os.getenv("PDF_DIR", "./pdfs")
IMAGE_DIR = os.getenv("IMAGE_DIR", "").strip()

# ✅ 임베딩 모델 변경
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "agri_disaster_docs")

# ===== LangChain / VectorStore =====
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ===== PDF / OCR =====
import fitz  # PyMuPDF
import easyocr
import numpy as np

# ===== pdfplumber (표 파싱) =====
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

# ===== pymilvus =====
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)

# ===== 전역 모델/리더 =====
_embedder: Optional[HuggingFaceEmbeddings] = None
ocr_reader = None

# ===== Milvus 연결 =====
_MILVUS_CONN_ALIAS = "default"

def _ensure_pymilvus_conn(alias: str = _MILVUS_CONN_ALIAS):
    connections.connect(alias, host=MILVUS_HOST, port=MILVUS_PORT)

def _drop_collection(name: str, alias: str = _MILVUS_CONN_ALIAS):
    try:
        _ensure_pymilvus_conn(alias)
        if utility.has_collection(name, using=alias):
            print(f"🗑️ 기존 '{name}' 컬렉션 삭제")
            utility.drop_collection(name, using=alias)
    except Exception as e:
        print(f"⚠️ 컬렉션 삭제 중 오류: {e}")

def _create_milvus_collection_if_not_exists(collection_name: str, embedding_dim: int):
    if not utility.has_collection(collection_name):
        print(f"📦 Milvus 컬렉션 '{collection_name}' 생성")
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="regions", dtype=DataType.JSON),
            FieldSchema(name="years", dtype=DataType.JSON),   # ✅ 연도 필드 추가
        ]
        schema = CollectionSchema(fields, "재해 대응 컬렉션")
        collection = Collection(name=collection_name, schema=schema)
        index_params = {"metric_type": "IP", "index_type": "AUTOINDEX", "params": {}}
        collection.create_index(field_name="vector", index_params=index_params)
        print(f"✅ '{collection_name}' 컬렉션 및 인덱스 생성 완료")
    else:
        print(f"ℹ️ Milvus 컬렉션 '{collection_name}' 이미 존재")

# ===== 메타데이터 스키마 =====
class MilvusMetadata(TypedDict):
    file_name: str
    page: int
    type: str
    regions: List[str]
    years: List[int]

def _sanitize_metadata_for_milvus(metadata: Dict[str, Any]) -> MilvusMetadata:
    schema_keys = {"file_name", "page", "type", "regions", "years"}
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in schema_keys:
            continue
        if key == "regions":
            regions_list = value if isinstance(value, list) else ([value] if value else [])
            sanitized[key] = regions_list
        elif key == "years":
            if isinstance(value, list):
                sanitized[key] = sorted({int(v) for v in value if str(v).isdigit()})
            else:
                sanitized[key] = []
        elif value is not None:
            sanitized[key] = value
    if "file_name" not in sanitized: sanitized["file_name"] = "unknown"
    if "page" not in sanitized: sanitized["page"] = -1
    if "type" not in sanitized: sanitized["type"] = "text"
    if "regions" not in sanitized: sanitized["regions"] = []
    if "years" not in sanitized: sanitized["years"] = []
    return sanitized  # type: ignore

# ===== 임베더 =====
def get_embedder(device: str = "cuda") -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        print(f"HuggingFace 임베딩 로드: {EMBED_MODEL_NAME} (device={device})")
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder

def _embed_texts(texts: List[str], device: str = "cuda") -> np.ndarray:
    emb = get_embedder(device=device)
    return np.array(emb.embed_documents(texts), dtype="float32")

# ===== 연도 추출 =====
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
def detect_years(text: str) -> List[int]:
    years = set()
    for m in _YEAR_RE.finditer(text or ""):
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            years.add(y)
    return sorted(years)

# ===== 지역 추출 =====
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

# ===== OCR Reader =====
def _get_ocr_reader() -> easyocr.Reader:
    global ocr_reader
    if ocr_reader is None:
        print("EasyOCR Reader 로드 중...")
        ocr_reader = easyocr.Reader(["ko", "en"], gpu=True)
    return ocr_reader

# ===== 업서트 =====
def upsert_documents_to_milvus(
    docs: List[Document],
    collection_name: str,
    batch_size: int = 100,
) -> None:
    _ensure_pymilvus_conn()
    texts = [doc.page_content for doc in docs]
    # 임베딩 차원 확인 (CPU로 한 번만)
    temp = _embed_texts(texts[:1], device="cpu")
    embedding_dim = temp.shape[1]
    _create_milvus_collection_if_not_exists(collection_name, embedding_dim)

    collection = Collection(name=collection_name)
    print(f"총 {len(docs)}개 문서를 Milvus에 업로드...")

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        print(f"  - 배치 {i//batch_size + 1}: {i+1} ~ {i+len(batch_docs)}")

        vectors = _embed_texts(batch_texts, device="cuda").tolist()
        file_names, pages, types, regions, years = [], [], [], [], []
        for d in batch_docs:
            md = _sanitize_metadata_for_milvus(d.metadata or {})
            file_names.append(md["file_name"])
            pages.append(int(md["page"]))
            types.append(md["type"])
            regions.append(md["regions"])
            years.append(md["years"])

        try:
            collection.insert([batch_texts, vectors, file_names, pages, types, regions, years])
            print(f"    > ✅ 삽입 성공: {len(batch_docs)}개")
        except Exception as e:
            print(f"    > ⚠️ 삽입 실패: {e}")

    print("⌛ flush 대기...")
    collection.flush()
    print(f"📊 num_entities: {collection.num_entities}")

# ===== 전처리 =====
def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s\.\,\(\)\/%\:\-\~가-힣]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )

# ===== PDF 처리 (텍스트 + OCR + 표) =====
def process_single_pdf(file_path: str) -> List[Document]:
    if not _HAS_PDFPLUMBER:
        raise RuntimeError("pdfplumber가 설치되어 있지 않습니다. `pip install pdfplumber` 필요")

    docs: List[Document] = []
    try:
        # --- 텍스트 + OCR ---
        doc_fitz = fitz.open(file_path)
        for page_num, page in enumerate(doc_fitz, start=1):
            text = page.get_text()
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
                ocr_applied = False

            cleaned_text = clean_text(text)
            if cleaned_text:
                splitter = make_splitter()
                chunks = splitter.split_text(cleaned_text)
                page_regions = detect_regions(cleaned_text)
                page_years = detect_years(cleaned_text)
                for chunk in chunks:
                    metadata = {
                        "file_name": os.path.basename(file_path),
                        "page": page_num,
                        "type": "pdf_text",
                        "regions": page_regions,
                        "years": page_years,
                        "ocr_applied": ocr_applied
                    }
                    docs.append(Document(page_content=chunk, metadata=metadata))

        # --- 표 추출 ---
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                    for table in tables:
                        table_str = "\n".join(
                            ["\t".join([(cell or "") for cell in row]) for row in table if row]
                        )
                        cleaned = clean_text(table_str)
                        if not cleaned:
                            continue
                        splitter = make_splitter()
                        chunks = splitter.split_text(cleaned)
                        page_regions = detect_regions(cleaned)
                        page_years = detect_years(cleaned)
                        for chunk in chunks:
                            metadata = {
                                "file_name": os.path.basename(file_path),
                                "page": page_num,
                                "type": "pdf_table",
                                "regions": page_regions,
                                "years": page_years
                            }
                            docs.append(Document(page_content=chunk, metadata=metadata))
                except Exception as e:
                    print(f"표 파싱 실패: {file_path} p.{page_num} - {e}")

    except Exception as e:
        print(f"PDF 처리 실패 {file_path}: {e}")
    return docs

# ===== 이미지 처리 (OCR) =====
def process_single_image(file_path: str) -> List[Document]:
    docs: List[Document] = []
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
        image_years = detect_years(cleaned_text)
        for chunk in chunks:
            metadata = {
                "file_name": os.path.basename(file_path),
                "page": 1,
                "type": "image_ocr",
                "regions": image_regions,
                "years": image_years
            }
            docs.append(Document(page_content=chunk, metadata=metadata))
    except Exception as e:
        print(f"이미지 처리 실패 {file_path}: {e}")
    return docs

# ===== 메인 =====
if __name__ == "__main__":
    print("🚀 하이브리드 인덱스 → Milvus 업서트 시작")

    if not _HAS_PDFPLUMBER:
        raise RuntimeError("pdfplumber 미설치 → 표 파싱 불가. `pip install pdfplumber` 필요")

    _drop_collection(COLLECTION_NAME)

    pdf_paths = sorted(glob(os.path.join(PDF_DIR, "*.pdf")))
    image_paths: List[str] = []
    if IMAGE_DIR:
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            image_paths.extend(glob(os.path.join(IMAGE_DIR, ext)))
        image_paths = sorted(image_paths)

    all_docs: List[Document] = []
    if pdf_paths:
        print(f"📄 PDF {len(pdf_paths)}개 처리 시작")
        for p_path in tqdm(pdf_paths, desc="PDF 처리"):
            all_docs.extend(process_single_pdf(p_path))

    if image_paths:
        print(f"🖼️ 이미지 {len(image_paths)}개 처리 시작")
        for i_path in tqdm(image_paths, desc="이미지 처리"):
            all_docs.extend(process_single_image(i_path))

    print(f"📚 전체 청크 수: {len(all_docs)}")

    if not all_docs:
        print("ℹ️ 처리할 문서 없음 → 종료")
    else:
        upsert_documents_to_milvus(all_docs, COLLECTION_NAME)
        print("🎉 작업 완료")
