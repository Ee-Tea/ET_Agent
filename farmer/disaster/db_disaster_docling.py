# -*- coding: utf-8 -*-
"""
하이브리드 인덱서 (Docling 텍스트 + OCR 항상 실행 + 표 파싱 항상 실행) → Milvus 저장
- Docling으로 텍스트 추출 (구조적 텍스트)
- OCR과 표 파싱은 db_disaster.py와 동일하게 유지
- 모든 청크에서 연도(years) 메타데이터 자동 추출 및 저장

필요 패키지:
  pip install langchain-community langchain-text-splitters langchain-huggingface
  pip install pypdf pymupdf easyocr pdfplumber python-dotenv pillow numpy sentence-transformers pymilvus
  pip install docling docling-core
"""

import os
import re
from glob import glob
from typing import List, Any, Dict, Optional, TypedDict
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# ===== 설정 =====
PDF_DIR = os.getenv("PDF_DIR", "./farmer/disaster/data")
IMAGE_DIR = os.getenv("IMAGE_DIR", "").strip()

# ✅ 임베딩 모델 변경
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "disaster_documents")

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

# ===== Docling (텍스트 파싱) =====
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
    import tiktoken
    _HAS_DOCLING = True
except Exception:
    _HAS_DOCLING = False

# ===== pymilvus =====
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)

# ===== 전역 모델/리더 =====
_embedder: Optional[HuggingFaceEmbeddings] = None
ocr_reader = None
docling_converter = None

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

# ===== Docling Converter =====
def _get_docling_converter() -> DocumentConverter:
    global docling_converter
    if docling_converter is None:
        if not _HAS_DOCLING:
            raise RuntimeError("Docling이 설치되어 있지 않습니다. `pip install docling docling-core` 필요")
        
        print("🔄 Docling 변환기 초기화 중... (텍스트만)")
        
        # 텍스트만 추출하도록 설정 (OCR과 표 파싱은 별도 처리)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # OCR 비활성화
        pipeline_options.do_table_structure = False  # 표 구조 비활성화
        
        # CPU만 사용
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=2, 
            device=AcceleratorDevice.CPU
        )
        
        docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options, 
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        
        print("✅ Docling 변환기 초기화 완료 (텍스트만)")
    return docling_converter

# ===== Docling 하이브리드 청킹 =====
def _docling_hybrid_chunking(text: str, file_name: str) -> List[str]:
    """Docling 하이브리드 청킹 사용"""
    if not _HAS_DOCLING:
        print("⚠️ Docling이 설치되지 않음, 기본 청킹 사용")
        splitter = make_splitter()
        return splitter.split_text(text)
    
    try:
        print("🧠 Docling Hybrid Chunking 시작...")
        
        # Docling의 공식 HybridChunker 사용
        tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model("gpt-4o-mini"),
            max_tokens=1200,  # 약 900자에 해당하는 토큰 수
        )
        
        chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,
        )
        
        # 임시 DoclingDocument 생성 (최신 버전 호환)
        from docling.datamodel.document import DoclingDocument
        doc = DoclingDocument(name=file_name)
        
        # 텍스트를 elements로 추가
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.document import TextElement
        
        # TextElement 생성 및 추가
        text_element = TextElement(
            text=text,
            label="text",
            page=1
        )
        doc.add_element(text_element)
        
        # 문서를 청크로 분할
        chunk_iter = chunker.chunk(dl_doc=doc)
        
        # 청크를 텍스트 리스트로 변환
        texts = []
        for chunk in chunk_iter:
            texts.append(chunk.text)
        
        # 작은 청크들 병합
        merged_texts = _merge_small_chunks(texts)
        
        print(f"✅ Docling Hybrid Chunking 성공: {len(texts)}개 → {len(merged_texts)}개 청크")
        return merged_texts
        
    except Exception as e:
        print(f"❌ Docling Hybrid Chunking 실패: {e}")
        # 폴백: 기본 텍스트 분할
        splitter = make_splitter()
        return splitter.split_text(text)

def _merge_small_chunks(texts: List[str], min_size: int = 100) -> List[str]:
    """100자 미만 청크들을 앞뒤 청크에 붙이기"""
    try:
        if not texts:
            return texts
        
        merged_texts = []
        i = 0
        
        while i < len(texts):
            current_chunk = texts[i]
            
            # 현재 청크가 100자 미만이면 앞뒤 중 더 적절한 곳에 병합
            if len(current_chunk.strip()) < min_size:
                # 앞 청크와 병합할지, 뒤 청크와 병합할지 결정
                prev_chunk = merged_texts[-1] if merged_texts else None
                next_chunk = texts[i + 1] if i + 1 < len(texts) else None
                
                # 앞 청크가 있고 병합 가능하면 앞에 붙이기
                if prev_chunk and len(prev_chunk + " " + current_chunk.strip()) <= 1200:
                    merged_texts[-1] = prev_chunk + " " + current_chunk.strip()
                    i += 1
                # 뒤 청크가 있고 병합 가능하면 뒤에 붙이기
                elif next_chunk and len(current_chunk.strip() + " " + next_chunk) <= 1200:
                    merged_chunk = current_chunk.strip() + " " + next_chunk
                    merged_texts.append(merged_chunk)
                    i += 2  # 두 청크를 건너뛰기
                else:
                    # 병합할 수 없으면 그대로 추가
                    merged_texts.append(current_chunk)
                    i += 1
            else:
                # 충분히 크면 그대로 추가
                merged_texts.append(current_chunk)
                i += 1
        
        return merged_texts
        
    except Exception as e:
        print(f"⚠️ 청크 병합 실패: {e}")
        return texts

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

# ===== PDF 처리 (Docling 텍스트 + OCR + 표) =====
def process_single_pdf(file_path: str) -> List[Document]:
    if not _HAS_PDFPLUMBER:
        raise RuntimeError("pdfplumber가 설치되어 있지 않습니다. `pip install pdfplumber` 필요")

    docs: List[Document] = []
    try:
        # --- Docling 텍스트 추출 ---
        print(f"🔄 Docling 텍스트 추출 중: {os.path.basename(file_path)}")
        try:
            converter = _get_docling_converter()
            result = converter.convert(file_path)
            doc = result.document
            
            # DoclingDocument에서 텍스트 추출 (최신 버전 호환)
            doc_text = ""
            if hasattr(doc, 'text'):
                doc_text = doc.text
            elif hasattr(doc, 'export_to_markdown'):
                doc_text = doc.export_to_markdown()
            elif hasattr(doc, 'elements'):
                # elements에서 텍스트 추출
                for element in doc.elements:
                    if hasattr(element, 'text'):
                        doc_text += element.text + "\n"
                    elif hasattr(element, 'content'):
                        doc_text += str(element.content) + "\n"
            else:
                print("⚠️ DoclingDocument에서 텍스트를 추출할 수 없습니다.")
                doc_text = ""
            
            if not doc_text.strip():
                print("⚠️ Docling에서 추출된 텍스트가 비어있습니다.")
                return docs
            
            # Docling 하이브리드 청킹 사용
            texts = _docling_hybrid_chunking(doc_text, os.path.basename(file_path))
            
            for i, text in enumerate(texts):
                cleaned_text = clean_text(text)
                if cleaned_text:
                    page_regions = detect_regions(cleaned_text)
                    page_years = detect_years(cleaned_text)
                    metadata = {
                        "file_name": os.path.basename(file_path),
                        "page": 1,  # Docling은 페이지 정보를 별도로 제공하지 않음
                        "type": "docling_text",
                        "regions": page_regions,
                        "years": page_years,
                        "chunk_index": i
                    }
                    docs.append(Document(page_content=cleaned_text, metadata=metadata))
            
            print(f"✅ Docling 텍스트 추출 완료: {len(texts)}개 청크")
        except Exception as e:
            print(f"⚠️ Docling 텍스트 추출 실패: {e}")

        # --- 텍스트 + OCR ---
        print(f"🔄 OCR 처리 중: {os.path.basename(file_path)}")
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
                # Docling 하이브리드 청킹 사용
                texts = _docling_hybrid_chunking(cleaned_text, os.path.basename(file_path))
                page_regions = detect_regions(cleaned_text)
                page_years = detect_years(cleaned_text)
                for i, chunk_text in enumerate(texts):
                    metadata = {
                        "file_name": os.path.basename(file_path),
                        "page": page_num,
                        "type": "pdf_text",
                        "regions": page_regions,
                        "years": page_years,
                        "ocr_applied": ocr_applied,
                        "chunk_index": i
                    }
                    docs.append(Document(page_content=chunk_text, metadata=metadata))

        # --- 표 추출 ---
        print(f"🔄 표 파싱 중: {os.path.basename(file_path)}")
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
                        # Docling 하이브리드 청킹 사용
                        texts = _docling_hybrid_chunking(cleaned, os.path.basename(file_path))
                        page_regions = detect_regions(cleaned)
                        page_years = detect_years(cleaned)
                        for i, chunk_text in enumerate(texts):
                            metadata = {
                                "file_name": os.path.basename(file_path),
                                "page": page_num,
                                "type": "pdf_table",
                                "regions": page_regions,
                                "years": page_years,
                                "chunk_index": i
                            }
                            docs.append(Document(page_content=chunk_text, metadata=metadata))
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
        # Docling 하이브리드 청킹 사용
        texts = _docling_hybrid_chunking(cleaned_text, os.path.basename(file_path))
        image_regions = detect_regions(cleaned_text)
        image_years = detect_years(cleaned_text)
        for i, chunk_text in enumerate(texts):
            metadata = {
                "file_name": os.path.basename(file_path),
                "page": 1,
                "type": "image_ocr",
                "regions": image_regions,
                "years": image_years,
                "chunk_index": i
            }
            docs.append(Document(page_content=chunk_text, metadata=metadata))
    except Exception as e:
        print(f"이미지 처리 실패 {file_path}: {e}")
    return docs

# ===== 메인 =====
if __name__ == "__main__":
    print("🚀 하이브리드 인덱스 (Docling + OCR + 표) → Milvus 업서트 시작")

    if not _HAS_PDFPLUMBER:
        raise RuntimeError("pdfplumber 미설치 → 표 파싱 불가. `pip install pdfplumber` 필요")
    
    if not _HAS_DOCLING:
        raise RuntimeError("Docling 미설치 → 텍스트 파싱 불가. `pip install docling docling-core` 필요")

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
