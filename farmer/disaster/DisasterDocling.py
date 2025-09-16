"""
Docling을 사용한 문서 처리 및 MilvusDB 업로드 RAG 시스템
공식 예시 참조: https://docling-project.github.io/docling/examples/hybrid_chunking/
"""
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)
# Docling 관련
from docling.document_converter import DocumentConverter

# OCR과 표 파싱을 위한 추가 라이브러리 (db_disaster.py에서 가져옴)
import fitz  # PyMuPDF
import easyocr
import numpy as np

# pdfplumber import (표 파싱용)
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

load_dotenv()

# HuggingFace 캐시 디렉토리 설정 (권한 문제 해결)
import os
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')

# =========[ 전역 설정 변수 ]=========
# PDF 파일 경로 설정
PDF_FILE_PATH = "./farmer/disaster/data/2025 기상재해 대응기술 가이드북(주요 20작물).pdf"
PDF_DIRECTORY_PATH = "./farmer/disaster/data/"  # 디렉토리에 있는 전체 PDF 처리

# Milvus 설정
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "disaster_documents"

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
EMBEDDING_DIM = 768  # ko-sroberta-multitask 모델의 차원



class DoclingRAGProcessor:
    """Docling 텍스트 + OCR + 표 파싱 하이브리드 문서 처리 클래스"""
    
    def __init__(self, 
                 collection_name: str = COLLECTION_NAME):
        """
        초기화
        
        Args:
            collection_name: 컬렉션 이름
        """
        self.collection_name = collection_name
        
        # Docling 변환기 초기화 (텍스트만)
        self.converter = self._init_docling_converter()
        
        # OCR 리더 초기화 (db_disaster.py에서 가져옴)
        self.ocr_reader = None
        
        # 임베딩 모델 초기화 (기존 한국어 모델)
        self.embeddings = self._init_embeddings()
        
        # Milvus 연결 초기화 (기존 방식과 동일)
        self._init_milvus_client()
        
    
    def _init_docling_converter(self) -> DocumentConverter:
        """Docling 변환기 초기화 (텍스트만 추출)"""
        try:
            print("🔄 Docling 변환기 초기화 중... (텍스트만)")
            
            # 텍스트만 추출하도록 설정 (OCR과 표 파싱은 별도 처리)
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
            
            pipeline_options = PdfPipelineOptions()
            
            # OCR과 표 파싱 비활성화 (별도 처리)
            pipeline_options.do_ocr = False  # OCR 비활성화
            pipeline_options.do_table_structure = False  # 표 구조 비활성화
            
            # CPU만 사용
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=2, 
                device=AcceleratorDevice.CPU
            )
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options, 
                        backend=PyPdfiumDocumentBackend
                    )
                }
            )
            
            print("✅ Docling 변환기 초기화 완료 (텍스트만)")
            return converter
            
        except Exception as e:
            print(f"⚠️ 최적화된 설정 실패, 기본 설정으로 시도: {e}")
            # 대안: 기본 설정
            try:
                converter = DocumentConverter()
                print("✅ 기본 변환기 초기화 완료")
                return converter
            except Exception as e2:
                print(f"❌ Docling 변환기 초기화 실패: {e2}")
                raise
    
    def _init_milvus_client(self) -> None:
        """Milvus 연결 초기화 (기존 방식과 동일)"""
        try:
            print("🔄 Milvus 연결 중...")
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            print("✅ Milvus 연결 완료")
        except Exception as e:
            print(f"❌ Milvus 연결 실패: {e}")
            raise
    
    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """임베딩 모델 초기화 (기존 한국어 모델)"""
        try:
            print("🔄 임베딩 모델 초기화 중...")
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            print("✅ 임베딩 모델 초기화 완료")
            return embeddings
            
        except Exception as e:
            print(f"❌ 임베딩 모델 초기화 실패: {e}")
            raise
    
    
    def clean_text(self, text: str) -> str:
        """텍스트 전처리 (db_disaster.py와 동일)"""
        text = re.sub(r"[^\w\s\.\,\(\)\/%\:\-\~가-힣]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def detect_years(self, text: str) -> List[int]:
        """연도 추출 (db_disaster.py와 동일)"""
        _YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
        years = set()
        for m in _YEAR_RE.finditer(text or ""):
            y = int(m.group(1))
            if 1900 <= y <= 2100:
                years.add(y)
        return sorted(years)
    
    def detect_regions(self, text: str) -> List[str]:
        """지역 추출 (db_disaster.py와 동일)"""
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
        
        t = (text or "").replace(" ", "")
        hits = []
        for k in REGION_KEYS:
            if k in t: hits.append(canon_region(k))
        return sorted(list(set(hits)))
    
    def _get_ocr_reader(self) -> easyocr.Reader:
        """OCR 리더 초기화 (db_disaster.py에서 가져옴)"""
        if self.ocr_reader is None:
            print("EasyOCR Reader 로드 중...")
            self.ocr_reader = easyocr.Reader(["ko", "en"], gpu=True)
        return self.ocr_reader
    
    def _process_pdf_with_ocr_and_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """PDF를 OCR과 표 파싱으로 처리 (db_disaster.py와 완전 동일)"""
        if not _HAS_PDFPLUMBER:
            raise RuntimeError("pdfplumber가 설치되어 있지 않습니다. `pip install pdfplumber` 필요")

        chunks_data = []
        try:
            # --- 텍스트 + OCR ---
            doc_fitz = fitz.open(file_path)
            for page_num, page in enumerate(doc_fitz, start=1):
                text = page.get_text()
                try:
                    reader = self._get_ocr_reader()
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    ocr_results = reader.readtext(img_bytes)
                    ocr_text = " ".join([res[1] for res in ocr_results])
                    text = text + " " + ocr_text
                    ocr_applied = True
                except Exception as e:
                    print(f"OCR 실패: {file_path} p.{page_num} - {e}")
                    ocr_applied = False

                cleaned_text = self.clean_text(text)
                if cleaned_text:
                    # Docling 하이브리드 청킹 사용
                    texts = self._hybrid_chunking_from_text(cleaned_text)
                    page_regions = self.detect_regions(cleaned_text)
                    page_years = self.detect_years(cleaned_text)
                    for i, chunk_text in enumerate(texts):
                        chunks_data.append({
                            "text": chunk_text,
                            "file_name": os.path.basename(file_path),
                            "page": page_num,
                            "type": "pdf_text",
                            "regions": page_regions,
                            "years": page_years,
                            "chunk_index": i,
                            "ocr_applied": ocr_applied
                        })

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
                            cleaned = self.clean_text(table_str)
                            if not cleaned:
                                continue
                            # Docling 하이브리드 청킹 사용
                            texts = self._hybrid_chunking_from_text(cleaned)
                            page_regions = self.detect_regions(cleaned)
                            page_years = self.detect_years(cleaned)
                            for i, chunk_text in enumerate(texts):
                                chunks_data.append({
                                    "text": chunk_text,
                                    "file_name": os.path.basename(file_path),
                                    "page": page_num,
                                    "type": "pdf_table",
                                    "regions": page_regions,
                                    "years": page_years,
                                    "chunk_index": i
                                })
                    except Exception as e:
                        print(f"표 파싱 실패: {file_path} p.{page_num} - {e}")

        except Exception as e:
            print(f"PDF 처리 실패 {file_path}: {e}")
        return chunks_data
    
    def _hybrid_chunking_from_text(self, text: str) -> List[str]:
        """텍스트에서 Docling 하이브리드 청킹 사용"""
        try:
            print("🧠 Docling Hybrid Chunking 시작...")
            
            # Docling의 공식 HybridChunker 사용
            import tiktoken
            from docling.chunking import HybridChunker
            from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
            
            # 토크나이저 설정
            tokenizer = OpenAITokenizer(
                tokenizer=tiktoken.encoding_for_model("gpt-4o-mini"),
                max_tokens=1200,  # 약 900자에 해당하는 토큰 수
            )
            
            chunker = HybridChunker(
                tokenizer=tokenizer,
                merge_peers=True,
            )
            
            # 임시 DoclingDocument 생성
            from docling.datamodel.document import DoclingDocument
            doc = DoclingDocument(
                name="temp",
                text=text,
                elements=[]
            )
            
            # 문서를 청크로 분할
            chunk_iter = chunker.chunk(dl_doc=doc)
            
            # 청크를 텍스트 리스트로 변환
            texts = []
            for chunk in chunk_iter:
                texts.append(chunk.text)
            
            # 작은 청크들 병합
            merged_texts = self._merge_small_chunks(texts)
            
            print(f"✅ Docling Hybrid Chunking 성공: {len(texts)}개 → {len(merged_texts)}개 청크")
            return merged_texts
            
        except Exception as e:
            print(f"❌ Docling Hybrid Chunking 실패: {e}")
            # 폴백: 기본 텍스트 분할
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=900,
                chunk_overlap=150,
                length_function=len
            )
            return splitter.split_text(text)
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        하이브리드 문서 처리: Docling 텍스트 + OCR + 표 파싱
        
        Args:
            file_path: 처리할 문서 파일 경로
            
        Returns:
            청크와 메타데이터가 포함된 딕셔너리 리스트
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            print(f"📄 하이브리드 문서 처리 시작: {file_path.name}")
            
            # 1. Docling으로 텍스트 추출 (구조적 텍스트)
            docling_chunks = []
            try:
                print("🔄 Docling 텍스트 추출 중...")
                result = self.converter.convert(str(file_path))
                doc = result.document
                texts = self._hybrid_chunking(doc)
                
                for i, text in enumerate(texts):
                    cleaned_text = self.clean_text(text)
                    if cleaned_text:
                        regions = self.detect_regions(cleaned_text)
                        years = self.detect_years(cleaned_text)
                        
                        docling_chunks.append({
                            "text": cleaned_text,
                            "file_name": file_path.name,
                            "page": 1,  # Docling은 페이지 정보를 별도로 제공하지 않음
                            "type": "docling_text",
                            "regions": regions,
                            "years": years,
                            "chunk_index": i
                        })
                print(f"✅ Docling 텍스트 추출 완료: {len(docling_chunks)}개 청크")
            except Exception as e:
                print(f"⚠️ Docling 텍스트 추출 실패: {e}")
            
            # 2. OCR + 표 파싱 처리 (db_disaster.py 방식 - 항상 실행)
            print("🔄 OCR + 표 파싱 처리 중...")
            ocr_table_chunks = self._process_pdf_with_ocr_and_tables(str(file_path))
            print(f"✅ OCR + 표 파싱 완료: {len(ocr_table_chunks)}개 청크")
            
            # 3. 모든 청크 통합
            all_chunks = docling_chunks + ocr_table_chunks
            
            print(f"✅ 하이브리드 문서 처리 완료: {file_path.name} - {len(all_chunks)}개 청크")
            print(f"   - Docling 텍스트: {len(docling_chunks)}개")
            print(f"   - OCR + 표: {len(ocr_table_chunks)}개")
            
            return all_chunks
            
        except Exception as e:
            print(f"❌ 문서 처리 실패: {file_path} - {e}")
            return []
    
    def _hybrid_chunking(self, doc) -> List[str]:
        """Docling 공식 Hybrid Chunking 사용"""
        try:
            print("🧠 Docling Hybrid Chunking 시작...")
            
            # Docling의 공식 HybridChunker 사용 (예제 코드 방식)
            import tiktoken
            from docling.chunking import HybridChunker
            from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
            
            # 토크나이저 설정 (db_disaster.py와 비슷한 크기로 조정)
            tokenizer = OpenAITokenizer(
                tokenizer=tiktoken.encoding_for_model("gpt-4o-mini"),
                max_tokens=1200,  # 약 900자에 해당하는 토큰 수 (db_disaster.py CHUNK_SIZE=900과 비슷)
            )
            
            chunker = HybridChunker(
                tokenizer=tokenizer,
                merge_peers=True,  # optional, defaults to True
            )
            
            # 문서를 청크로 분할
            chunk_iter = chunker.chunk(dl_doc=doc)
            
            # 청크를 텍스트 리스트로 변환
            texts = []
            for i, chunk in enumerate(chunk_iter):
                # 기본 청크 텍스트 사용 (제목 없이 순수한 내용만)
                chunk_text = chunk.text
                texts.append(chunk_text)
            
            # 50자 미만 청크들을 앞뒤 청크에 붙이기
            merged_texts = self._merge_small_chunks(texts)
            
            print(f"✅ Docling Hybrid Chunking 성공: {len(texts)}개 → {len(merged_texts)}개 청크")
            return merged_texts
            
        except Exception as e:
            print(f"❌ Docling Hybrid Chunking 실패: {e}")
            return []
    
    def _merge_small_chunks(self, texts: List[str], min_size: int = 100) -> List[str]:
        """100자 미만 청크들을 앞뒤 청크에 붙이기 (db_disaster.py와 비슷한 크기)"""
        try:
            if not texts:
                return texts
            
            merged_texts = []
            i = 0
            
            while i < len(texts):
                current_chunk = texts[i]
                
                # 현재 청크가 50자 미만이면 앞뒤 중 더 적절한 곳에 병합
                if len(current_chunk.strip()) < min_size:
                    # 앞 청크와 병합할지, 뒤 청크와 병합할지 결정
                    prev_chunk = merged_texts[-1] if merged_texts else None
                    next_chunk = texts[i + 1] if i + 1 < len(texts) else None
                    
                    # 앞 청크가 있고 병합 가능하면 앞에 붙이기 (db_disaster.py CHUNK_SIZE+OVERLAP=1050과 비슷)
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
    
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성 (기존 한국어 모델 사용)"""
        try:
            # HuggingFace 임베딩 모델 사용
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"❌ 임베딩 생성 실패: {e}")
            return []
    
    
    def create_collection(self) -> bool:
        """Milvus 컬렉션 생성 (기존 방식과 동일)"""
        try:
            print(f"🔄 컬렉션 생성 중: {self.collection_name}")
            
            # 기존 컬렉션이 있으면 삭제
            if utility.has_collection(self.collection_name):
                print(f"🔄 기존 컬렉션 삭제: {self.collection_name}")
                utility.drop_collection(self.collection_name)
            
            # 새 컬렉션 생성 (기존 방식과 동일한 스키마)
            fields = [
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="page", dtype=DataType.INT64),
                FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="regions", dtype=DataType.JSON),
                FieldSchema(name="years", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields, "재해 대응 컬렉션")
            collection = Collection(name=self.collection_name, schema=schema)
            
            # 인덱스 생성
            index_params = {"metric_type": "IP", "index_type": "AUTOINDEX", "params": {}}
            collection.create_index(field_name="vector", index_params=index_params)
            
            print(f"✅ 컬렉션 생성 완료: {self.collection_name}")
            return True
            
        except Exception as e:
            print(f"❌ 컬렉션 생성 실패: {e}")
            return False
    
    def upload_documents_to_milvus(self, chunks_data: List[Dict[str, Any]], file_path: str) -> bool:
        """문서들을 MilvusDB에 업로드 (메타데이터 포함)"""
        try:
            if not chunks_data:
                print("⚠️ 업로드할 문서가 없습니다")
                return False
            
            print(f"📤 {len(chunks_data)}개 청크를 MilvusDB에 업로드 중...")
            
            # 컬렉션 가져오기
            collection = Collection(name=self.collection_name)
            
            # 배치 처리
            batch_size = 100
            
            for i in range(0, len(chunks_data), batch_size):
                batch_chunks = chunks_data[i:i+batch_size]
                print(f"  - 배치 {i//batch_size + 1}: {i+1} ~ {i+len(batch_chunks)}")
                
                # 임베딩 생성
                vectors = []
                texts = []
                file_names = []
                pages = []
                types = []
                regions = []
                years = []
                
                for chunk_data in batch_chunks:
                    embedding = self.get_embedding(chunk_data["text"])
                    if embedding:
                        vectors.append(embedding)
                        texts.append(chunk_data["text"])
                        file_names.append(chunk_data["file_name"])
                        pages.append(chunk_data["page"])
                        types.append(chunk_data["type"])
                        regions.append(chunk_data["regions"])
                        years.append(chunk_data["years"])
                
                # MilvusDB에 삽입
                try:
                    collection.insert([texts, vectors, file_names, pages, types, regions, years])
                    print(f"    > ✅ 삽입 성공: {len(batch_chunks)}개")
                except Exception as e:
                    print(f"    > ⚠️ 삽입 실패: {e}")
            
            # Flush 대기
            print("⌛ flush 대기...")
            collection.flush()
            print(f"📊 num_entities: {collection.num_entities}")
            
            print(f"✅ 문서 업로드 완료: {len(chunks_data)}개 문서")
            return True
            
        except Exception as e:
            print(f"❌ 문서 업로드 실패: {e}")
            return False
    
    def preview_document(self, file_path: str, show_chunks: int = 10) -> None:
        """문서 처리 결과를 미리보기 (확장된 정보 표시)"""
        try:
            # 문서 처리
            chunks_data = self.process_document(file_path)
            
            if not chunks_data:
                print("❌ 처리할 문서가 없습니다.")
                return
            
            file_path = Path(file_path)
            print(f"📁 파일명: {file_path.name} ({len(chunks_data)}개 청크)")
            print("=" * 100)
            
            for i, chunk_data in enumerate(chunks_data[:show_chunks]):
                print(f"청크 {i+1}/{len(chunks_data)}:")
                print(f"길이: {len(chunk_data['text'])}자")
                print(f"지역: {chunk_data['regions']}")
                print(f"연도: {chunk_data['years']}")
                print(f"타입: {chunk_data['type']}")
                print(f"전체 내용:")
                print(chunk_data['text'])  # 전체 내용 표시
                print("-" * 100)
            
            if len(chunks_data) > show_chunks:
                print(f"... 및 {len(chunks_data) - show_chunks}개 청크 더")
            
            print("=" * 100)
            
        except Exception as e:
            print(f"❌ 미리보기 실패: {e}")
    
    def process_and_upload_file(self, file_path: str) -> bool:
        """파일을 처리하고 MilvusDB에 업로드하는 통합 함수"""
        try:
            # 1. 문서 처리
            chunks_data = self.process_document(file_path)
            if not chunks_data:
                print("❌ 문서 처리 실패")
                return False
            
            # 2. 컬렉션 생성 (필요한 경우)
            if not utility.has_collection(self.collection_name):
                if not self.create_collection():
                    print("❌ 컬렉션 생성 실패")
                    return False
            
            # 3. 문서 업로드
            if not self.upload_documents_to_milvus(chunks_data, file_path):
                print("❌ 문서 업로드 실패")
                return False
            
            print(f"✅ 파일 처리 및 업로드 완료: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ 파일 처리 및 업로드 실패: {file_path} - {e}")
            return False
    
    def process_directory(self, directory_path: str) -> Dict[str, bool]:
        """디렉토리 내 모든 PDF 파일을 처리하고 MilvusDB에 업로드"""
        try:
            from glob import glob
            
            # PDF 파일 찾기
            pdf_pattern = os.path.join(directory_path, "*.pdf")
            pdf_files = glob(pdf_pattern)
            
            if not pdf_files:
                print(f"⚠️ 디렉토리에서 PDF 파일을 찾을 수 없습니다: {directory_path}")
                return {}
            
            print(f"📁 {len(pdf_files)}개 PDF 파일 발견: {directory_path}")
            
            # 컬렉션 생성 (필요한 경우)
            if not utility.has_collection(self.collection_name):
                if not self.create_collection():
                    print("❌ 컬렉션 생성 실패")
                    return {}
            
            # 각 파일 처리 (진행률 표시)
            results = {}
            for pdf_file in tqdm(pdf_files, desc="PDF 파일 처리"):
                print(f"🔄 처리 중: {os.path.basename(pdf_file)}")
                success = self.process_and_upload_file(pdf_file)
                results[pdf_file] = success
                
                if success:
                    print(f"✅ 성공: {os.path.basename(pdf_file)}")
                else:
                    print(f"❌ 실패: {os.path.basename(pdf_file)}")
            
            # 결과 요약
            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            
            print(f"📊 디렉토리 처리 완료: {success_count}/{total_count}개 파일 성공")
            return results
            
        except Exception as e:
            print(f"❌ 디렉토리 처리 실패: {directory_path} - {e}")
            return {}
    
    def search_documents(self, query: str, k: int = 3) -> List[Dict]:
        """문서 검색 (기존 방식과 동일)"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                print("❌ 쿼리 임베딩 생성 실패")
                return []
            
            # 컬렉션 가져오기
            collection = Collection(name=self.collection_name)
            collection.load()
            
            # MilvusDB에서 검색
            search_params = {"metric_type": "IP", "params": {}}
            search_res = collection.search(
                data=[query_embedding],
                anns_field="vector",
                param=search_params,
                limit=k,
                output_fields=["text", "file_name", "page", "type"]
            )
            
            # 결과 처리
            results = []
            for hits in search_res:
                for hit in hits:
                    results.append({
                        "text": hit.entity.get("text"),
                        "file_name": hit.entity.get("file_name"),
                        "page": hit.entity.get("page"),
                        "type": hit.entity.get("type"),
                        "distance": hit.distance
                    })
            
            print(f"✅ 검색 완료: {len(results)}개 문서 반환")
            return results
            
        except Exception as e:
            print(f"❌ 검색 실패: {e}")
            return []
    
    


def main():
    """메인 실행 함수"""
    try:
        # RAG 프로세서 초기화
        processor = DoclingRAGProcessor(
            collection_name=COLLECTION_NAME
        )
        
        print("✅ DoclingRAGProcessor 초기화 완료")

        # 선택 메뉴
        print("=" * 60)
        print("🎯 처리 방식을 선택하세요:")
        print("1. 단일 PDF 파일 미리보기 (저장 안함)")
        print("2. 디렉토리 내 모든 PDF 파일을 Milvus에 저장")
        print("=" * 60)
        
        while True:
            try:
                choice = input("선택 (1 또는 2): ").strip()
                
                if choice == "1":
                    print("\n🔍 단일 파일 미리보기...")
                    processor.preview_document(PDF_FILE_PATH, show_chunks=10)
                    print("\n✅ 미리보기 완료!")
                    break
                    
                elif choice == "2":
                    print(f"\n🔄 디렉토리 내 모든 PDF 파일을 Milvus에 저장합니다...")
                    from glob import glob
                    pdf_files = glob(os.path.join(PDF_DIRECTORY_PATH, "*.pdf"))
                    print(f"발견된 PDF 파일: {len(pdf_files)}개")
                    for i, pdf_file in enumerate(pdf_files, 1):
                        print(f"  {i}. {os.path.basename(pdf_file)}")
                    print()
                    
                    print("🔄 디렉토리 처리 중...")
                    results = processor.process_directory(PDF_DIRECTORY_PATH)
                    
                    print("\n📊 처리 결과:")
                    for file_path, success in results.items():
                        status = "✅ 성공" if success else "❌ 실패"
                        print(f"  {os.path.basename(file_path)}: {status}")
                    
                    success_count = sum(1 for success in results.values() if success)
                    total_count = len(results)
                    print(f"\n전체 결과: {success_count}/{total_count}개 파일 성공")
                    break
                    
                else:
                    print("❌ 잘못된 선택입니다. 1 또는 2를 입력하세요.")
                    
            except KeyboardInterrupt:
                print("\n\n프로그램이 중단되었습니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                break
        
        
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")


if __name__ == "__main__":
    main()