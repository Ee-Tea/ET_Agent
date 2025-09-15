"""
Docling을 사용한 문서 처리 및 MilvusDB 업로드 RAG 시스템
공식 예시 참조: https://docling-project.github.io/docling/examples/hybrid_chunking/
"""
import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)
# Docling 관련
from docling.document_converter import DocumentConverter

load_dotenv()
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
    """Docling을 사용한 문서 처리 및 MilvusDB 업로드 클래스"""
    
    def __init__(self, 
                 collection_name: str = COLLECTION_NAME):
        """
        초기화
        
        Args:
            collection_name: 컬렉션 이름
        """
        self.collection_name = collection_name
        
        # Docling 변환기 초기화
        self.converter = self._init_docling_converter()
        
        # 임베딩 모델 초기화 (기존 한국어 모델)
        self.embeddings = self._init_embeddings()
        
        # Milvus 연결 초기화 (기존 방식과 동일)
        self._init_milvus_client()
        
    
    def _init_docling_converter(self) -> DocumentConverter:
        """Docling 변환기 초기화 (모든 방법 동시 적용)"""
        try:
            print("🔄 Docling 변환기 초기화 중...")
            
            # 모든 방법을 동시에 적용
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
            
            # 방법 1 + 2 + 3을 모두 적용한 최적 설정
            pipeline_options = PdfPipelineOptions()
            
            # 방법 1: 최소한의 설정 (안정적)
            pipeline_options.do_ocr = False  # OCR 비활성화
            pipeline_options.do_table_structure = False  # 테이블 구조 비활성화
            
            # 방법 3: CPU만 사용 (GPU 문제 방지)
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=2, 
                device=AcceleratorDevice.CPU
            )
            
            # 방법 2: PyPdfium 백엔드 사용 (더 안정적)
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options, 
                        backend=PyPdfiumDocumentBackend
                    )
                }
            )
            
            print("✅ Docling 변환기 초기화 완료")
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
    
    
    def process_document(self, file_path: str) -> List[str]:
        """
        문서를 Docling으로 처리하여 텍스트 청크 리스트 반환
        
        Args:
            file_path: 처리할 문서 파일 경로
            
        Returns:
            텍스트 청크 리스트
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            print(f"📄 문서 처리 시작: {file_path.name}")
            
            # Docling으로 문서 변환 (공식 방법)
            result = self.converter.convert(str(file_path))
            doc = result.document  # DoclingDocument 객체
            
            # 하이브리드 청킹 시도 (Docling 구조 기반)
            texts = self._hybrid_chunking(doc)
            
            print(f"✅ 문서 처리 완료: {file_path.name} - {len(texts)}개 청크")
            return texts
            
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
            
            # 토크나이저 설정
            tokenizer = OpenAITokenizer(
                tokenizer=tiktoken.encoding_for_model("gpt-4o-mini"),
                max_tokens=128 * 1024,  # context window length required for OpenAI tokenizers
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
    
    def _merge_small_chunks(self, texts: List[str], min_size: int = 50) -> List[str]:
        """50자 미만 청크들을 앞뒤 청크에 붙이기"""
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
                    
                    # 앞 청크가 있고 병합 가능하면 앞에 붙이기
                    if prev_chunk and len(prev_chunk + " " + current_chunk.strip()) <= 2000:
                        merged_texts[-1] = prev_chunk + " " + current_chunk.strip()
                        i += 1
                    # 뒤 청크가 있고 병합 가능하면 뒤에 붙이기
                    elif next_chunk and len(current_chunk.strip() + " " + next_chunk) <= 2000:
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
    
    def upload_documents_to_milvus(self, texts: List[str], file_path: str) -> bool:
        """문서들을 MilvusDB에 업로드 (기존 방식과 동일)"""
        try:
            if not texts:
                print("⚠️ 업로드할 문서가 없습니다")
                return False
            
            print(f"📤 {len(texts)}개 청크를 MilvusDB에 업로드 중...")
            
            # 컬렉션 가져오기
            collection = Collection(name=self.collection_name)
            
            # 배치 처리
            batch_size = 100
            file_name = os.path.basename(file_path)
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                print(f"  - 배치 {i//batch_size + 1}: {i+1} ~ {i+len(batch_texts)}")
                
                # 임베딩 생성
                vectors = []
                for text in batch_texts:
                    embedding = self.get_embedding(text)
                    if embedding:
                        vectors.append(embedding)
                
                # 메타데이터 준비 (기존 방식과 동일)
                file_names = [file_name] * len(batch_texts)
                pages = [1] * len(batch_texts)  # Docling은 페이지 정보를 별도로 제공하지 않음
                types = ["docling_text"] * len(batch_texts)
                regions = [[]] * len(batch_texts)  # 빈 리스트
                years = [[]] * len(batch_texts)    # 빈 리스트
                
                # MilvusDB에 삽입
                try:
                    collection.insert([batch_texts, vectors, file_names, pages, types, regions, years])
                    print(f"    > ✅ 삽입 성공: {len(batch_texts)}개")
                except Exception as e:
                    print(f"    > ⚠️ 삽입 실패: {e}")
            
            # Flush 대기
            print("⌛ flush 대기...")
            collection.flush()
            print(f"📊 num_entities: {collection.num_entities}")
            
            print(f"✅ 문서 업로드 완료: {len(texts)}개 문서")
            return True
            
        except Exception as e:
            print(f"❌ 문서 업로드 실패: {e}")
            return False
    
    def preview_document(self, file_path: str, show_chunks: int = 10) -> None:
        """문서 처리 결과를 미리보기 (확장된 정보 표시)"""
        try:
            # 문서 처리
            texts = self.process_document(file_path)
            
            if not texts:
                print("❌ 처리할 문서가 없습니다.")
                return
            
            file_path = Path(file_path)
            print(f"📁 파일명: {file_path.name} ({len(texts)}개 청크)")
            print("=" * 100)
            
            for i, text in enumerate(texts[:show_chunks]):
                print(f"청크 {i+1}/{len(texts)}:")
                print(f"길이: {len(text)}자")
                print(f"전체 내용:")
                print(text)  # 전체 내용 표시
                print("-" * 100)
            
            if len(texts) > show_chunks:
                print(f"... 및 {len(texts) - show_chunks}개 청크 더")
            
            print("=" * 100)
            
        except Exception as e:
            print(f"❌ 미리보기 실패: {e}")
    
    def process_and_upload_file(self, file_path: str) -> bool:
        """파일을 처리하고 MilvusDB에 업로드하는 통합 함수"""
        try:
            # 1. 문서 처리
            texts = self.process_document(file_path)
            if not texts:
                print("❌ 문서 처리 실패")
                return False
            
            # 2. 컬렉션 생성 (필요한 경우)
            if not utility.has_collection(self.collection_name):
                if not self.create_collection():
                    print("❌ 컬렉션 생성 실패")
                    return False
            
            # 3. 문서 업로드
            if not self.upload_documents_to_milvus(texts, file_path):
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
            
            # 각 파일 처리
            results = {}
            for pdf_file in pdf_files:
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