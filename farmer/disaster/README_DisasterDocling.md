# DisasterDocling - Docling 기반 문서 처리 및 MilvusDB 업로드 시스템

## 📋 개요

`DisasterDocling.py`는 **Docling**을 사용하여 PDF 문서를 처리하고 **MilvusDB**에 벡터 임베딩으로 저장하는 시스템입니다. 기상재해 대응 기술 가이드북과 같은 문서를 효율적으로 처리하고 검색할 수 있도록 설계되었습니다.

## 🚀 주요 기능

### 1. **Docling 기반 문서 처리**
- PDF 문서를 구조화된 텍스트로 변환
- OCR 및 테이블 구조 인식 비활성화로 안정성 향상
- CPU 전용 처리로 GPU 의존성 제거

### 2. **Hybrid Chunking**
- Docling의 공식 HybridChunker 사용
- OpenAI Tokenizer 기반 토큰화
- 문서 구조를 고려한 의미적 청크 분할
- 50자 미만 청크 자동 병합

### 3. **한국어 임베딩**
- `jhgan/ko-sroberta-multitask` 모델 사용
- 768차원 벡터 임베딩 생성
- CPU 기반 처리로 안정성 확보

### 4. **MilvusDB 통합**
- 벡터 데이터베이스에 문서 저장
- 배치 처리로 효율적인 업로드
- 메타데이터 포함 (파일명, 페이지, 타입 등)

### 5. **사용자 친화적 인터페이스**
- 실시간 진행 상황 표시
- 미리보기 기능으로 처리 결과 확인
- 단일 파일 또는 디렉토리 일괄 처리

## 🛠️ 설치 및 설정

### 필수 패키지
```bash
pip install docling>=2.43.0
pip install pymilvus>=2.5.14
pip install langchain-huggingface>=0.3.1
pip install python-dotenv
pip install tiktoken
```

### 환경 설정
```python
# 전역 설정 변수 (코드 상단에서 수정 가능)
PDF_FILE_PATH = "./farmer/disaster/data/2025 기상재해 대응기술 가이드북(주요 20작물).pdf"
PDF_DIRECTORY_PATH = "./farmer/disaster/data/"

# Milvus 설정
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "disaster_documents"

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
EMBEDDING_DIM = 768
```

## 📖 사용법

### 1. 기본 실행
```bash
python DisasterDocling.py
```

### 2. 실행 옵션
프로그램 실행 시 다음 중 하나를 선택할 수 있습니다:

- **옵션 1**: 단일 PDF 파일 미리보기 (저장하지 않음)
- **옵션 2**: 디렉토리 내 모든 PDF 파일을 MilvusDB에 저장

### 3. 프로그래밍 방식 사용
```python
from DisasterDocling import DoclingRAGProcessor

# 프로세서 초기화
processor = DoclingRAGProcessor(collection_name="my_documents")

# 단일 파일 처리 및 업로드
success = processor.process_and_upload_file("path/to/document.pdf")

# 디렉토리 일괄 처리
results = processor.process_directory("path/to/documents/")

# 문서 검색
results = processor.search_documents("검색어", k=5)
```

## 🏗️ 코드 구조

### 주요 클래스: `DoclingRAGProcessor`

#### 초기화 메서드
- `_init_docling_converter()`: Docling 변환기 설정
- `_init_milvus_client()`: MilvusDB 연결
- `_init_embeddings()`: 한국어 임베딩 모델 로드

#### 문서 처리 메서드
- `process_document(file_path)`: PDF를 텍스트 청크로 변환
- `_hybrid_chunking(doc)`: Docling Hybrid Chunking 적용
- `_merge_small_chunks(texts)`: 작은 청크 병합

#### 데이터베이스 메서드
- `create_collection()`: Milvus 컬렉션 생성
- `upload_documents_to_milvus(texts, file_path)`: 문서 업로드
- `search_documents(query, k)`: 벡터 검색

#### 유틸리티 메서드
- `preview_document(file_path, show_chunks)`: 처리 결과 미리보기
- `process_and_upload_file(file_path)`: 파일 처리 및 업로드 통합
- `process_directory(directory_path)`: 디렉토리 일괄 처리

## 🔧 기술적 특징

### Docling 설정
```python
# 최적화된 설정
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False  # OCR 비활성화
pipeline_options.do_table_structure = False  # 테이블 구조 비활성화
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=2, 
    device=AcceleratorDevice.CPU
)
```

### Hybrid Chunking 설정
```python
tokenizer = OpenAITokenizer(
    tokenizer=tiktoken.encoding_for_model("gpt-4o-mini"),
    max_tokens=128 * 1024,
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,
)
```

### MilvusDB 스키마
```python
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="page", dtype=DataType.INT64),
    FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="regions", dtype=DataType.JSON),
    FieldSchema(name="years", dtype=DataType.JSON),
]
```

## 📊 실행 예시

### 미리보기 실행
```
🔄 Docling 변환기 초기화 중...
✅ Docling 변환기 초기화 완료
🔄 Milvus 연결 중...
✅ Milvus 연결 완료
🔄 임베딩 모델 초기화 중...
✅ 임베딩 모델 초기화 완료
✅ DoclingRAGProcessor 초기화 완료

🎯 처리 방식을 선택하세요:
1. 단일 PDF 파일 미리보기 (저장 안함)
2. 디렉토리 내 모든 PDF 파일을 Milvus에 저장

선택 (1 또는 2): 1

🔍 단일 파일 미리보기...
📄 문서 처리 시작: 2025 기상재해 대응기술 가이드북(주요 20작물).pdf
🧠 Docling Hybrid Chunking 시작...
✅ Docling Hybrid Chunking 성공: 45개 → 38개 청크
✅ 문서 처리 완료: 2025 기상재해 대응기술 가이드북(주요 20작물).pdf - 38개 청크

📁 파일명: 2025 기상재해 대응기술 가이드북(주요 20작물).pdf (38개 청크)
====================================================================================================
청크 1/38:
길이: 1250자
전체 내용:
[청크 내용 표시]
```

### 업로드 실행
```
선택 (1 또는 2): 2

🔄 디렉토리 내 모든 PDF 파일을 Milvus에 저장합니다...
발견된 PDF 파일: 1개
  1. 2025 기상재해 대응기술 가이드북(주요 20작물).pdf

🔄 디렉토리 처리 중...
📁 1개 PDF 파일 발견: ./farmer/disaster/data/
🔄 컬렉션 생성 중: disaster_documents
🔄 기존 컬렉션 삭제: disaster_documents
✅ 컬렉션 생성 완료: disaster_documents
🔄 처리 중: 2025 기상재해 대응기술 가이드북(주요 20작물).pdf
📄 문서 처리 시작: 2025 기상재해 대응기술 가이드북(주요 20작물).pdf
🧠 Docling Hybrid Chunking 시작...
✅ Docling Hybrid Chunking 성공: 45개 → 38개 청크
✅ 문서 처리 완료: 2025 기상재해 대응기술 가이드북(주요 20작물).pdf - 38개 청크
📤 38개 청크를 MilvusDB에 업로드 중...
  - 배치 1: 1 ~ 38
    > ✅ 삽입 성공: 38개
⌛ flush 대기...
📊 num_entities: 38
✅ 문서 업로드 완료: 38개 문서
✅ 파일 처리 및 업로드 완료: ./farmer/disaster/data/2025 기상재해 대응기술 가이드북(주요 20작물).pdf
✅ 성공: 2025 기상재해 대응기술 가이드북(주요 20작물).pdf
📊 디렉토리 처리 완료: 1/1개 파일 성공

📊 처리 결과:
  2025 기상재해 대응기술 가이드북(주요 20작물).pdf: ✅ 성공

전체 결과: 1/1개 파일 성공
```

## ⚠️ 주의사항

### 1. **MilvusDB 서버 실행 필요**
- MilvusDB 서버가 `localhost:19530`에서 실행 중이어야 합니다
- Docker로 실행하는 경우: `docker run -p 19530:19530 milvusdb/milvus:latest`

### 2. **메모리 사용량**
- 대용량 PDF 파일 처리 시 메모리 사용량이 증가할 수 있습니다
- 배치 크기(100)를 조정하여 메모리 사용량을 제어할 수 있습니다

### 3. **네트워크 연결**
- HuggingFace 모델 다운로드를 위해 인터넷 연결이 필요합니다
- 첫 실행 시 모델 다운로드로 인해 시간이 소요될 수 있습니다

## 🔍 문제 해결

### 일반적인 오류
1. **Milvus 연결 실패**: MilvusDB 서버가 실행 중인지 확인
2. **임베딩 모델 로드 실패**: 인터넷 연결 및 HuggingFace 토큰 확인
3. **Docling 처리 실패**: PDF 파일이 손상되지 않았는지 확인

### 성능 최적화
- `batch_size` 조정으로 메모리 사용량 제어
- `num_threads` 설정으로 CPU 사용량 조절
- 작은 청크 병합으로 검색 품질 향상

## 📚 참고 자료

- [Docling 공식 문서](https://docling-project.github.io/docling/)
- [Docling Hybrid Chunking 예제](https://docling-project.github.io/docling/examples/hybrid_chunking/)
- [MilvusDB 공식 문서](https://milvus.io/docs)
- [LangChain HuggingFace 임베딩](https://python.langchain.com/docs/integrations/text_embedding/huggingface)

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
