# ET_Agent - AI 기반 농업 및 교육 에이전트 시스템

## 📋 프로젝트 개요

ET_Agent는 LangGraph 기반의 AI 에이전트 시스템으로, 농업(재배, 재해, 판매)과 교육(IT 시험) 분야의 전문적인 질문-답변 서비스를 제공합니다. RAGAS(RAG Assessment) 프레임워크를 활용하여 답변 품질을 자동으로 평가하고 개선합니다.

## 🏗️ 시스템 아키텍처

```
ET_Agent/
├── farmer/                    # 농업 관련 에이전트
│   ├── cultivation/          # 재배 에이전트
│   ├── disaster/             # 재해 대응 에이전트
│   ├── sales/                # 판매 에이전트
│   └── weather/              # 날씨 에이전트
├── teacher/                   # 교육 관련 에이전트
│   ├── agents/               # 다양한 교육 에이전트
│   ├── exam/                 # 시험 문제 생성
│   └── golden/               # 골든 데이터셋
├── common/                    # 공통 모듈
│   ├── milvus_manager.py     # MilvusDB 관리
│   └── milvus_helpers.py     # MilvusDB 헬퍼
├── api/                       # FastAPI 백엔드
├── auth/                      # 인증 시스템
└── supervisor.py              # 메인 오케스트레이터
```

## 🚀 주요 기능

### 1. 농업 에이전트 (Farmer)
- **재배 에이전트**: 작물 재배 방법, 병해충 관리, 수확 시기 등
- **재해 대응 에이전트**: 자연재해 대응, 피해 복구, 보상 신청 등
- **판매 에이전트**: 농산물 시세 조회, 판매처 정보, 마케팅 전략 등
- **날씨 에이전트**: 농업 관련 날씨 정보, 기상 예보 등

### 2. 교육 에이전트 (Teacher)
- **시험 문제 생성**: IT 분야 모의고사 문제 자동 생성
- **문제 풀이**: 사용자 질문에 대한 문제 해설 제공
- **용어 해설**: 위키 기반 기술 용어 설명
- **채점 및 분석**: 오답 분석 및 취약점 파악

### 3. RAGAS 평가 시스템
- **자동 품질 평가**: 4가지 메트릭으로 답변 품질 측정
- **CSV 결과 저장**: 평가 결과를 Excel에서 분석 가능한 형태로 저장
- **MilvusDB 통합**: 벡터 검색을 통한 정확한 컨텍스트 제공

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| **AI Framework** | LangGraph, LangChain, OpenAI API |
| **RAG System** | RAGAS, MilvusDB, HuggingFace Embeddings |
| **Backend** | FastAPI, Python 3.9+ |
| **Database** | MilvusDB, Redis, PostgreSQL |
| **Frontend** | React, React Router, Redux |
| **DevOps** | Docker, Docker Compose |
| **Evaluation** | RAGAS, Pandas, CSV Export |

## 📦 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd ET_Agent

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# MilvusDB
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_URI=http://localhost:19530

# Redis
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_DB=0

# Tavily (웹 검색)
TAVILY_API_KEY=your_tavily_api_key

# KAMIS API (농산물 시세)
KAMIS_API_KEY=your_kamis_api_key
KAMIS_ID=your_kamis_id
```

### 3. Docker Compose 실행

```bash
# 전체 시스템 실행
docker-compose up -d

# 특정 서비스만 실행
docker-compose up -d milvus redis
```

### 4. 개별 실행

```bash
# 메인 오케스트레이터 실행
python main.py

# 특정 에이전트 테스트
python farmer/sales/SalesRAGAS.py
python farmer/disaster/DisasterRAGAS_New.py
```

## 📊 RAGAS 평가 시스템

### 평가 메트릭

1. **Context Precision**: 검색된 컨텍스트의 정확성
2. **Faithfulness**: 답변이 컨텍스트에 충실한지
3. **Answer Relevancy**: 답변이 질문과 관련성이 있는지
4. **Context Recall**: 필요한 정보를 얼마나 잘 찾았는지

### 사용법

```python
# SalesRAGAS 실행
from farmer.sales.SalesRAGAS import SalesRAGASEvaluator

evaluator = SalesRAGASEvaluator()
results = evaluator.run_full_evaluation()
evaluator.save_results()  # CSV 파일로 저장

# DisasterRAGAS 실행
from farmer.disaster.DisasterRAGAS_New import DisasterRAGASEvaluator

evaluator = DisasterRAGASEvaluator()
results = evaluator.run_full_evaluation()
evaluator.save_results()  # CSV 파일로 저장
```

### 결과 파일

평가 결과는 다음 위치에 CSV 파일로 저장됩니다:
- **SalesRAGAS**: `farmer/sales/data/sales_ragas_evaluation_results_YYYYMMDD_HHMM.csv`
- **DisasterRAGAS**: `farmer/disaster/data/disaster_ragas_evaluation_results_YYYYMMDD_HHMM.csv`

## 🔧 주요 모듈 설명

### 1. MilvusDB 관리 (`common/milvus_manager.py`)

```python
from common.milvus_manager import MilvusDBManager

# MilvusDB 연결
manager = MilvusDBManager()
manager.connect()

# 문서 검색
documents = manager.search_documents_by_collection(
    collection_name="market_price_docs",
    query="농산물 시세",
    k=5
)
```

### 2. 에이전트 실행

```python
# SalesAgent 실행
from farmer.sales.SalesAgent import run

result = run({
    "query": "대구 감자 시세 알려줘",
    "milvus_data": milvus_connection_info
})

# DisasterAgent 실행
from farmer.disaster.DisasterAgent_LLM import run

result = run({
    "query": "홍수 피해 대응 방법",
    "milvus_data": milvus_connection_info
})
```

### 3. RAGAS 평가

```python
# 개별 질문 평가
evaluator = DisasterRAGASEvaluator()
result = evaluator.evaluate_single_question({
    "question": "질문",
    "reference": "참조 답변",
    "contexts": ["컨텍스트1", "컨텍스트2"]
})

# 전체 평가
results = evaluator.run_full_evaluation()
```

## 📁 데이터 구조

### 입력 데이터 (CSV)
```csv
question,ground_truth,contexts
"농산물 시세는?", "시세 정보입니다", "['컨텍스트1', '컨텍스트2']"
```

### 출력 데이터 (CSV)
```csv
question_id,question,reference,answer,context,timestamp,context_precision,faithfulness,answer_relevancy,context_recall
1,"질문","참조","답변","컨텍스트","2025-01-15T14:30:00",0.85,0.92,0.88,0.79
AVERAGE,"전체 평균 점수","","","","2025-01-15T14:30:02",0.82,0.91,0.90,0.81
```

## 🐛 문제 해결

### 1. MilvusDB 연결 오류
```bash
# MilvusDB 상태 확인
docker-compose ps milvus

# 로그 확인
docker-compose logs milvus
```

### 2. RAGAS 평가 오류
- **Answer Relevancy 0점**: SingleTurnSample에서 reference 파라미터 제거
- **컨텍스트 부족**: MilvusDB 연결 상태 및 컬렉션 확인
- **메모리 부족**: 배치 크기 조정 (`batch_size` 파라미터)

### 3. API 키 오류
```bash
# 환경 변수 확인
echo $OPENAI_API_KEY
echo $TAVILY_API_KEY
```

## 📈 성능 최적화

### 1. MilvusDB 최적화
- 인덱스 생성: `IVF_FLAT` 또는 `HNSW` 사용
- 검색 파라미터: `nprobe` 값 조정
- 문서 수 제한: `k=3` (기본값)

### 2. RAGAS 평가 최적화
- 배치 처리: 여러 질문을 한 번에 평가
- 캐싱: 동일한 질문에 대한 중복 평가 방지
- 병렬 처리: 4개 메트릭을 순차적으로 실행

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.

---

**ET_Agent** - AI로 농업과 교육의 미래를 열어갑니다 🌱📚
