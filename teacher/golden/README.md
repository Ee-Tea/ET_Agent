# 골든 데이터셋 생성 및 RAG 파이프라인 평가

이 디렉토리는 Ragas를 활용하여 teacher RAG 시스템의 성능을 평가하기 위한 골든 데이터셋을 생성하고 평가하는 도구들을 포함합니다.

## 📁 파일 구조

```
teacher/golden/
├── README.md                           # 이 파일
├── generate_golden_dataset.py          # 골든 데이터셋 생성 스크립트
├── evaluate_rag_pipeline.py            # RAG 파이프라인 평가 스크립트
├── run_evaluation.py                   # 통합 실행 스크립트
├── datasets/                           # 생성된 데이터셋 저장 폴더
│   └── testset_YYYYMMDD_HHMMSS.json   # 생성된 테스트셋 파일들
└── evaluation_results/                 # 평가 결과 저장 폴더
    ├── evaluation_detailed_YYYYMMDD_HHMMSS.csv    # 상세 평가 결과
    ├── evaluation_summary_YYYYMMDD_HHMMSS.json    # 요약 통계
    └── evaluation_raw_YYYYMMDD_HHMMSS.json        # 원본 평가 데이터
```

## 🚀 사용법

### 1. 전체 프로세스 실행 (권장)

```bash
# 기본 설정으로 전체 프로세스 실행
uv run teacher/golden/run_evaluation.py

# 커스텀 설정으로 실행
uv run teacher/golden/run_evaluation.py --mode full --testset-size 50 --llm-model gpt-4o-mini
```

### 2. 단계별 실행

#### 데이터셋 생성만
```bash
uv run teacher/golden/run_evaluation.py --mode dataset --testset-size 30
```

#### 평가만 (기존 데이터셋 사용)
```bash
uv run teacher/golden/run_evaluation.py --mode evaluation
```

### 3. 개별 스크립트 실행

#### 골든 데이터셋 생성
```bash
uv run teacher/golden/generate_golden_dataset.py
```

#### RAG 파이프라인 평가
```bash
uv run teacher/golden/evaluate_rag_pipeline.py
```

## 📊 평가 메트릭

Ragas를 사용하여 다음 메트릭들로 RAG 시스템의 성능을 평가합니다:

### 핵심 메트릭

1. **Faithfulness (신뢰성)**
   - 응답이 제공된 컨텍스트에 사실적으로 부합하는가?
   - 임계값: ≥ 0.90

2. **Answer Relevancy (답변 관련성)**
   - 응답이 질문에 얼마나 관련적인가?
   - 임계값: ≥ 0.85

3. **Context Precision (컨텍스트 정밀도)**
   - 리트리버가 관련 컨텍스트를 상위에 올렸는가?
   - 임계값: ≥ 0.70

4. **Context Recall (컨텍스트 재현율)**
   - 필요한 정보를 모두 가져왔는가?
   - 임계값: ≥ 0.70

### 성능 등급

- **A+ (우수)**: 통과율 ≥ 90%
- **A (양호)**: 통과율 ≥ 80%
- **B (보통)**: 통과율 ≥ 70%
- **C (미흡)**: 통과율 ≥ 60%
- **D (부족)**: 통과율 < 60%

## 🔧 설정 옵션

### 명령행 인수

- `--mode`: 실행 모드 (`full`, `dataset`, `evaluation`)
- `--testset-size`: 생성할 테스트셋 크기 (기본값: 30)
- `--llm-model`: 사용할 LLM 모델 (기본값: gpt-4o-mini)
- `--embedding-model`: 사용할 임베딩 모델 (기본값: text-embedding-3-small)

### 환경 변수

다음 환경 변수들이 설정되어 있어야 합니다:

```bash
OPENAI_API_KEY=your_openai_api_key
```

## 📈 결과 해석

### 출력 파일

1. **evaluation_detailed_*.csv**: 각 샘플별 상세 평가 점수
2. **evaluation_summary_*.json**: 전체 통계 및 요약 정보
3. **evaluation_raw_*.json**: 원본 평가 데이터 (질문, 답변, 컨텍스트 등)

### 주요 지표

- **전체 통과율**: 모든 메트릭이 임계값을 통과한 샘플의 비율
- **메트릭별 평균 점수**: 각 메트릭의 평균 성능
- **메트릭별 통과율**: 각 메트릭별로 임계값을 통과한 샘플의 비율

## 🛠️ 커스터마이징

### 임계값 조정

`evaluate_rag_pipeline.py`의 `RAGPipelineEvaluator` 클래스에서 임계값을 조정할 수 있습니다:

```python
self.thresholds = {
    'faithfulness': 0.90,      # 신뢰성 임계값
    'answer_relevancy': 0.85,  # 답변 관련성 임계값
    'context_precision': 0.70, # 컨텍스트 정밀도 임계값
    'context_recall': 0.70     # 컨텍스트 재현율 임계값
}
```

### 데이터 소스 추가

`generate_golden_dataset.py`의 `load_exam_documents()` 메서드에서 추가 데이터 소스를 로드할 수 있습니다.

## ⚠️ 주의사항

1. **API 비용**: OpenAI API 사용으로 인한 비용이 발생할 수 있습니다.
2. **실행 시간**: 테스트셋 크기에 따라 실행 시간이 길어질 수 있습니다.
3. **메모리 사용량**: 큰 데이터셋의 경우 충분한 메모리가 필요합니다.

## 🔍 문제 해결

### 일반적인 오류

1. **API 키 오류**: `OPENAI_API_KEY` 환경 변수가 설정되어 있는지 확인
2. **메모리 부족**: 테스트셋 크기를 줄여서 실행
3. **네트워크 오류**: 인터넷 연결 상태 확인

### 로그 확인

실행 중 상세한 로그가 출력되므로, 오류 발생 시 로그를 확인하여 문제를 진단할 수 있습니다.

## 📚 참고 자료

- [Ragas 공식 문서](https://docs.ragas.io/)
- [LangChain 문서](https://python.langchain.com/)
- [OpenAI API 문서](https://platform.openai.com/docs)
