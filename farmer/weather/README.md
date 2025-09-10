# Weather Agent 모듈

모듈화된 날씨 에이전트 시스템으로, 기상특보, 단기예보, 중기예보를 통합하여 제공합니다.

## 📁 파일 구조

```
farmer/weather/
├── __init__.py              # 패키지 초기화
├── run_weather_agent.py     # 메인 실행 파일 (권장)
├── advisory_node.py         # 기상특보 노드
├── short_forecast_node.py   # 단기예보 노드
├── mid_forecast_node.py     # 중기예보 노드
├── utils.py                 # 공통 유틸리티 함수
├── WeatherAgent.py          # 원본 파일 (수정 금지)
└── WeatherAgent_New.py      # 이전 버전 (상대 import 문제)
```

## 🚀 사용법

### 1. 기본 실행 (권장)

```python
from farmer.weather import run_weather_agent

# 질문 실행
result = run_weather_agent("오늘 서울 날씨 어때?")
print(result)
```

### 2. 개별 노드 사용

```python
from farmer.weather import AdvisoryNode, ShortForecastNode, MidForecastNode

# 기상특보
advisory = AdvisoryNode()
advisory_data = advisory.run({"question": "서울 기상특보"})

# 단기예보
short_forecast = ShortForecastNode()
short_data = short_forecast.run({"question": "서울 내일 날씨"})

# 중기예보
mid_forecast = MidForecastNode()
mid_data = mid_forecast.run({"question": "서울 이번주 날씨"})
```

### 3. 유틸리티 함수 사용

```python
from farmer.weather import combine_weather_data, search_similar_documents

# 날씨 데이터 통합
combined = combine_weather_data(advisory_data, short_data, mid_data)

# 유사도 검색
similar_docs = search_similar_documents("비 올까?", documents)
```

## ⚙️ 환경 설정

### 필수 환경 변수 (.env)

```env
# 기상청 API
WHEATHER_API_KEY_HUB=your_kma_api_key

# OpenAI API
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
TEMPERATURE=0.2

# 임베딩 모델 (선택)
EMBED_MODEL_NAME=BAAI/bge-m3

# API 타임아웃 (선택)
KMA_TIMEOUT=30

# 지역 매핑 CSV 파일 경로 (선택)
REGION_CSV_PATH=farmer/all_regions_combined.csv
```

### 필수 파일

- `farmer/all_regions_combined.csv` - 지역 코드 매핑 파일
  - 컬럼: `REG_ID` (지역코드), `REG_NAME` (지역명)

## 🎯 주요 기능

### 1. Human-in-the-Loop 지원
- 사용자와의 대화형 상호작용
- 추가 질문 및 정보 요청 처리
- 3회 재시도 후 fallback 처리

### 2. 조건부 실행
- LLM이 질문을 분석하여 필요한 노드만 실행
- 단일/병렬 실행 자동 결정
- 메모리 효율적인 처리

### 3. 자동 지역 처리
- 지역 정보가 없는 질문에 대해 서울/수도권 자동 처리
- 추가 지역 선택 옵션 제공

### 4. 메모리 최적화
- 임베딩 모델 지연 로딩
- 공통 함수 통합으로 중복 코드 제거

## 📊 워크플로우

```
사용자 질문 → LLM 분석 → 조건부 노드 실행 → 데이터 통합 → 답변 생성
     ↓
Human-in-the-Loop (추가 질문/정보 요청)
     ↓
Fallback 처리 (3회 재시도 후)
```

## 🔧 개발자 정보

### 모듈화 원칙
- 각 노드는 독립적으로 실행 가능
- 공통 기능은 `utils.py`에 통합
- 원본 `WeatherAgent.py`는 수정 금지

### 성능 최적화
- 임베딩 모델 지연 로딩으로 메모리 절약
- CSV 기반 지역 매핑으로 유연성 확보
- 병렬 실행으로 처리 속도 향상

## 🐛 문제 해결

### ImportError 해결
```python
# 절대 import 사용 (run_weather_agent.py 권장)
from farmer.weather import run_weather_agent
```

### CSV 파일 오류
- `farmer/all_regions_combined.csv` 파일 존재 확인
- 컬럼명이 `REG_ID`, `REG_NAME`인지 확인

### API 키 오류
- `.env` 파일에 올바른 API 키 설정 확인
- 기상청 API 키 유효성 확인

## 📝 예제

### 기본 질문
```python
# 지역 정보 포함
result = run_weather_agent("서울 내일 날씨 어때?")

# 지역 정보 없음 (자동 서울 처리)
result = run_weather_agent("오늘 날씨 어때?")

# 복합 질문
result = run_weather_agent("이번주 서울 날씨와 기상특보 알려줘")
```

### Human-in-the-Loop 예제
```python
# 첫 번째 질문
result = run_weather_agent("오늘 날씨 어때?")
# → "서울 및 수도권 오늘 날씨를 알려드리겠습니다..."

# 추가 질문 (자동으로 Human-in-the-Loop 활성화)
result = run_weather_agent("부산은 어때?")
# → "부산 오늘 날씨를 알려드리겠습니다..."
```

## 🔄 업데이트 내역

- **v1.0**: 기본 모듈화 완료
- **v1.1**: Human-in-the-Loop 추가
- **v1.2**: 메모리 최적화 및 공통 함수 통합
- **v1.3**: 조건부 실행 및 자동 지역 처리