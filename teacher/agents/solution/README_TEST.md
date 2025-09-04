# Solution Agent HITL 테스트 가이드

이 문서는 `solution_agent_hitl.py`의 풀이 평가 및 개선 워크플로우를 테스트하는 방법을 설명합니다.

## 🧪 테스트 개요

이 테스트는 다음 워크플로우를 검증합니다:

1. **풀이 평가** - LLM을 통해 풀이 품질을 0-100점으로 평가
2. **사용자 피드백 수집** - 개선이 필요한 경우 사용자 피드백 요청
3. **풀이 개선** - 피드백에 따라 풀이를 개선
4. **추가 정보 검색** - 용어 설명 등이 필요한 경우 관련 정보 검색
5. **최종 풀이 정리** - 개선된 풀이와 검색 결과를 통합

## 🚀 테스트 실행 방법

### 방법 1: 직접 실행
```bash
cd teacher/agents/solution
python test_solution_agent_hitl.py
```

### 방법 2: 실행 스크립트 사용
```bash
cd teacher/agents/solution
python run_test.py
```

## 📝 테스트 문제

테스트는 다음 문제를 사용합니다:

**문제**: 소프트웨어 설계에서 사용되는 대표적인 추상화 기법이 아닌 것은?

**보기**:
1. 자료 추상화
2. 제어 추상화
3. 과정 추상화
4. 강도 추상화

**정답**: 4번 (강도 추상화)

## 🧪 테스트 모드 설정

테스트는 강제로 개선 모드로 설정되어 실행됩니다:

```python
# 테스트 상태에서 설정되는 값들
test_mode = True                    # 테스트 모드 활성화
test_score = 35                     # 강제 점수 (낮은 점수)
test_feedback_type = "term_explanation"  # 강제 피드백 타입
```

## 📊 테스트 시나리오

### 시나리오 1: 용어 설명 필요
- **설정**: `test_feedback_type = "term_explanation"`
- **예상 결과**: "강도 추상화"라는 용어에 대한 설명이 추가된 개선된 풀이

### 시나리오 2: 쉬운 설명 필요
- **설정**: `test_feedback_type = "easier_explanation"`
- **예상 결과**: 더 이해하기 쉬운 단계별 설명이 포함된 개선된 풀이

### 시나리오 3: 용어 설명 + 쉬운 설명
- **설정**: `test_feedback_type = "term_easier_explanation"`
- **예상 결과**: 용어 설명과 쉬운 설명이 모두 포함된 개선된 풀이

## 🔧 테스트 커스터마이징

테스트를 다른 시나리오로 실행하려면 `create_test_state()` 함수를 수정하세요:

```python
def create_test_state() -> SolutionState:
    # ... 기존 코드 ...
    
    # 테스트 모드 설정 변경
    test_mode = True
    test_score = 25                    # 더 낮은 점수
    test_feedback_type = "easier_explanation"  # 다른 피드백 타입
    
    # ... 기존 코드 ...
```

## 📋 테스트 결과 확인

테스트 실행 후 다음 정보를 확인할 수 있습니다:

- **원본 풀이**: 테스트용으로 생성된 기본 풀이
- **평가 점수**: LLM이 평가한 풀이 품질 점수
- **개선 필요 여부**: 풀이 개선이 필요한지 판단
- **개선된 풀이**: 사용자 피드백을 바탕으로 개선된 풀이
- **검색 결과**: 관련 개념이나 용어에 대한 추가 정보
- **최종 풀이**: 모든 개선사항이 반영된 최종 풀이

## ⚠️ 주의사항

1. **LLM API 키**: 테스트 실행을 위해서는 유효한 LLM API 키가 필요합니다
2. **인터넷 연결**: LLM API 호출을 위해 인터넷 연결이 필요합니다
3. **의존성**: 필요한 Python 패키지들이 설치되어 있어야 합니다

## 🐛 문제 해결

### 오류: 모듈을 찾을 수 없음
```bash
# 프로젝트 루트에서 실행하거나 PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project"
```

### 오류: API 키가 유효하지 않음
```bash
# 환경변수 설정 확인
echo $GROQAI_API_KEY
echo $OPENAI_BASE_URL
```

### 오류: 의존성 패키지 누락
```bash
# 필요한 패키지 설치
pip install langchain openai langgraph
```

## 📞 지원

테스트 실행 중 문제가 발생하면 다음을 확인하세요:

1. Python 버전 (3.8+ 권장)
2. 필요한 패키지 설치 여부
3. 환경변수 설정
4. 프로젝트 경로 설정

---

**테스트 목적**: 풀이 평가 및 개선 워크플로우의 정상 동작을 검증하여 Human-in-the-Loop 시스템의 품질을 보장합니다.
