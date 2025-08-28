# HITL (Human-in-the-Loop) 피드백 시스템

## 개요

이 시스템은 사용자와의 상호작용을 통해 문제 풀이를 지속적으로 개선하는 Human-in-the-Loop 시스템입니다. **LangGraph의 `interrupt`와 `Command`를 활용**하여 실행을 일시 중단하고 사용자 피드백을 받아 실행을 재개할 수 있습니다.

## 주요 기능

### 1. **Interrupt 기반 실행 제어**
- `interrupt()` 함수를 사용하여 실행을 일시 중단
- `Command` 객체를 통해 사용자 입력을 받아 실행 재개
- 체크포인터를 통한 상태 지속성 유지

### 2. **자동 피드백 분류**
- LLM이 사용자 입력을 분석하여 의도를 자동으로 분류
- 3가지 카테고리로 정확한 분류 수행

### 3. **3가지 피드백 카테고리**

#### 📚 **이해됨** (Comprehension)
- **설명**: 사용자가 풀이를 이해했고 만족하는 경우
- **예시**: "이해가 됩니다", "좋습니다", "만족합니다", "충분합니다", "괜찮습니다"
- **처리**: 바로 풀이 저장하여 원래 흐름대로 진행

#### 🔄 **더 쉬운 풀이 필요** (Improvement)
- **설명**: 사용자가 풀이가 너무 복잡하거나 어렵다고 느끼는 경우
- **예시**: "더 쉽게 설명해주세요", "복잡해요", "어려워요", "간단하게", "초보자도 이해할 수 있게"
- **처리**: 풀이를 다시 생성 (더 쉬운 버전으로)

#### 🔍 **용어 설명 필요** (Clarification)
- **설명**: 사용자가 특정 용어나 개념에 대한 추가 설명을 요청하는 경우
- **예시**: "이 용어가 뭔지 모르겠어요", "설명이 부족해요", "용어를 더 자세히", "개념 설명 추가"
- **처리**: 검색 노드 실행하여 풀이에 추가 정보 보강
  - **검색 쿼리에 유저 피드백 포함**: 문제 + 풀이 + 사용자 피드백을 모두 포함하여 정확한 검색
  - **retrieve_agent.invoke() 사용**: 올바른 메서드 호출로 안정적인 검색 수행

## 시스템 구조

```
사용자 입력 → LLM 분석 → 분류 결정 → 적절한 처리
    ↓
[이해됨] → 바로 저장
    ↓
[더 쉬운 풀이] → 풀이 개선
    ↓
[용어 설명] → 검색 + 풀이 보강
```

## Interrupt 기반 HITL 워크플로우

### 1. **실행 시작**
```python
agent = SolutionAgent(hitl_mode="manual")
final_state = agent.invoke(
    user_input_txt="질문",
    user_problem="문제",
    user_problem_options=["보기1", "보기2", "보기3", "보기4"]
)
```

### 2. **실행 일시 중단**
- `interrupt()` 함수가 호출되면 실행이 일시 중단
- 체크포인터에 현재 상태가 저장됨
- 사용자 입력 대기

### 3. **실행 재개**
```python
from langgraph.types import Command

# Command 객체 생성
command = Command(resume={"data": "사용자 피드백"})

# 실행 재개
final_state = agent.invoke(
    user_input_txt="질문",
    user_problem="문제", 
    user_problem_options=["보기1", "보기2", "보기3", "보기4"],
    command=command  # Command 객체 전달
)
```

## 사용법

### 1. **기본 사용**
```python
from solution_agent_hitl import SolutionAgent

# HITL 모드로 에이전트 생성
agent = SolutionAgent(max_interactions=5, hitl_mode="manual")

# 문제 해결 요청
result = agent.invoke(
    user_input_txt="프로세스와 스레드의 차이점을 이해하고 싶습니다.",
    user_problem="프로세스와 스레드의 차이점으로 올바른 것은?",
    user_problem_options=["보기1", "보기2", "보기3", "보기4"]
)
```

### 2. **HITL 모드 선택**
- **`auto`**: 자동 모드 (HITL 없음)
- **`smart`**: 스마트 모드 (품질에 따라 자동 결정)
- **`manual`**: 수동 모드 (항상 HITL 적용)

### 3. **테스트 실행**
```bash
# 기본 HITL 테스트
python solution_agent_hitl.py

# Interrupt 기반 HITL 테스트
python test_interrupt_hitl.py

# 기존 피드백 분류 테스트
python test_hitl_feedback.py
```

## 워크플로우

### 1단계: 문제 분석
- 유사 문제 검색
- 해답 및 풀이 생성

### 2단계: 풀이 검증
- 자동 품질 평가
- HITL 적용 여부 결정

### 3단계: 사용자 피드백 수집
- **`interrupt()` 호출로 실행 일시 중단**
- 현재 풀이 상태 표시
- 사용자 의견 입력 대기

### 4단계: LLM 의도 분석
- 3가지 카테고리로 자동 분류
- 적절한 처리 방향 결정

### 5단계: 피드백별 처리
- **이해됨**: 바로 저장
- **더 쉬운 풀이**: 풀이 개선
- **용어 설명**: 검색 + 보강

### 6단계: 결과 저장
- 개선된 풀이를 벡터 DB에 저장
- 상호작용 히스토리 기록

## 품질 평가 시스템

### 다차원 품질 점수 (0-100점)
- **정확성** (30%): 정답과 풀이 과정의 정확성
- **완성도** (25%): 핵심 개념 포함 및 단계별 설명
- **이해도** (25%): 문장 명확성 및 전문 용어 설명
- **논리성** (20%): 논리적 추론 및 인과관계

### 스마트 HITL 결정
- **80점 이상**: 자동 통과
- **60-79점**: HITL 적용
- **60점 미만**: HITL 필수 적용

## 설정 및 환경변수

```bash
# OpenAI API 설정
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_LLM_MODEL=moonshotai/kimi-k2-instruct

# LLM 설정
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2048
```

## 장점

1. **실행 제어**: `interrupt`를 통한 정확한 실행 일시 중단
2. **상태 지속성**: 체크포인터를 통한 상태 보존 및 재개
3. **자동화된 의도 파악**: LLM이 사용자 피드백을 자동으로 분석
4. **맞춤형 개선**: 피드백 유형에 따른 적절한 처리
5. **지속적 품질 향상**: 사용자 피드백을 통한 반복 개선
6. **자연스러운 상호작용**: 자연어로 자유롭게 피드백 제공
7. **정확한 검색**: 유저 피드백이 포함된 검색 쿼리로 더 정확한 정보 검색

## 테스트 시나리오

### 시나리오 1: 이해됨
```
사용자: "이해가 됩니다. 만족합니다."
→ LLM 분석: "이해됨"
→ 처리: 바로 저장으로 진행
```

### 시나리오 2: 더 쉬운 풀이 필요
```
사용자: "풀이를 더 쉽게 설명해주세요."
→ LLM 분석: "더 쉬운 풀이 필요"
→ 처리: 풀이 개선 (간단한 용어, 단계별 설명, 예시 추가)
```

### 시나리오 3: 용어 설명 필요
```
사용자: "프로세스라는 용어가 뭔지 모르겠어요."
→ LLM 분석: "용어 설명 필요"
→ 처리: 검색 노드 실행 → 풀이에 용어 정의 및 예시 추가
  - 검색 쿼리: "프로세스와 스레드의 차이점으로 올바른 것은? [풀이 내용] 프로세스라는 용어가 뭔지 모르겠어요"
```

## Interrupt 사용 예시

### 1. **실행 중단**
```python
# _collect_user_feedback 함수 내부
feedback_query = {
    "query": "풀이에 대한 의견을 자유롭게 입력해주세요",
    "examples": {...},
    "current_problem": state['user_problem'],
    "current_answer": state['generated_answer'],
    "current_explanation": state['generated_explanation']
}

# interrupt 호출로 실행 일시 중단
human_response = interrupt(feedback_query)
```

### 2. **실행 재개**
```python
from langgraph.types import Command

# Command 객체 생성
command = Command(resume={"data": "사용자 피드백"})

# 실행 재개
final_state = agent.invoke(
    user_input_txt=user_input_txt,
    user_problem=user_problem,
    user_problem_options=user_problem_options,
    command=command
)
```

## 주의사항

1. **API 키 설정**: OpenAI API 키가 필요합니다
2. **네트워크 연결**: LLM 호출을 위해 인터넷 연결이 필요합니다
3. **Interrupt 처리**: `interrupt()` 호출 시 적절한 예외 처리가 필요합니다
4. **Command 객체**: 실행 재개 시 올바른 형식의 Command 객체를 전달해야 합니다
5. **체크포인터**: 상태 지속성을 위해 체크포인터가 올바르게 설정되어야 합니다
6. **검색 에이전트**: retrieve_agent가 올바르게 설정되어야 합니다

## 향후 개선 계획

1. **다국어 지원**: 한국어 외 다양한 언어 지원
2. **감정 분석**: 사용자 피드백의 감정적 측면 분석
3. **학습 기능**: 사용자 패턴 학습을 통한 개인화된 풀이 제공
4. **시각화**: 풀이 개선 과정의 시각적 표현
5. **배치 처리**: 여러 문제에 대한 일괄 HITL 처리
6. **웹 인터페이스**: 브라우저 기반의 HITL 인터페이스
7. **실시간 협업**: 여러 사용자가 동시에 피드백을 제공할 수 있는 시스템
