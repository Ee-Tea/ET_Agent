# Ee-Tea
# 이장 선생님

**Ee-Tea는 귀농인과 수험생을 위한 AI 챗봇입니다.**
  귀농을 하여 농작물을 처음 키우시는 분들, 현재 농업에 종사하시는 분들은 작물을 키울 때 필요한 모든 정보를 얻을 수 있으며
  자격증 시험을 준비하는 수험생들은 모의고사 생성, 모르는 문제에 대한 풀이 제공, 오답노트 생성 및 분석을 할 수 있습니다.
  필요한 정보를 간단하게 AI와의 채팅으로 얻어보세요.

---

## 🧩 주요 기능

### 사용자 인증
- 회원가입 / 로그인

### 작물 추천 및 생육 정보 제공
- 지역 및 기후 기반 작물 추천
- 원하는 작물에 대한 생육 정보 제공

### 재배 중 관리 방법 제공
- 재배 중 필요한 관리 방법(비료 농약 등)에 대한 정보 제공
- 재해 대비 방법 제공

### 수확 단계 도우미
- 작물의 수확 적합도와 방법 제공
- 지역별 판매처 추천



---

## 🏗️ 기술 스택

| 분류 | 기술 |
|------|------|
| AI Agent | LangGraph, UV, Python, OpenAI API |
| Backend | Spring, FastAPI |
| Frontend | React, React Router, Axios, Redux |
| DB | Milvus, Redis, PostgreSQL |
| Dev/Ops | Docker, AWS, Kubernetes |


---

##  코딩 컨벤션 (Coding Convention)

| 항목 | 규칙 |
|------|------|
| **들여쓰기** | 2칸 (space 2개) |
| **따옴표** | `'싱글쿼트'` 사용 |
| **세미콜론** | 항상 줄 끝에 `;` 붙이기 |
| **변수 이름** | `camelCase` 사용 (예: `userName`, `isLoggedIn`) |
| **컴포넌트 이름** | `PascalCase` 사용 (예: `ReviewCard`, `UserProfile`) |
| **파일 이름** | 소문자 또는 카멜케이스 사용 (예: `review-card.jsx`, `userProfile.jsx`) |

---


> 예시: `feat: 리뷰 작성 기능 추가`

---

### 🔤 커밋 타입 종류

| 타입 | 의미 | 예시 |
|------|------|------|
| `feat:` | 기능 추가 | `feat: 별점 등록 기능 구현` |
| `fix:` | 버그 수정 | `fix: 로그인 시 토큰 오류 해결` |
| `style:` | 코드 스타일 수정 (기능 변화 없음) | `style: 세미콜론 누락 수정` |
| `refactor:` | 리팩토링 (기능 변화 없음) | `refactor: 리뷰 모듈 구조 개선` |
| `docs:` | 문서 수정 | `docs: README에 코딩 컨벤션 추가` |
| `test:` | 테스트 코드 추가/수정 | `test: 리뷰 서비스 테스트 추가` |
| `chore:` | 설정, 패키지 관리 등 잡일 | `chore: ESLint 설정 추가` |
| `build:` | 빌드 시스템 변경 | `build: Vite 설정 변경` |
| `ci:` | CI 설정 변경 | `ci: GitHub Actions 워크플로우 추가` |

---

### ✅ 커밋 메시지 작성 예시

```bash
feat: 유튜브 링크 등록 기능 추가
fix: 리뷰 수정 시 페이지 새로고침 문제 해결
chore: Prettier 설치 및 설정 파일 추가
docs: 커밋 메시지 규칙 문서화
