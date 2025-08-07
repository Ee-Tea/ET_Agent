# Ee-Tea
### 시험문제 생성
- 모의고사 1회 전체 문제(100문제)
- 맞춤형 문제 생성 가능
- 사용자 희망 문제 생성 가능

### 시험문제 풀이
- 사용자 질문을 통한 문제 풀이 및 해설 제공

### 용어 해설
- 위키 기반 검색을 통한 용어 해설 기능

### 채점 및 오답 분석
- 사용자 취약 유형 분석을 통한 보완점 제공
- 틀린 문제에 대한 오답 노트 생성


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
| `feat:` | 기능 추가 | `feat: 문제 생성 기능 구현` |
| `fix:` | 버그 수정 | `fix: 로그인 시 토큰 오류 해결` |
| `style:` | 코드 스타일 수정 (기능 변화 없음) | `style: 세미콜론 누락 수정` |
| `refactor:` | 리팩토링 (기능 변화 없음) | `refactor: 리뷰 모듈 구조 개선` |
| `docs:` | 문서 수정 | `docs: README에 코딩 컨벤션 추가` |
| `test:` | 테스트 코드 추가/수정 | `test: API 테스트 코드 수정` |
| `chore:` | 설정, 패키지 관리 등 잡일 | `chore: ESLint 설정 추가` |
| `build:` | 빌드 시스템 변경 | `build: Vite 설정 변경` |
| `ci:` | CI 설정 변경 | `ci: GitHub Actions 워크플로우 추가` |

---

### ✅ 커밋 메시지 작성 예시

```bash
feat: 검색 에이전트 추가
fix: API 호출 시 파라미터 오류 수정
chore: Prettier 설치 및 설정 파일 추가
docs: 커밋 메시지 규칙 문서화
```
---

## 구글 드라이브 문서 링크

| 파일 | 링크 |
|-----|-----|
| **WBS** | https://docs.google.com/spreadsheets/d/15CpdDpFgmhpRz0SvhkaE3jrHX7b2lLZL/edit?gid=10265153#gid=10265153 |
| **이슈리스트** | https://docs.google.com/spreadsheets/d/1FYnkbNEERuY9K436zcLwGOSl8Z7tKNhx/edit?gid=575308982#gid=575308982 |
