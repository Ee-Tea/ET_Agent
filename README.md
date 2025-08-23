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

---

## 🧩 Redis 설정 (유연한 환경 변수 구성)

`common/short_term/redis_memory.py` 는 아래 우선순위로 Redis 연결 정보를 결정합니다.

1. `REDIS_URL` (예: `redis://:password@redis:6379/0`)
2. 개별 변수: `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`, `REDIS_SSL`, `REDIS_SOCKET_TIMEOUT`
3. 미설정 시 기본값:
	 - Docker 컨테이너 내부: host=`redis`, port=`6379`
	 - 로컬 개발: host=`localhost`, port=`6380` (docker-compose 포트 매핑)

추가 동작:
- 컨테이너 내부 여부는 `/.dockerenv` 존재로 감지
- 연결 재시도: 기본 3회 (0.5s * 시도번호 지수적 증가)
- `decode_responses=True` 로 문자열 자동 디코딩
- 실패 시 `Orchestrator` 가 In-Memory 폴백

### .env 예시
```
# 단순 호스트/포트 지정
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_DB=0

# 혹은 URL 한 줄로
# REDIS_URL=redis://localhost:6380/0

# 선택적
# REDIS_PASSWORD=secret
# REDIS_SSL=false
# REDIS_SOCKET_TIMEOUT=2.5
```

### docker-compose 기본 포트
`docker-compose.yml` 에서 컨테이너 내부 6379 → 호스트 6380 매핑:
```
	redis:
		image: redis:7.2
		ports:
			- "6380:6379"
```
컨테이너 내부에서 다른 서비스가 사용할 경우 `REDIS_HOST=redis`, `REDIS_PORT=6379` 또는 `REDIS_URL=redis://redis:6379/0` 로 지정하면 됩니다.

### 문제 해결 TIP
| 증상 | 가능 원인 | 해결 |
|------|-----------|------|
| Connection refused | 포트 불일치 | 로컬이면 6380, 컨테이너면 6379 확인 |
| Authentication required | 비밀번호 설정된 Redis | `.env`에 `REDIS_PASSWORD` 추가 |
| Timeout | 방화벽/네트워크 지연 | `REDIS_HOST` 접근성 확인, `REDIS_SOCKET_TIMEOUT` 조정 |
| 데이터 안 쌓임 | TTL 만료 / 다른 DB 사용 | TTL(72h) 확인, `REDIS_DB` 일치 여부 확인 |

---
