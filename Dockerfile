# 공식 파이썬 이미지를 사용합니다.
FROM python:3.11-slim

# 파이썬이 .pyc 파일을 생성하지 않도록 합니다.
ENV PYTHONDONTWRITEBYTECODE 1
# 파이썬 출력이 버퍼링되지 않도록 하여 Docker 로그에 바로 표시되게 합니다.
ENV PYTHONUNBUFFERED 1

# 작업 디렉토리를 /api로 설정합니다.
WORKDIR /api

## 필수 OS 패키지 (opencv/reportlab 폰트 등)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   build-essential \
	   libgl1 \
	   libglib2.0-0 \
	   fonts-nanum \
	&& rm -rf /var/lib/apt/lists/*

# uv 와 기본 런타임 도구 설치 (langserve 사전 설치시 httpx 최신 깔리므로 requirements 이전 설치 최소화)
RUN pip install --no-cache-dir uv

# 의존성 파일 복사
COPY pyproject.toml uv.lock ./

# 나머지 애플리케이션 코드를 복사합니다.
COPY . .


RUN uv pip install --system --no-cache .

# 앱이 실행될 포트를 노출합니다.
EXPOSE 8000

# LangGraph 템플릿의 표준 시작점으로 애플리케이션을 실행합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]