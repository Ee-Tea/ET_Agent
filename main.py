from fastapi import FastAPI
from langserve import add_routes

from teacher.teacher import graph

# FastAPI 애플리케이션을 생성합니다. 이것이 웹 서버의 본체입니다.
app = FastAPI(
  title="ET_Agent Server",
  version="1.0",
  description="ET_Agent LangGraph 애플리케이션을 위한 서버",
)

# LangServe를 사용해 FastAPI 앱에 그래프 라우트를 추가합니다.
# 이렇게 하면 /graph/invoke, /graph/stream 같은 API 엔드포인트가 자동으로 생성됩니다.
add_routes(
    app,
    graph,
    path="/graph",
)

# (선택 사항) 서버가 살아있는지 확인할 수 있는 기본 경로
@app.get("/")
async def root():
    return {"message": "ET_Agent Server is running"}