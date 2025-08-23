"""초경량 LangGraph 서버 (무거운 에이전트 초기화 생략)
실행:
  python -m pip install fastapi uvicorn langgraph langserve langchain-core httpx<0.28
  uvicorn mini_server:app --reload --port 8000
"""
import os, sys
from fastapi import FastAPI
from langserve import add_routes

# 프로젝트 루트 경로 추가
sys.path.append(os.getcwd())

from teacher.teacher_graph import Orchestrator  # init_agents=False 로 무거운 초기화 스킵

orch = Orchestrator(user_id="dev", service="teacher", chat_id="local", init_agents=False)
graph = orch.build_teacher_graph()

app = FastAPI(title="Mini ET_Agent LangGraph", version="0.1")
add_routes(app, graph, path="/graph")

@app.get("/")
async def root():
    return {"ok": True, "nodes": len(list(graph.nodes))}
