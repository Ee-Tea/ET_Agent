# api/deps.py
from fastapi import Request

def get_services(request: Request):
    return request.app.state.services  # {"orchestrator": ..., "teacher": ...}

# api/routers/agent.py (기존 /chat/stream 대체 예)
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import httpx, os

router = APIRouter(prefix="/agent", tags=["agent"])
LGS_BASE = os.getenv("LGS_BASE", "http://langserve:8123")
GRAPH = os.getenv("LGS_GRAPH_PATH", "/agent")

@router.post("/stream")
async def stream_agent(body: dict):
    async def gen():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{LGS_BASE}{GRAPH}/stream", json={"input": body}) as r:
                if r.status_code >= 400:
                    raise HTTPException(r.status_code, await r.text())
                async for chunk in r.aiter_bytes():
                    yield chunk
    return StreamingResponse(gen(), media_type="text/event-stream")
