import os
import httpx

AUTH_API_BASE = os.getenv("AUTH_API_BASE", "http://auth-api:8000")


async def verify_token(token: str) -> dict:
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(
            f"{AUTH_API_BASE}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()


