# img2txt_generation.py
from __future__ import annotations
import base64, json, os, time
from typing import List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError, APIStatusError

# =========================
# 환경 변수 로드 & 클라이언트
# =========================
def _load_env() -> None:
    """
    프로젝트 루트(.env) 우선, 없으면 CWD에서 .env 시도
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / ".env",  # .../llm-T/.env (기존 경로)
        here.parents[2] / ".env",
        here.parents[1] / ".env",
        here.parent / ".env",
        Path.cwd() / ".env",
    ]
    for p in candidates:
        try:
            if p.is_file():
                load_dotenv(p)
                break
        except Exception:
            pass

_load_env()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 없습니다. 루트 또는 현재 경로의 .env를 확인하세요.")

# 모델 & 토큰 한도
MODEL = os.getenv("OPENAI_VISION_MODEL", "o4-mini")  # 예: "o4-mini" 또는 "gpt-4o-mini"
MAX_OUTPUT_TOKENS_DEFAULT = int(os.getenv("MAX_OUTPUT_TOKENS", "1200"))
MAX_OUTPUT_TOKENS_FALLBACK = 2000  # 길이 초과 시에만 사용
# 추가: JSON만 출력하도록 강하게 요구하는 힌트
STRICT_JSON_HINT = (
    "출력은 오직 하나의 JSON 객체만! 앞뒤 설명/코드블록/백틱 금지. "
    "스키마: { problems: [{ number, question, options, skipped?, reason? }], has_more? }"
)

client = OpenAI(api_key=API_KEY)

# =========================
# 프롬프트 & JSON 스키마
# =========================
SYSTEM_PROMPT = (
    "너는 OCR+문항 추출 어시스턴트야. 입력 이미지를 보고 '각 문제의 질문'과 '보기 4개'만 "
    "정확히 분리해 JSON으로 출력해. 글자는 원문 그대로(오탈자/줄바꿈 포함) 복사해. "
    "추측하거나 보강하지 마. 보기 접두부(1., ①, A), 괄호 등은 제거하고 본문만 담아. "
    "여러 문제가 한 장에 있으면 각 문제를 구분해 모두 추출해. "
    "보기가 정확히 4개가 아닐 경우 그 문제는 'skipped': true 로 표시하고 'reason'에 이유를 적어. "
    "한 번 호출에서 최대 5문제까지만 출력하고, 더 남아 보이면 마지막 문제 다음에 'has_more': true를 포함해."
)

USER_INSTRUCTIONS = (
    "아래 이미지는 한국어 객관식 시험지 예시다. "
    "문제 번호(예: '문제 1', '1.', '①' 등) 기준으로 문제를 분리하고, "
    "각 문제의 '질문(question)'과 '보기(options)' 4개를 뽑아줘. "
    "문항 외의 설명/부제/과목명/난이도 표기는 제외. "
    "모든 텍스트는 이미지에서 보이는 그대로 복사해. "
    "이미지에서 판독 불가한 글자는 가능한 범위에서 그대로 유지하고, 확신이 없으면 그 문제를 skip 처리해."
)

JSON_SCHEMA = {
    "name": "mcq_list",
    "schema": {
        "type": "object",
        "properties": {
            "problems": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "number": {"type": ["string", "integer"]},
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 6
                        },
                        "skipped": {"type": "boolean", "default": False},
                        "reason": {"type": "string"}
                    },
                    "required": ["number", "question", "options"]
                }
            },
            "has_more": {"type": "boolean"}
        },
        "required": ["problems"],
        "additionalProperties": False
    }
}

# =========================
# 유틸
# =========================
def to_data_url(image_path: str) -> str:
    ext = Path(image_path).suffix.lower().lstrip(".") or "png"
    if ext == "jpg":
        ext = "jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"

def _extract_output_text(resp) -> str:
    """
    Responses API의 다양한 포맷에 안전하게 대응하여 텍스트를 꺼낸다.
    """
    # 1) 속성 제공 시
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    # 2) 표준 경로 시도
    try:
        # resp.output -> list[Message]
        for msg in getattr(resp, "output", []) or []:
            for part in getattr(msg, "content", []) or []:
                maybe = getattr(part, "text", None)
                if isinstance(maybe, str) and maybe.strip():
                    return maybe
                # 일부 SDK는 dict로 들어올 수 있음
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    if part["text"].strip():
                        return part["text"]
    except Exception:
        pass
    raise RuntimeError("응답에서 텍스트를 찾지 못했습니다. SDK/모델 호환성을 확인하세요.")

def _is_length_truncated(resp) -> bool:
    """
    finish_reason이 'length'인지 추정
    """
    try:
        for msg in getattr(resp, "output", []) or []:
            fr = getattr(msg, "finish_reason", None)
            if isinstance(fr, str) and fr.lower() == "length":
                return True
            # dict 호환
            if isinstance(msg, dict) and (msg.get("finish_reason", "") == "length"):
                return True
    except Exception:
        pass
    return False


def _create_with_resilience(contents: list, max_output_tokens: int, use_schema: bool = True, retries: int = 3):
    attempt = 0
    schema_payload = {"type": "json_schema", "json_schema": JSON_SCHEMA} if use_schema else {"type": "json_object"}

    while True:
        try:
            # 기본: structured outputs 사용
            return client.responses.create(
                model=MODEL,
                max_output_tokens=max_output_tokens,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                    {"role": "user",   "content": contents},
                ],
                response_format=schema_payload,
            )
        except TypeError:
            # >>> SDK가 response_format 인자를 지원하지 않는 경우(지금 에러)
            # 프롬프트에 JSON-only 지시를 주입하고, response_format 없이 재호출
            patched = [{"type": "input_text", "text": STRICT_JSON_HINT}] + contents
            return client.responses.create(
                model=MODEL,
                max_output_tokens=max_output_tokens,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT + ' ' + STRICT_JSON_HINT}]},
                    {"role": "user",   "content": patched},
                ],
            )
        except BadRequestError as e:
            # 서버 측에서 json_schema 미지원 → json_object로 폴백
            msg = str(e)
            if use_schema and ("response_format" in msg or "json_schema" in msg or "Unknown parameter" in msg):
                use_schema = False
                schema_payload = {"type": "json_object"}
                continue
            raise
        except APIStatusError:
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(min(2 ** attempt, 10))


def call_gpt_on_images(image_paths: List[str]) -> Dict[str, Any]:
    # 여러 이미지를 한 번에 전달(멀티페이지 시험지 등)
    contents = [{"type": "input_text", "text": USER_INSTRUCTIONS}]
    for p in image_paths:
        contents.append({"type": "input_image", "image_url": to_data_url(p)})

    # 1차: 1200 토큰
    resp = _create_with_resilience(contents, max_output_tokens=MAX_OUTPUT_TOKENS_DEFAULT, use_schema=True)

    # 길이 초과 시 2000으로 재시도
    if _is_length_truncated(resp):
        resp = _create_with_resilience(contents, max_output_tokens=MAX_OUTPUT_TOKENS_FALLBACK, use_schema=True)

    text = _extract_output_text(resp)

    # json_schema 또는 json_object인 경우 파싱
    data = json.loads(text)

    # json_object 폴백 시에도 최소 형태 보장
    if "problems" not in data or not isinstance(data["problems"], list):
        raise RuntimeError("모델 응답에 'problems' 배열이 없습니다. 프롬프트/모델을 확인하세요.")
    return data

def extract_to_file(image_paths: List[str], out_path: str) -> None:
    data = call_gpt_on_images(image_paths)

    # 보기가 정확히 4개인 문제만 엄격 필터
    strict: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for p in data.get("problems", []):
        if isinstance(p.get("options"), list) and len(p["options"]) == 4 and not p.get("skipped", False):
            strict.append(p)
        else:
            skipped.append(p)

    result = {"problems": strict, "skipped": skipped}
    if "has_more" in data:
        result["has_more"] = bool(data["has_more"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# =========================
# 실행 예시
# =========================
if __name__ == "__main__":
    # 기본 예시 이미지 경로 (로컬/컨테이너 상황에 맞게 선택)
    # 1) 이 대화에서 제공된 이미지(컨테이너 경로)
    default_candidates = [
        "./teacher/agents/solution/user_problems.png",
        "/mnt/data/user_problems.png",
    ]
    img_path = next((p for p in default_candidates if Path(p).exists()), default_candidates[0])
    imgs = [img_path]

    ts = int(time.time())
    out = f"./teacher/agents/solution/mcq_output_{ts}.json"
    extract_to_file(imgs, out)
    print(f"Saved -> {out}")
