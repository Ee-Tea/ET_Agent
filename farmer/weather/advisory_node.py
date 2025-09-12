# -*- coding: utf-8 -*-
"""
기상특보 노드 모듈
KMA 기상특보 데이터를 가져오는 노드
- 1차: 현재 특보 전용 wrn_now_data_new.php
- 2차: 이력 wrn_met_data.php (어제~내일) 백업 후 현재/예정만 필터
- 이 노드는 기본적으로 지역 필터를 하지 않지만, state에 주간 범위가 오면 기간/지역 필터를 래퍼에서 쉽게 하도록 payload 확장
"""

import os
import re
import json
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# 환경 설정
KMA_TIMEOUT = int(os.getenv("KMA_TIMEOUT", "30"))
WHEATHER_API_KEY_HUB = os.getenv("WHEATHER_API_KEY_HUB")  # 프로젝트 전반과 동일한 변수명 유지
KST = ZoneInfo("Asia/Seoul")

# 특보 매핑 (확장된 버전)
WRN_MAP = {
    "T":"태풍","W":"강풍","R":"호우","C":"한파","D":"건조","O":"해일","N":"지진해일","V":"풍랑","S":"대설","Y":"황사","H":"폭염","F":"안개",
    "A":"기타","B":"기타","E":"기타","G":"기타","I":"기타","J":"기타","K":"기타","L":"기타","M":"기타","P":"기타","Q":"기타","U":"기타","X":"기타","Z":"기타"
}
LVL_MAP = {"1":"예비특보","2":"주의보","3":"경보"}
CMD_MAP = {"1":"발표","2":"대치","3":"해제","4":"대치해제","5":"연장","6":"변경","7":"변경해제"}
REGION_CODE_RE = re.compile(r"^[A-Z]\d{7}$")

# 지역 매핑
from .utils import load_region_map, REGION_MAP
load_region_map()

def resolve_region(token: str) -> str:
    if not token:
        return "N/A"
    return REGION_MAP.get(token.strip(), token)

def _parse_kst_yyyymmddHHMM(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y%m%d%H%M").replace(tzinfo=KST)
    except Exception:
        return None

def _fmt_kst(yyyymmddHHMM: str) -> str:
    dt = _parse_kst_yyyymmddHHMM(yyyymmddHHMM)
    if not dt:
        return yyyymmddHHMM or "N/A"
    t24 = dt.strftime("%H:%M")
    ampm = dt.strftime("%p %I:%M").replace("AM", "오전").replace("PM", "오후")
    return f"{dt.strftime('%Y-%m-%d')} {t24}({ampm}) KST"

def _make_payload_from_parts(
    *, tm_st: str | None, tm_ed: str | None, tm_fc: str | None, tm_ef: str | None,
    reg_id: str | None, region_name_fallback: str | None,
    wrn: str | None, lvl: str | None, cmd: str | None, grd: str | None
) -> Dict[str, str]:
    reg_token = (reg_id or "").strip()
    region_name = region_name_fallback or resolve_region(reg_token)

    payload = {
        "source": "KMA_ADVISORY",
        "region_raw": reg_token,
        "region_name": region_name,
        "region_type": "code" if REGION_CODE_RE.match(reg_token or "") else "name",
        "hazard_code": (wrn or "").strip(),
        "hazard_name": WRN_MAP.get((wrn or "").strip(), "알수없음"),
        "level_code": (lvl or "").strip(),
        "level_name": LVL_MAP.get((lvl or "").strip(), "N/A"),
        "command_code": (cmd or "").strip(),
        "command_name": CMD_MAP.get((cmd or "").strip(), (cmd or "")),
        # 창구표기: 발효시각 우선
        "window_start": (tm_ef or tm_st or "") or "N/A",
        "window_end": tm_ed or "N/A",
        "window_start_kst": _fmt_kst((tm_ef or tm_st or "") or ""),
        "window_end_kst": _fmt_kst(tm_ed or "") if (tm_ed or "") else "N/A",
        "announce_time_kst": _fmt_kst(tm_fc or "") if tm_fc else None,
    }
    if (wrn or "") == "T" and grd:
        payload["typhoon_grade"] = grd

    # 상태/표기 보조
    status_label = "예정" if (payload["window_start_kst"] != "N/A" and _parse_kst_yyyymmddHHMM(payload["window_start"]) and _parse_kst_yyyymmddHHMM(payload["window_start"]) > datetime.now(tz=KST)) else "진행/유지"
    payload["status_label"] = status_label

    parts = [f"지역: {region_name} ({reg_token})"]
    if payload["window_start_kst"] != "N/A" and payload["window_end_kst"] != "N/A":
        parts.append(f"기간: {payload['window_start_kst']} ~ {payload['window_end_kst']}")
    elif payload["window_start_kst"] != "N/A":
        parts.append(f"시각: {payload['window_start_kst']}")
    if payload["announce_time_kst"]:
        parts.append(f"발표시각: {payload['announce_time_kst']}")
    parts += [
        f"현상: {payload['hazard_name']}({payload['hazard_code']})",
        f"수준: {payload['level_name']}({payload['level_code']})",
        f"명령: {payload['command_name']}({payload['command_code']})",
        f"상태: {payload['status_label']}"
    ]
    if "typhoon_grade" in payload:
        parts.append(f"태풍 등급: {payload['typhoon_grade']}")

    return {
        "json": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "human": " | ".join(parts)
    }

def _read_lines(text: str) -> List[List[str]]:
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("7777END"):
            continue
        if s.endswith("="):
            s = s[:-1]
        lines.append([c.strip() for c in s.split(",")])
    return lines

# 1) 현재 특보 전용 API
def fetch_wrn_now() -> List[Dict[str, str]]:
    if not WHEATHER_API_KEY_HUB:
        print("❌ API 키(WHEATHER_API_KEY_HUB)가 없습니다.")
        return []
    base = "https://apihub.kma.go.kr/api/typ01/url/wrn_now_data_new.php"
    params = {
        "fe": "f",        # 발표시간 기준
        "tm": "",         # 빈 값이면 현재 시각
        "disp": "0",
        "authKey": WHEATHER_API_KEY_HUB,
    }
    try:
        r = requests.get(base, params=params, timeout=KMA_TIMEOUT)
        r.raise_for_status()
        text = r.content.decode("euc-kr", errors="replace")
    except Exception:
        return []

    rows = _read_lines(text)
    docs: List[Dict[str, str]] = []
    for raw in rows:
        # 문서 필드 순서(문서 기준): REG_UP, REG_UP_KO, REG_ID, REG_KO, TM_FC, TM_EF, WRN, LVL, CMD
        reg_id  = raw[2] if len(raw) > 2 else ""
        reg_ko  = raw[3] if len(raw) > 3 else ""
        tm_fc   = raw[4] if len(raw) > 4 else ""
        tm_ef   = raw[5] if len(raw) > 5 else ""
        wrn     = raw[6] if len(raw) > 6 else ""
        lvl     = raw[7] if len(raw) > 7 else ""
        cmd     = raw[8] if len(raw) > 8 else ""

        # 해제 계열 명령 제외
        if cmd in {"3", "4", "7"}:
            continue

        doc = _make_payload_from_parts(
            tm_st=None, tm_ed=None, tm_fc=tm_fc, tm_ef=tm_ef,
            reg_id=reg_id, region_name_fallback=reg_ko,
            wrn=wrn, lvl=lvl, cmd=cmd, grd=None
        )
        docs.append(doc)
    return docs

# 2) 이력 API(백업): 어제~내일 범위에서 현재/예정만 포함
def fetch_wrn_history_window() -> List[Dict[str, str]]:
    if not WHEATHER_API_KEY_HUB:
        return []
    base = "https://apihub.kma.go.kr/api/typ01/url/wrn_met_data.php"
    now = datetime.now(tz=KST)
    params = {
        "authKey": WHEATHER_API_KEY_HUB,
        "wrn": "",        # 전체
        "reg": "",        # 전체
        "tmfc1": (now - timedelta(days=1)).strftime("%Y%m%d%H%M"),
        "tmfc2": (now + timedelta(days=1)).strftime("%Y%m%d%H%M"),
        "disp": "1",
    }

    try:
        r = requests.get(base, params=params, timeout=KMA_TIMEOUT)
        r.raise_for_status()
        text = r.content.decode("euc-kr", errors="replace")
    except Exception:
        return []

    rows = _read_lines(text)
    docs: List[Dict[str, str]] = []
    for raw in rows:
        # 필드: TM_ST, TM_ED, REG_SP, REG_UP, REG_ID, WRN, LVL, CMD, GRD, ..., TM_FC, TM_EF, ...
        tm_st  = raw[0] if len(raw) > 0 else ""
        tm_ed  = raw[1] if len(raw) > 1 else ""
        reg_id = raw[4] if len(raw) > 4 else ""
        wrn    = raw[5] if len(raw) > 5 else ""
        lvl    = raw[6] if len(raw) > 6 else ""
        cmd    = raw[7] if len(raw) > 7 else ""
        grd    = raw[8] if len(raw) > 8 else ""
        tm_fc  = None
        tm_ef  = None

        # 해제 계열 제외
        if cmd in {"3", "4", "7"}:
            continue

        st = _parse_kst_yyyymmddHHMM(tm_st)
        ed = _parse_kst_yyyymmddHHMM(tm_ed)
        if st and ed:
            # 현재/예정만 포함
            if not (st >= now or (st <= now <= ed)):
                continue

        doc = _make_payload_from_parts(
            tm_st=tm_st, tm_ed=tm_ed, tm_fc=tm_fc, tm_ef=tm_ef,
            reg_id=reg_id, region_name_fallback=None,
            wrn=wrn, lvl=lvl, cmd=cmd, grd=grd
        )
        docs.append(doc)
    return docs

class AdvisoryNode:
    """기상특보 노드 클래스 (지역 필터는 하지 않음; 상위 래퍼에서 처리)"""

    def __init__(self):
        self.name = "advisory_node"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out: List[Dict[str, str]] = []

        # 1) 현재 특보 우선
        now_docs = fetch_wrn_now()
        out.extend(now_docs)

        # 2) 보강: 0건이면 이력 API에서 현재/예정만 추림
        if len(out) == 0:
            hist_docs = fetch_wrn_history_window()
            out.extend(hist_docs)

        # 상태에 저장 (지역/기간 필터는 상위에서)
        state.setdefault("advisory_data", [])
        state["advisory_data"].extend(out)

        # quiet mode: no debug prints
        return state
