# -*- coding: utf-8 -*-
"""
중기예보 노드 모듈
KMA 권역 육상예보 주간 데이터를 가져오는 노드
"""

import os
import re
import json
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

# 환경 설정
KMA_TIMEOUT = int(os.getenv("KMA_TIMEOUT", "30"))
WHEATHER_API_KEY_HUB = os.getenv("WHEATHER_API_KEY_HUB")
KST = ZoneInfo("Asia/Seoul")

# 권역 매핑
from .utils import load_region_map, REGION_MAP
load_region_map()

# =========[ 권역예보 전용: CSV 권역/질의 해석 ]=========
REGION_REGIONS_CSV = os.getenv("REGION_CSV_PATH", "farmer/all_regions_combined.csv")

def REGION_load_all_from_csv(path: str) -> Dict[str, str]:
    """CSV에서 모든 지역 정보 로드"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"권역 CSV를 찾을 수 없습니다: {path}")
    mapping: Dict[str, str] = {}
    import csv
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("reg_id") or row.get("REG_ID") or "").strip()
            rname = (row.get("reg_name") or row.get("REG_NAME") or "").strip()
            if rid and rname and rid not in mapping:
                mapping[rid] = rname
    if not mapping:
        raise ValueError("CSV에서 유효한 권역 정보를 읽지 못했습니다.")
    return mapping

def REGION_split_region_only(all_map: Dict[str, str]) -> Dict[str, str]:
    """권역 코드만 추출"""
    REGION_CODE_RE = re.compile(r"^(11[A-Z](\d)?0{4,5}|L\d{7}|S\d{7})$")
    region_map = {rid: nm for rid, nm in all_map.items() if REGION_CODE_RE.match(rid)}
    if not region_map:
        known = ["11B00000","11D10000","11D20000","11C20000","11C10000",
                 "11F20000","11F10000","11H20000","11H10000","11G00000"]
        region_map = {rid: all_map[rid] for rid in known if rid in all_map}
    return region_map

def REGION_build_alias_map(region_map: Dict[str, str]) -> Dict[str, str]:
    alias: Dict[str, str] = {}
    norm = lambda x: re.sub(r"\s+", "", x or "")
    def add(a: str, full: str):
        a = norm(a)
        if a and a not in alias:
            alias[a] = full
    for _code, full in region_map.items():
        add(full, full)
        for p in full.split("·"):
            add(p, full)
        subs = {
            "충청북도":"충북","충청남도":"충남",
            "전라북도":"전북","전라남도":"전남",
            "경상북도":"경북","경상남도":"경남",
            "제주도":"제주","경기도":"경기",
        }
        for long, short in subs.items():
            if long in full:
                add(short, full); add(long, full)
        if ("서울" in full) and ("인천" in full) and ("경기" in full or "경기도" in full):
            add("수도권", full)
        if "강원" in full and ("영서" in full or "영동" in full):
            add("강원", full)
            if "영서" in full: add("영서", full); add("강원영서", full)
            if "영동" in full: add("영동", full); add("강원영동", full)
    return alias

# CSV 로드
REGION_ALL_MAP: Dict[str, str] = {}
REGION_LAND_MAP: Dict[str, str] = {}
REGION_ALIASES: Dict[str, str] = {}
if os.path.exists(REGION_REGIONS_CSV):
    REGION_ALL_MAP = REGION_load_all_from_csv(REGION_REGIONS_CSV)
    REGION_LAND_MAP = REGION_split_region_only(REGION_ALL_MAP)
    REGION_ALIASES = REGION_build_alias_map(REGION_LAND_MAP)
    print(f"✅ 권역 CSV 로드: 전체 {len(REGION_ALL_MAP)}개 / 권역 {len(REGION_LAND_MAP)}개 (파일: {REGION_REGIONS_CSV})")
else:
    print("⚠️ 권역 CSV 파일이 없어 권역 매핑을 건너뜁니다. (REGION_REGIONS_CSV)")

REGION_FAMILY_RULES = [
    (r"^11B",  "11B00000"), (r"^11D1", "11D10000"), (r"^11D2", "11D20000"),
    (r"^11C1", "11C10000"), (r"^11C2", "11C20000"), (r"^11F1", "11F10000"),
    (r"^11F2", "11F20000"), (r"^11H1", "11H10000"), (r"^11H2", "11H20000"),
    (r"^11G",  "11G00000"),
    (r"^L100", "L1000000"), (r"^L101", "L1010000"), (r"^L102", "L1020000"),
    (r"^L103", "L1030000"), (r"^L104", "L1040000"), (r"^L105", "L1050000"),
    (r"^L106", "L1060000"), (r"^L107", "L1070000"), (r"^L108", "L1080000"),
    (r"^S100", "S1000000"), (r"^S110", "S1100000"), (r"^S120", "S1200000"),
    (r"^S130", "S1300000"), (r"^S140", "S1400000"), (r"^S150", "S1500000"),
    (r"^S160", "S1600000"), (r"^S170", "S1700000"), (r"^S180", "S1800000"),
]

def REGION_normalize_region_reg_code(code_like: str) -> Optional[str]:
    """세부 코드를 권역 코드로 정규화 (중기예보 API용 11X00000 형식)"""
    c = (code_like or "").strip()
    if not c: return None
    if c in REGION_LAND_MAP and c.startswith('11'): return c
    for pat, target in REGION_FAMILY_RULES:
        if re.match(pat, c) and target.startswith('11'):
            return target if target in REGION_LAND_MAP else target
    if c.startswith('L'):
        if c.startswith('L101'): return '11B00000'
        if c.startswith('L102'): return '11D10000'
        if c.startswith('L103'): return '11C10000'
        if c.startswith('L104'): return '11C20000'
        if c.startswith('L105'): return '11F10000'
        if c.startswith('L106'): return '11F20000'
        if c.startswith('L107'): return '11H10000'
        if c.startswith('L108'): return '11H20000'
    if c.startswith('S'): return None
    return None

def now_kst() -> datetime:
    return datetime.now(tz=KST)

def fmt_kst_any(s: str) -> str:
    s = (s or "").strip()
    patterns = [
        ("%Y%m%d%H%M", "%Y-%m-%d %H:%M KST"),
        ("%Y%m%d%H", "%Y-%m-%d %H:00 KST"),
        ("%Y%m%d", "%Y-%m-%d")
    ]
    for fin, fout in patterns:
        try:
            dt = datetime.strptime(s, fin).replace(tzinfo=KST)
            return dt.strftime(fout)
        except Exception:
            pass
    return s

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", "", s or "")

def _parse_wl_line(line: str) -> Optional[Dict[str, str]]:
    s = (line or "").strip()
    if not s or s.startswith("#") or s.startswith("7777END"):
        return None
    if s.endswith("="):
        s = s[:-1]
    cols = [c.strip() for c in s.split(",") if c.strip() != ""] if "," in s else [c.strip() for c in re.split(r"\s+", s) if c.strip()]
    if len(cols) < 3:
        return None
    looks_date = lambda x: bool(re.fullmatch(r"\d{8}(\d{2}(\d{2})?)?", x))
    if len(cols) >= 11 and looks_date(cols[1]) and looks_date(cols[2]):
        return {
            "reg_id": cols[0],
            "tmfc": cols[1],
            "tmef": cols[2],
            "sky_code": cols[6],
            "pre_code": cols[7],
            "conf": cols[8],
            "wf": cols[9],
            "rn_st": cols[10]
        }
    reg_id = cols[0]
    tmfc = None
    tmef = None
    rest = []
    for c in cols[1:]:
        if tmfc is None and looks_date(c):
            tmfc = c
            continue
        if tmfc is not None and tmef is None and looks_date(c):
            tmef = c
            continue
        rest.append(c)
    return {
        "reg_id": reg_id,
        "tmfc": tmfc or "",
        "tmef": tmef or "",
        "wf": " ".join(rest).strip(),
        "conf": "",
        "rn_st": "",
        "sky_code": "",
        "pre_code": ""
    }

def REGION_merge_by_day(latest_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    day_map = {}
    for it in latest_rows:
        tmef = it.get("tmef", "")
        if not tmef or len(tmef) < 8:
            continue
        day = tmef[:8]
        hh = (tmef + "  ")[8:10]
        slot = "am" if hh == "00" else ("pm" if hh == "12" else "etc")
        rec = day_map.setdefault(day, {"am": None, "pm": None, "etc": []})
        if slot in ("am", "pm"):
            rec[slot] = it
        else:
            rec["etc"].append(it)
    merged = []
    for day, slots in sorted(day_map.items()):
        am = slots["am"]
        pm = slots["pm"]
        if am or pm:
            tmfc_candidates = [x.get("tmfc", "") for x in (am, pm) if x]
            tmfc = max(tmfc_candidates) if tmfc_candidates else ""
            wf_am = (am or {}).get("wf", "")
            rn_am = (am or {}).get("rn_st", "")
            wf_pm = (pm or {}).get("wf", "")
            rn_pm = (pm or {}).get("rn_st", "")
            bits = []
            if wf_am:
                bits.append(f"오전 {wf_am}{f'({rn_am}%)' if rn_am.isdigit() else ''}")
            if wf_pm:
                bits.append(f"오후 {wf_pm}{f'({rn_pm}%)' if rn_pm.isdigit() else ''}")
            wf = " / ".join(bits) if bits else (wf_am or wf_pm or "")
            merged.append({
                "tmfc": tmfc,
                "tmef": day,
                "wf": wf,
                "rn_st": "",
                "conf": "",
                "sky_code": "",
                "pre_code": ""
            })
    return merged

def request_with_retries(url: str, timeout: int, retries: int, backoff: float) -> requests.Response:
    last = None
    for i in range(1, retries + 1):
        try:
            print(f"[HTTP] GET {url} (try={i})")
            r = requests.get(url, timeout=timeout)
            print(f"[HTTP] -> {r.status_code}, {len(r.content)} bytes")
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if i < retries:
                import time
                time.sleep(backoff * i)
    raise last or RuntimeError("request failed")

def to_tmfc(dt: datetime) -> str:
    k = dt.astimezone(KST)
    return k.strftime("%Y%m%d%H")

def to_tmef(dt: datetime) -> str:
    k = dt.astimezone(KST)
    return k.strftime("%Y%m%d")

def fetch_mid_week_land(reg_id: str, target_date: datetime, tmfc_range_days: int = 3, widen_days: int = 0,
                        disp: str = "1", help_flag: str = "0", merge_day: bool = True) -> List[Dict[str, str]]:
    """KMA 권역 육상예보 주간 데이터 가져오기"""
    if not WHEATHER_API_KEY_HUB:
        print("❌ API 키(WHEATHER_API_KEY_HUB)가 없습니다.")
        return []
    
    now = now_kst()
    tmfc1 = to_tmfc(now - timedelta(days=tmfc_range_days))
    tmfc2 = to_tmfc(now)
    tmef1 = to_tmef(target_date - timedelta(days=widen_days))
    tmef2 = to_tmef(target_date + timedelta(days=widen_days))
    
    BASE = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_wl.php"
    params = {
        "reg": reg_id,
        "tmfc1": tmfc1, "tmfc2": tmfc2,
        "tmef1": tmef1, "tmef2": tmef2,
        "disp": disp, "help": help_flag,
        "authKey": WHEATHER_API_KEY_HUB,
    }
    
    url = f"{BASE}?{urlencode(params)}"
    
    try:
        r = request_with_retries(url, timeout=KMA_TIMEOUT, retries=3, backoff=1.5)
        text = r.content.decode("euc-kr", errors="replace")
        
        latest = {}
        want_d8 = to_tmef(target_date)
        
        for ln in text.splitlines():
            if not ln.strip():
                continue
            item = _parse_wl_line(ln)
            if not item:
                continue
            if item.get("reg_id") != reg_id:
                continue
            tmef = item.get("tmef", "")
            if (tmef or "")[:8] != want_d8:
                continue
            prev = latest.get(tmef)
            if (prev is None) or (item.get("tmfc", "") > prev.get("tmfc", "")):
                latest[tmef] = item
        
        if not latest:
            print(f"   - ❌ 데이터 없음: 대상 날짜({want_d8})에 대한 데이터가 없습니다.")
            return []
        
        rows = [latest[k] for k in sorted(latest.keys())]
        if merge_day:
            rows = REGION_merge_by_day(rows)
        
        docs = []
        region_name = REGION_MAP.get(reg_id, reg_id)
        
        for it in rows:
            is_day_level = len(it.get("tmef", "")) == 8
            target_label = (fmt_kst_any(it["tmef"]) if not is_day_level else
                          datetime.strptime(it["tmef"], "%Y%m%d").replace(tzinfo=KST).strftime("%Y-%m-%d"))
            wf = (it.get("wf", "") or "").strip()
            conf = (it.get("conf", "") or "").strip()
            rn = (it.get("rn_st", "") or "").strip()
            bits = []
            if wf:
                bits.append(wf)
            if conf and conf != "없음":
                bits.append(conf)
            if rn.isdigit():
                bits.append(f"강수확률 {rn}%")
            summary = " · ".join(bits) if bits else (wf or "예보문 없음")
            payload = {
                "source": "KMA_REGION_WEEK_LAND",
                "region_id": reg_id,
                "region_name": region_name,
                "announce_time": it.get("tmfc", ""),
                "forecast_time": it.get("tmef", ""),
                "sky_text": wf,
                "precip_type": conf,
                "precip_prob": rn,
                "sky_code": it.get("sky_code", ""),
                "pre_code": it.get("pre_code", ""),
            }
            human = f"{region_name} 권역예보(주간) — 발표 {fmt_kst_any(payload['announce_time'])}, 대상 {target_label}: {summary}"
            docs.append({
                "json": json.dumps(payload, ensure_ascii=False, separators=(',', ':')),
                "human": human
            })
        return docs
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP 오류: {e}")
        return []
    except Exception as e:
        print(f"❌ API 호출/파싱 오류: {e}")
        return []

class MidForecastNode:
    """중기예보 노드 클래스"""
    
    def __init__(self):
        self.name = "mid_forecast_node"
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """중기예보 노드 실행"""
        try:
            question = state.get("question", "")
            date_range = state.get("question_date_range")  # (start_dt, end_dt, is_weekly)
            # 지역 추출 (기본: 서울·인천·경기도)
            region_code = "11B00000"
            region_name = "서울·인천·경기도"
            if REGION_ALL_MAP:
                for code, name in REGION_ALL_MAP.items():
                    if name in question or code in question:
                        region_code = code
                        region_name = name
                        print(f"   - CSV에서 지역코드 로드: {region_code} ({region_name})")
                        break
            region_code = REGION_normalize_region_reg_code(region_code)
            print(f"   - 권역 코드 정규화: {region_code}")

            # 기본: 4일 후 (중기 최소)
            now = now_kst()
            target_date = now + timedelta(days=4)

            targets: List[datetime] = [target_date]
            # 주간 요청이면 범위 내 4~10일(중기 커버 영역 추정) 중 교집합 날짜들 뽑기
            if date_range:
                sdt, edt, _weekly = date_range
                # 중기는 통상 3~10일 범위를 다룸. 여기서는 sdt~edt 내의 날짜 중 4일 후 이상을 대상으로 함
                cur = sdt
                tmp = []
                while cur <= edt:
                    if (cur.date() - now.date()).days >= 4:
                        tmp.append(cur)
                    cur = cur + timedelta(days=1)
                if tmp:
                    targets = tmp

            all_docs: List[Dict[str, str]] = []
            for td in targets:
                docs = fetch_mid_week_land(
                    reg_id=region_code,
                    target_date=td,
                    tmfc_range_days=3,
                    widen_days=0
                )
                all_docs.extend(docs)

            state.setdefault("mid_forecast_data", [])
            state["mid_forecast_data"].extend(all_docs)
            print(f"   - ✅ 중기예보 데이터 수집 완료: {len(all_docs)}개 (지역: {region_name})")
        except Exception as e:
            print(f"   - ❌ 중기예보 데이터 수집 실패: {e}")
            state.setdefault("mid_forecast_data", [])
        return state
