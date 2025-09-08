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
WHEATHER_API_KEY_HUB=REDACTED("WHEATHER_API_KEY_HUB")
KST = ZoneInfo("Asia/Seoul")

# 권역 매핑 (utils.py에서 import)
from .utils import load_region_map, REGION_MAP

# 모듈 로드 시 지역 매핑 초기화
load_region_map()

def now_kst() -> datetime:
    """현재 KST 시간 반환"""
    return datetime.now(tz=KST)

def fmt_kst_any(s: str) -> str:
    """다양한 KST 시간 형식 포맷팅"""
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
    """공백 정규화"""
    return re.sub(r"\s+", "", s or "")

def REGION_parse_wl_line(line: str) -> Optional[Dict[str, str]]:
    """권역예보 라인 파싱"""
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
    """일별로 권역예보 데이터 병합"""
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
    """재시도가 포함된 HTTP 요청"""
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

def fetch_mid_week_land(reg_id: str, target_date: datetime, tmfc_range_days: int = 3, widen_days: int = 0,
                        disp: str = "1", help_flag: str = "0", merge_day: bool = True) -> List[Dict[str, str]]:
    """KMA 권역 육상예보 주간 데이터 가져오기"""
    if not WHEATHER_API_KEY_HUB:
        print("❌ API 키(WHEATHER_API_KEY_HUB)가 없습니다.")
        return []
    
    now = now_kst()
    tmfc1 = now.astimezone(KST) - timedelta(days=tmfc_range_days)
    tmfc2 = now
    tmef1 = target_date - timedelta(days=widen_days)
    tmef2 = target_date + timedelta(days=widen_days)
    
    BASE = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_wl.php"
    params = {
        "reg": reg_id,
        "tmfc1": tmfc1.strftime("%Y%m%d%H"),
        "tmfc2": tmfc2.strftime("%Y%m%d%H"),
        "tmef1": tmef1.strftime("%Y%m%d"),
        "tmef2": tmef2.strftime("%Y%m%d"),
        "disp": disp,
        "help": help_flag,
        "authKey": WHEATHER_API_KEY_HUB
    }
    
    url = f"{BASE}?{urlencode(params)}"
    
    try:
        r = request_with_retries(url, timeout=KMA_TIMEOUT, retries=3, backoff=1.5)
        text = r.content.decode("euc-kr", errors="replace")
        
        latest = {}
        want_d8 = target_date.strftime("%Y%m%d")
        
        for ln in text.splitlines():
            item = REGION_parse_wl_line(ln)
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
        print("🧩 노드: 중기예보 데이터 수집")
        
        try:
            # 질문에서 지역과 날짜 추출
            question = state.get("question", "")
            
            # 지역 추출 (간단한 버전)
            region_code = "11B00000"  # 기본값: 서울
            region_name = "서울·인천·경기도"
            
            # CSV에서 지역 매핑이 있으면 사용
            if REGION_MAP:
                for code, name in REGION_MAP.items():
                    if name in question or code in question:
                        region_code = code
                        region_name = name
                        print(f"   - 권역코드 로드: {region_code} ({region_name})")
                        break
            
            # 날짜 추출 (간단한 버전)
            target_date = now_kst() + timedelta(days=5)  # 기본값: 5일 후
            
            # 질문에서 날짜 관련 키워드 확인
            if "내일" in question or "1일" in question:
                target_date = now_kst() + timedelta(days=1)
            elif "모레" in question or "2일" in question:
                target_date = now_kst() + timedelta(days=2)
            elif "3일" in question:
                target_date = now_kst() + timedelta(days=3)
            elif "4일" in question:
                target_date = now_kst() + timedelta(days=4)
            elif "5일" in question:
                target_date = now_kst() + timedelta(days=5)
            elif "6일" in question:
                target_date = now_kst() + timedelta(days=6)
            elif "7일" in question or "일주일" in question:
                target_date = now_kst() + timedelta(days=7)
            
            # 중기예보 데이터 가져오기
            mid_forecasts = fetch_mid_week_land(
                reg_id=region_code,
                target_date=target_date,
                tmfc_range_days=3,
                widen_days=0
            )
            
            # 상태에 결과 저장
            if "mid_forecast_data" not in state:
                state["mid_forecast_data"] = []
            state["mid_forecast_data"].extend(mid_forecasts)
            
            print(f"   - ✅ 중기예보 데이터 수집 완료: {len(mid_forecasts)}개 (지역: {region_name}, 날짜: {target_date.strftime('%Y-%m-%d')})")
            
        except Exception as e:
            print(f"   - ❌ 중기예보 데이터 수집 실패: {e}")
            if "mid_forecast_data" not in state:
                state["mid_forecast_data"] = []
        
        return state