# -*- coding: utf-8 -*-
"""
기상특보 노드 모듈
KMA 기상특보 데이터를 가져오는 노드
"""

import os
import re
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# 환경 설정
KMA_TIMEOUT = int(os.getenv("KMA_TIMEOUT", "30"))
WHEATHER_API_KEY_HUB=REDACTED("WHEATHER_API_KEY_HUB")
KST = ZoneInfo("Asia/Seoul")

# 특보 매핑
WRN_MAP = {"T":"태풍","W":"강풍","R":"호우","C":"한파","D":"건조","O":"해일","N":"지진해일","V":"풍랑","S":"대설","Y":"황사","H":"폭염","F":"안개"}
LVL_MAP = {"1":"예비특보","2":"주의보","3":"경보"}
CMD_MAP = {"1":"발표","2":"대치","3":"해제","4":"대치해제","5":"연장","6":"변경","7":"변경해제"}
REGION_CODE_RE = re.compile(r"^[A-Z]\d{7}$")

# 지역 매핑 (utils.py에서 import)
from .utils import load_region_map, REGION_MAP

# 모듈 로드 시 지역 매핑 초기화
load_region_map()

def resolve_region(token: str) -> str:
    """지역 코드를 지역명으로 변환"""
    if not token:
        return "N/A"
    return REGION_MAP.get(token.strip(), token)

def fmt_kst_with_ampm(yyyymmddHHMM: str) -> str:
    """시간을 24시간제와 오전/오후 둘 다 표시"""
    try:
        dt = datetime.strptime(yyyymmddHHMM, "%Y%m%d%H%M").replace(tzinfo=KST)
        time_24h = dt.strftime("%H:%M")
        time_ampm = dt.strftime("%p %I:%M").replace("AM", "오전").replace("PM", "오후")
        date_str = dt.strftime("%Y-%m-%d")
        return f"{date_str} {time_24h}({time_ampm}) KST"
    except Exception:
        return yyyymmddHHMM

def _format_kma_record(raw: List[str]) -> Dict[str, str]:
    """KMA 특보 레코드 포맷팅"""
    tm_st = raw[0] if len(raw) > 0 else "N/A"
    tm_ed = raw[1] if len(raw) > 1 else "N/A"
    reg_token = raw[4].strip() if len(raw) > 4 else "N/A"
    wrn = (raw[5].strip() if len(raw) > 5 else "")
    lvl = (raw[6].strip() if len(raw) > 6 else "")
    cmd = (raw[7].strip() if len(raw) > 7 else "")
    grd = (raw[8].strip() if len(raw) > 8 else "")
    
    region_name = resolve_region(reg_token)
    
    payload = {
        "source": "KMA_ADVISORY",
        "region_raw": reg_token,
        "region_name": region_name,
        "region_type": "code" if REGION_CODE_RE.match(reg_token or "") else "name",
        "hazard_code": wrn,
        "hazard_name": WRN_MAP.get(wrn, "알수없음"),
        "level_code": lvl,
        "level_name": LVL_MAP.get(lvl, "N/A"),
        "command_code": cmd,
        "command_name": CMD_MAP.get(cmd, cmd),
        "window_start": tm_st,
        "window_end": tm_ed,
        "window_start_kst": fmt_kst_with_ampm(tm_st) if tm_st != "N/A" else "N/A",
        "window_end_kst": fmt_kst_with_ampm(tm_ed) if tm_ed != "N/A" else "N/A",
        "announce_time_kst": fmt_kst_with_ampm(tm_st) if cmd == "1" and tm_st != "N/A" else None,
    }
    
    if wrn == "T" and grd:
        payload["typhoon_grade"] = grd
    
    # 시간 정보 구성
    time_bits = []
    if payload["window_start_kst"] != "N/A" and payload["window_end_kst"] != "N/A":
        time_bits.append(f"기간: {payload['window_start_kst']} ~ {payload['window_end_kst']}")
    elif payload["window_start_kst"] != "N/A":
        time_bits.append(f"시각: {payload['window_start_kst']}")
    if payload["announce_time_kst"]:
        time_bits.append(f"발표시각: {payload['announce_time_kst']}")
    
    # 인간이 읽기 쉬운 형태로 구성
    parts = [
        f"지역: {region_name} ({reg_token})",
        *time_bits,
        f"현상: {payload['hazard_name']}({payload['hazard_code']})",
        f"수준: {payload['level_name']}({payload['level_code']})",
        f"명령: {payload['command_name']}({payload['command_code']})"
    ]
    
    if "typhoon_grade" in payload:
        parts.append(f"태풍 등급: {payload['typhoon_grade']}")
    
    return {
        "json": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "human": " | ".join(parts)
    }

def fetch_kma_advisories(start_time: str, end_time: str, disp: str = "1") -> List[Dict[str, str]]:
    """KMA 기상특보 데이터 가져오기"""
    if not WHEATHER_API_KEY_HUB:
        print("❌ API 키(WHEATHER_API_KEY_HUB)가 없습니다.")
        return []
    
    base = "https://apihub.kma.go.kr/api/typ01/url/wrn_met_data.php"
    params = {
        "authKey": WHEATHER_API_KEY_HUB,
        "wrn": "",
        "tmfc1": start_time,
        "tmfc2": end_time,
        "disp": disp
    }
    
    for retry in range(3):  # 최대 3번 재시도
        try:
            r = requests.get(base, params=params, timeout=KMA_TIMEOUT)
            r.raise_for_status()
            text = r.content.decode("euc-kr", errors="ignore")
            
            docs, seen = [], set()
            for line in [ln for ln in text.strip().split("\n") if ln.strip() and not ln.startswith("#") and not ln.startswith("7777END")]:
                raw = line.strip().rstrip("=").split(",")
                if len(raw) < 9:
                    continue
                
                rec = _format_kma_record(raw)
                key = re.sub(r"\s+", " ", rec["json"]).strip()
                if key in seen:
                    continue
                seen.add(key)
                docs.append(rec)
            
            return docs
            
        except Exception as e:
            print(f"❌ 기상특보 API 오류 (시도 {retry+1}/3): {e}")
            if retry < 2:  # 마지막 시도가 아니면 잠시 대기
                import time
                time.sleep(2)
            else:
                print(f"❌ 기상특보 API 최종 실패")
                return []

class AdvisoryNode:
    """기상특보 노드 클래스"""
    
    def __init__(self):
        self.name = "advisory_node"
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """기상특보 노드 실행"""
        print("🧩 노드: 기상특보 데이터 수집")
        
        try:
            # 현재 시간 기준으로 특보 데이터 가져오기
            now = datetime.now(tz=KST)
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M")
            end_time = now.strftime("%Y%m%d%H%M")
            
            # 특보 데이터 가져오기
            advisories = fetch_kma_advisories(start_time, end_time)
            
            # 상태에 결과 저장
            if "advisory_data" not in state:
                state["advisory_data"] = []
            state["advisory_data"].extend(advisories)
            
            print(f"   - ✅ 기상특보 데이터 수집 완료: {len(advisories)}개")
            
        except Exception as e:
            print(f"   - ❌ 기상특보 데이터 수집 실패: {e}")
            if "advisory_data" not in state:
                state["advisory_data"] = []
        
        return state