# -*- coding: utf-8 -*-
"""
단기예보 노드 모듈
KMA 단기육상예보 데이터를 가져오는 노드
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

# 환경 설정
KMA_TIMEOUT = int(os.getenv("KMA_TIMEOUT", "30"))
WHEATHER_API_KEY_HUB = os.getenv("WHEATHER_API_KEY_HUB")
KST = ZoneInfo("Asia/Seoul")

# 단기예보 매핑
SKY_MAP = {"DB01": "맑음", "DB02": "구름조금", "DB03": "구름많음", "DB04": "흐림"}
PREP_MAP = {"0": "없음", "1": "비", "2": "비/눈", "3": "눈", "4": "눈/비"}
WIND_KO = {
    "N": "북", "NNE": "북북동", "NE": "북동", "ENE": "동북동", "E": "동", "ESE": "동남동",
    "SE": "남동", "SSE": "남남동", "S": "남", "SSW": "남남서", "SW": "남서", "WSW": "서남서",
    "W": "서", "WNW": "서북서", "NW": "북서", "NNW": "북북서"
}

# 지역 매핑
from .utils import load_region_map, REGION_MAP
load_region_map()

def resolve_region(token: str) -> str:
    """지역 코드를 지역명으로 변환"""
    if not token:
        return "N/A"
    return REGION_MAP.get(token.strip(), token)

def fmt_kst(yyyymmddHHMM: str) -> str:
    """KST 시간 포맷팅"""
    try:
        dt = datetime.strptime(yyyymmddHHMM, "%Y%m%d%H%M").replace(tzinfo=KST)
        return dt.strftime("%Y-%m-%d %H:%M KST")
    except Exception:
        return yyyymmddHHMM

def _format_short_land_record(raw: list) -> Dict[str, str]:
    """단기육상예보 레코드 포맷팅"""
    def g(i):
        return raw[i] if i < len(raw) else ""
    
    reg_id = g(0)
    tm_fc = g(1)
    tm_ef = g(2)
    mod = g(3)
    ne = g(4)
    w1, w2 = g(9), g(11)
    ta, st = g(12), g(13)
    sky, prep, wf = g(14), g(15), g(16)
    
    reg_name = resolve_region(reg_id)
    
    payload = {
        "source": "KMA_SHORT_LAND",
        "region_id": reg_id,
        "region_name": reg_name,
        "forecast_time": fmt_kst(tm_fc) if tm_fc else "N/A",
        "effective_time": fmt_kst(tm_ef) if tm_ef else "N/A",
        "temp": f"{ta}°C" if ta and ta != "-99" else "N/A",
        "precip_prob": f"{st}%" if st else "N/A",
        "sky_status": SKY_MAP.get(sky, sky),
        "precip_status": PREP_MAP.get(prep, prep),
        "wind_direction": f"{WIND_KO.get(w1, w1)}~{WIND_KO.get(w2, w2)}" if w1 and w2 else WIND_KO.get(w1, w1)
    }
    
    human = f"{reg_name} 단기예보 — {payload['forecast_time']} 발표, {payload['effective_time']} 효력: {payload['sky_status']}, 기온 {payload['temp']}, 강수 {payload['precip_prob']}"
    if payload['wind_direction']:
        human += f", 바람 {payload['wind_direction']}"
    
    return {
        "json": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "human": human
    }

def fetch_short_land_records(question_date=None, target_region=None) -> List[Dict[str, str]]:
    """KMA 단기육상예보 데이터 가져오기 (지역 필터링 적용)"""
    if not WHEATHER_API_KEY_HUB:
        print("❌ API 키(WHEATHER_API_KEY_HUB)가 없습니다.")
        return []
    
    # 날짜 처리
    if question_date:
        tmfc = question_date.strftime("%Y%m%d%H")
        print(f"   - 질문 날짜 사용: {tmfc}")
    else:
        now = datetime.now(tz=KST)
        tmfc = now.strftime("%Y%m%d%H")
        print(f"   - 현재 날짜 사용: {tmfc}")
    
    BASE = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_dl.php"
    params = {
        "reg": "",
        "tmfc": tmfc,
        "disp": "1",
        "authKey": WHEATHER_API_KEY_HUB
    }
    
    for retry in range(3):  # 최대 3번 재시도
        try:
            r = requests.get(f"{BASE}?{urlencode(params)}", timeout=KMA_TIMEOUT)
            r.raise_for_status()
            text = r.content.decode("euc-kr", errors="replace")
            
            docs = []
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("7777END"):
                    continue
                if s.endswith("="):
                    s = s[:-1]
                
                raw_row = [c.strip() for c in s.split(",")]
                if len(raw_row) < 17:
                    continue
                
                formatted_record = _format_short_land_record(raw_row)
                
                # 지역 필터링 (API 호출 시점에서)
                if target_region:
                    try:
                        j = json.loads(formatted_record["json"])
                        region_name = j.get("region_name", "")
                        
                        # 서울의 경우 수도권 포함
                        if target_region == "서울":
                            if any(k in region_name for k in ["서울", "경기", "인천", "수도권"]):
                                docs.append(formatted_record)
                        else:
                            if target_region in region_name:
                                docs.append(formatted_record)
                    except:
                        continue
                else:
                    docs.append(formatted_record)
            
            print(f"   - API에서 {len(docs)}개 데이터 수집 (지역: {target_region or '전체'})")
            return docs
            
        except Exception as e:
            print(f"❌ 단기예보 API 오류 (시도 {retry+1}/3): {e}")
            if retry < 2:
                import time
                time.sleep(2)
            else:
                print(f"❌ 단기예보 API 최종 실패")
                return []

class ShortForecastNode:
    """단기예보 노드 클래스"""
    
    def __init__(self):
        self.name = "short_forecast_node"
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """단기예보 노드 실행"""
        try:
            question_date = state.get("question_date")
            target_region = state.get("target_region", "서울")
            date_range = state.get("question_date_range")  # (start_dt, end_dt, is_weekly_request)

            if question_date:
                print(f"   - 질문 날짜: {question_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"   - 대상 지역: {target_region}")

            # 기본 1회 호출
            aggregate: List[Dict[str, str]] = fetch_short_land_records(question_date, target_region)

            # 주간 범위가 있으면: 해당 범위 날짜 전체 포함(시간 미세 필터는 아래에서 날짜기준으로)
            if date_range:
                sdt, edt, _ = date_range
                # 추가 호출은 하지 않고, 이미 받은 doc들 중에서 날짜로 필터 (dl가 시간대별 제공)
                # 필요시 tmfc를 바꾸어 여러 번 호출할 수도 있으나, 현재 파이프라인과 비용 고려하여 1회 + 날짜필터로 처리
                pass

            # 상태에 결과 저장
            state.setdefault("short_forecast_data", [])
            state["short_forecast_data"].extend(aggregate)
            print(f"   - ✅ 단기예보 데이터 수집 완료: {len(aggregate)}개")

        except Exception as e:
            print(f"   - ❌ 단기예보 데이터 수집 실패: {e}")
            state.setdefault("short_forecast_data", [])
        return state
