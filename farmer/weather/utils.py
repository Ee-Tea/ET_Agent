# -*- coding: utf-8 -*-
"""
공통 유틸리티 함수들
날씨 에이전트에서 사용하는 공통 함수들
"""

import os
import re
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()

# 환경 설정
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
KST = ZoneInfo("Asia/Seoul")

# 임베딩 모델 지연 로딩
_text_embedder = None

def get_text_embedder():
    """임베딩 모델 지연 로딩"""
    global _text_embedder
    if _text_embedder is None:
        print("   - 🔄 임베딩 모델 로딩 중...")
        _text_embedder = SentenceTransformer(EMBED_MODEL_NAME)
        print("   - ✅ 임베딩 모델 로딩 완료")
    return _text_embedder

# 지역 매핑 (CSV 파일에서 로드)
REGION_MAP = {}

def _norm_name(s: str) -> str:
    """지역명 정규화"""
    s = str(s or "").strip()
    s = re.sub(r"[()\s·ㆍ]", "", s)  # 공백/특수문자 제거
    return s

def load_region_map():
    """CSV 파일에서 지역 매핑 로드 (정교한 방식)"""
    global REGION_MAP
    csv_path = os.getenv("REGION_CSV_PATH", "farmer/all_regions_combined.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV 파일이 없습니다: {csv_path}")
        REGION_MAP = {}
        return
    
    try:
        import pandas as pd
        from collections import defaultdict
        
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # 컬럼 자동 감지(대소문자 무시)
        cols_upper = {c.upper(): c for c in df.columns}
        code_col = next((cols_upper[k] for k in ("REG_ID", "CODE", "ID") if k in cols_upper), None)
        name_col = next((cols_upper[k] for k in ("REG_NAME", "NAME", "REGION_NAME") if k in cols_upper), None)
        
        if not code_col or not name_col:
            print("❌ CSV에서 코드/이름 컬럼을 찾을 수 없습니다.")
            REGION_MAP = {}
            return
        
        code_to_name: Dict[str, str] = {}
        name_to_codes: Dict[str, List[str]] = defaultdict(list)
        norm_name_to_codes: Dict[str, List[str]] = defaultdict(list)
        
        seen_pairs = set()  # (code, name) 중복 제거용
        
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            if not code or not name or code.lower() == "nan" or name.lower() == "nan":
                continue
            
            # (code, name) 페어 중복 방지
            if (code, name) in seen_pairs:
                continue
            seen_pairs.add((code, name))
            
            # 1) code -> name (1:1) : 첫 값 유지
            if code not in code_to_name:
                code_to_name[code] = name
            elif code_to_name[code] != name:
                pass
            
            # 2) name -> [codes] (1:多)
            if code not in name_to_codes[name]:
                name_to_codes[name].append(code)
            
            # 3) 정규화 이름 -> [codes]
            nname = _norm_name(name)
            if code not in norm_name_to_codes[nname]:
                norm_name_to_codes[nname].append(code)
        
        # REGION_MAP에 양방향 매핑 저장 (기존 호환성 유지)
        REGION_MAP = {}
        for code, name in code_to_name.items():
            REGION_MAP[name] = code  # 지역명 → 코드
            REGION_MAP[code] = name  # 코드 → 지역명
        
        print(f"✅ 지역 매핑 로드 완료: {len(code_to_name)}개")
        
    except Exception as e:
        print(f"❌ 지역 매핑 로드 실패: {e}")
        REGION_MAP = {}

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2 정규화"""
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

def embed_texts(texts: List[str]) -> np.ndarray:
    """텍스트 임베딩 처리"""
    if not texts:
        return np.array([], dtype="float32").reshape(0, 768)
    
    print(f"   - 🔄 임베딩 처리: {len(texts)}개 텍스트")
    
    try:
        # 지연 로딩된 임베딩 모델 사용
        embedder = get_text_embedder()
        embeddings = embedder.encode(texts, show_progress_bar=False)
        # L2 정규화 적용
        embeddings = np.array([l2_normalize(e) for e in embeddings], dtype="float32")
        print(f"   - ✅ 임베딩 완료: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"   - ❌ 임베딩 실패: {e}")
        return np.zeros((len(texts), 768), dtype="float32")

def minmax_norm(scores: List[float]) -> List[float]:
    """Min-Max 정규화"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-8:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

def _format_for_llm(src: str, payload_json: str, human: str) -> str:
    """LLM용 포맷팅"""
    try:
        p = json.loads(payload_json)
    except:
        return f"[{src}] {human}"
    
    if src == "live_advisory":
        line = (
            f"[{src}] 지역:{p.get('region_name','N/A')} | 현상:{p.get('hazard_name','N/A')} "
            f"| 수준:{p.get('level_name','N/A')} | 발표:{p.get('window_start_kst','N/A')} | 해제:{p.get('window_end_kst','N/A')}"
        )
        return f"{line}\n[NOTE] {human}"
        
    elif src == "live_forecast":
        temp = p.get("temp","N/A")
        prob = p.get("precip_prob","N/A")
        sky = p.get("sky_status","N/A")
        
        line = (
            f"[{src}] 지역:{p.get('region_name','N/A')} | 시각:{p.get('forecast_time','N/A')} "
            f"| 하늘:{sky} | 기온:{temp} | 강수확률:{prob}"
        )
        return f"{line}\n[NOTE] {human}"
        
    elif src == "region_forecast":
        line = (
            f"[{src}] 권역코드:{p.get('region_id','N/A')} | 지역:{p.get('region_name','N/A')} | 대상:{p.get('forecast_time','N/A')} "
            f"| 하늘:{(p.get('sky_text') or 'N/A')} | 강수형:{(p.get('precip_type') or 'N/A')} | 강수확률:{(p.get('precip_prob') or 'N/A')}"
        )
        return f"{line}\n[NOTE] {human}"
        
    return f"[{src}] {human}"

def search_similar_documents(query: str, documents: List[Dict[str, str]], top_k: int = 5) -> List[tuple]:
    """유사도 검색"""
    if not documents:
        return []
    
    human_texts = [doc["human"] for doc in documents]
    doc_embeddings = embed_texts(human_texts)
    query_embedding = embed_texts([query])
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    scores, indices = index.search(query_embedding, min(top_k, len(documents)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((float(score), documents[idx]))
    return results

def combine_weather_data(state: Dict[str, Any], max_docs_per_type: int = 3) -> str:
    """날씨 데이터 통합 (토큰 제한 고려)
       - 주간 질의(이번 주/다음 주 등)인 경우, max_docs_per_type를 자동 해제(=큰 값)"""
    context_parts = []

    # 주간 요청 여부 플래그
    weekly_mode = bool(state.get("is_weekly_request", False))
    _limit = (999 if weekly_mode else max_docs_per_type)

    # 기상특보 데이터
    if "advisory_data" in state and state["advisory_data"]:
        for doc in state["advisory_data"][:_limit]:
            formatted = _format_for_llm("live_advisory", doc["json"], doc["human"])
            context_parts.append(formatted)
    
    # 단기예보 데이터
    if "short_forecast_data" in state and state["short_forecast_data"]:
        for doc in state["short_forecast_data"][:_limit]:
            formatted = _format_for_llm("live_forecast", doc["json"], doc["human"])
            context_parts.append(formatted)
    
    # 중기예보 데이터
    if "mid_forecast_data" in state and state["mid_forecast_data"]:
        for doc in state["mid_forecast_data"][:_limit]:
            formatted = _format_for_llm("region_forecast", doc["json"], doc["human"])
            context_parts.append(formatted)
    
    if context_parts:
        return "\n\n".join(context_parts)
    else:
        return "NO_DATA_AVAILABLE"

# 지역 매핑 (CSV 파일에서 로드)
REGION_MAP = {}  # NOTE: 위에서 이미 정의했지만, 기존 코드 호환 위해 유지
def match_region_with_default(target_region: str, region_name: str, region_from_default: bool) -> bool:
    """
    지역 매칭 유틸 함수
    - target_region: 사용자가 선택한 지역 (예: "서울")
    - region_name: API 응답의 지역명
    - region_from_default: True면 지역 미지정으로 자동 기본값("서울") 적용된 경우, False면 사용자가 명시적으로 입력한 경우
    """
    if target_region == "서울":
        if region_from_default:
            # ✅ 지역 미지정 → 수도권까지 허용
            return any(k in region_name for k in ["서울", "경기", "인천", "수도권"])
        else:
            # ✅ 명시적으로 "서울" 입력 → 서울만
            return "서울" in region_name
    else:
        return target_region in region_name

# ========= 날짜/기간 추출 유틸 =========

def normalize_spaces(s: str) -> str:
    """문자열의 공백을 정규화"""
    return re.sub(r'\s+', ' ', (s or "").strip())

def now_kst() -> datetime:
    """현재 KST 시간 반환"""
    return datetime.now(tz=KST)

WEEKDAY_IDX = {"월":0,"화":1,"수":2,"목":3,"금":4,"토":5,"일":6}

def _week_range_containing(d: date) -> Tuple[date, date]:
    """일요일~토요일 주간 범위로 맞춤"""
    # 일요일=6? 파이썬 weekday(): 월=0 ... 일=6
    # '이번 주: 일요일~토요일' 요구에 맞추기 위해, 해당 주의 '일요일'을 구함
    # 현재 날짜 d 기준: d_weekday (월0~일6)
    wd = d.weekday()  # 월=0 ... 일=6
    # 우리 기준: 주 시작은 '일요일'이므로, d가 월(0)이면 -1일, ... 일(6)이면 0일
    days_from_sun = (wd + 1) % 7  # 일요일이면 0, 월요일이면 1, ...
    start = d - timedelta(days=days_from_sun)
    end = start + timedelta(days=6)
    return (start, end)



def REGION_extract_date_range_from_question(q: str) -> Optional[Tuple[datetime, datetime, bool]]:
    """
    질문에서 날짜 '범위'를 파싱.
    반환: (start_dt, end_dt, is_weekly_request)
      - is_weekly_request=True면 '이번 주/다음 주' 같은 포괄 질의
    """
    qn = normalize_spaces(q)
    now = now_kst()

    # "이번 주"
    if "이번주" in qn or "이번 주" in qn:
        s, e = _week_range_containing(now.date())
        sdt = datetime.combine(s, datetime.min.time(), tzinfo=KST)
        edt = datetime.combine(e, datetime.max.time(), tzinfo=KST)
        return (sdt, edt, True)

    # "다음 주"
    if "다음주" in qn or "다음 주" in qn:
        # 이번 주의 끝 + 1일 → 다음 주의 시작
        s, e = _week_range_containing(now.date())
        next_start = e + timedelta(days=1)
        next_end = next_start + timedelta(days=6)
        sdt = datetime.combine(next_start, datetime.min.time(), tzinfo=KST)
        edt = datetime.combine(next_end, datetime.max.time(), tzinfo=KST)
        return (sdt, edt, True)

    # "오늘/내일/모레"는 단일 날짜 → 동일 start/end
    if any(k in qn for k in ["오늘", "현재", "지금"]):
        sdt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        edt = now.replace(hour=23, minute=59, second=59, microsecond=0)
        return (sdt, edt, False)
    if "내일" in qn:
        tgt = (now + timedelta(days=1))
        sdt = tgt.replace(hour=0, minute=0, second=0, microsecond=0)
        edt = tgt.replace(hour=23, minute=59, second=59, microsecond=0)
        return (sdt, edt, False)
    if "모레" in qn:
        tgt = (now + timedelta(days=2))
        sdt = tgt.replace(hour=0, minute=0, second=0, microsecond=0)
        edt = tgt.replace(hour=23, minute=59, second=59, microsecond=0)
        return (sdt, edt, False)

    # M월D일 패턴 (단일 날짜)
    m2 = re.search(r"(\d{1,2})월(\d{1,2})일", qn)
    if m2:
        M, D = int(m2.group(1)), int(m2.group(2))
        try:
            tgt = datetime(now.year, M, D, tzinfo=KST)
            sdt = tgt.replace(hour=0, minute=0, second=0, microsecond=0)
            edt = tgt.replace(hour=23, minute=59, second=59, microsecond=0)
            return (sdt, edt, False)
        except ValueError:
            pass

    # N일/주/달/년 뒤 → 단일 날짜 취급
    m = re.search(r"(\d+)(일|주|달|년)(뒤|후)", qn)
    if m:
        num, unit = int(m.group(1)), m.group(2)
        if unit == "일":
            tgt = now + timedelta(days=num)
        elif unit == "주":
            tgt = now + timedelta(weeks=num)
        elif unit == "달":
            tgt = now + timedelta(days=30*num)
        else:
            tgt = now + timedelta(days=365*num)
        sdt = tgt.replace(hour=0, minute=0, second=0, microsecond=0)
        edt = tgt.replace(hour=23, minute=59, second=59, microsecond=0)
        return (sdt, edt, False)

    # 못 찾으면 None
    return None

# (기존) 단일 datetime 추출 함수(하위 호환)
def REGION_extract_datetime_from_question(q: str) -> Optional[datetime]:
    rng = REGION_extract_date_range_from_question(q)
    if not rng:
        return None
    sdt, _, _ = rng
    return sdt
