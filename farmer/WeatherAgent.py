# -*- coding: utf-8 -*-
"""
통합 기상 전문가 그래프:
- 라이브 특보(wrn_met_data.php) + 단기육상예보(fct_afs_dl.php)
- 권역 육상예보 주간(fct_afs_wl.php) [CSV 권역 매핑 기반]  ← 통합 추가
- API-only retrieve (벡터DB 미사용, in-memory FAISS 유사도)
- 2차 검증 노드 제거
- 예보관 톤(개요/상세/영향-위험도/권고) 프롬프트
- 인터넷 연결 상태 자동 확인

.env 설정:
  WHEATHER_API_KEY_HUB=REDACTED     # (필수) 기상청 API 키
  KMA_TIMEOUT=30               # (선택) API 타임아웃 초 (기본값: 30초, 타임아웃 오류 시 증가 권장)
  GROQ_API_KEY=REDACTED API 키
  TAVILY_API_KEY=REDACTED           # (선택) 웹 검색용

네트워크 연결 확인:
  - 인터넷 연결 상태 자동 확인 (DNS 서버 연결 테스트)
  - 기상청 API 서버 연결 상태 확인
  - 연결 실패 시 적절한 오류 메시지 표시
"""

import os
import re
import json
import csv
import time
import socket
from typing import TypedDict, Optional, Any, Dict, List, Tuple
from datetime import datetime, timedelta
from operator import itemgetter
from urllib.parse import urlencode
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import faiss
import requests
from dotenv import load_dotenv

# LangChain / LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

# ===== (신규) RAGAS 관련 Import (0.3.x 호환) =====
_HAS_RAGAS = False
try:
    from ragas import evaluate, SingleTurnSample
    from ragas.metrics import ResponseRelevancy, Faithfulness
    from ragas.metrics import LLMContextPrecisionWithoutReference
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    # RAGAS 0.3.x에서는 직접 LangChain 객체를 전달
    from datasets import Dataset
    _HAS_RAGAS = True
except ImportError as e:
    print(f"   - ⚠️ RAGAS/의존성 임포트 실패: {e}")

# torch는 선택 사항
try:
    import torch
    print("   - 🚀 GPU 가속 활성화 (RAGAS)" if torch.cuda.is_available() else "   - 💻 CPU 모드 (RAGAS)")
except Exception:
    torch = None
    print("   - ℹ️ torch 미설치: CPU 모드 (RAGAS)")

load_dotenv()

# =========[ 최적화 설정 ]=========
ENABLE_PARALLEL_PROCESSING = os.getenv("ENABLE_PARALLEL_PROCESSING", "true").lower() in ("1", "true", "yes")
ENABLE_CONDITIONAL_API_CALLS = os.getenv("ENABLE_CONDITIONAL_API_CALLS", "true").lower() in ("1", "true", "yes")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))

# =========[ 공통 환경설정 ]=========
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
KMA_TIMEOUT = int(os.getenv("KMA_TIMEOUT", "30"))
WHEATHER_API_KEY_HUB=REDACTED("WHEATHER_API_KEY_HUB")
USE_KMA_LIVE = os.getenv("USE_KMA_LIVE", "true").lower() in ("1", "true", "yes")
FORCE_DISABLE_KMA_LIVE = False

# 통합 지역코드 CSV 설정
UNIFIED_REGIONS_CSV = os.getenv("UNIFIED_REGIONS_CSV", "all_regions_combined.csv")

TAVILY_API_KEY=REDACTED("TAVILY_API_KEY")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# =========[ RAGAS 백엔드 설정 ]=========
RAGAS_BACKEND = os.getenv("RAGAS_BACKEND", "openai").lower()
RAGAS_OPENAI_LLM = os.getenv("RAGAS_OPENAI_LLM", "gpt-4o-mini")
RAGAS_OPENAI_EMB = os.getenv("RAGAS_OPENAI_EMB", "BAAI/bge-m3")

_RAGAS_LLM = None
_RAGAS_EMB = None
_RAGAS_LLM_WRAPPER = None
_RAGAS_EMB_WRAPPER = None

def _init_ragas_backend():
    """RAGAS LLM/Embedding 백엔드 초기화. OpenAI LLM + HuggingFace Embeddings 사용."""
    global _RAGAS_LLM, _RAGAS_EMB, RAGAS_BACKEND
    if not _HAS_RAGAS:
        return

    try:
        if not OPENAI_API_KEY=REDACTED("   - ⚠️ OPENAI_API_KEY=REDACTED 비활성화")
            return
        
        ***REMOVED*** LLM 설정 - RAGAS faithfulness 향상을 위한 설정
        llm = ChatOpenAI(
            model_name=RAGAS_OPENAI_LLM, 
            temperature=TEMPERATURE,  # 환경설정에서 가져온 temperature 사용
            api_key=OPENAI_API_KEY=REDACTED JSON 파싱 오류 방지를 위한 설정
            max_tokens=4000,
            top_p=0.3,  # 0.1 → 0.3으로 완화 (더 다양한 응답 허용)
            # JSON 출력 안정성을 위한 추가 설정
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        # HuggingFace 임베딩 사용
        emb = HuggingFaceEmbeddings(
            model_name=RAGAS_OPENAI_EMB,
            encode_kwargs={"normalize_embeddings": True}
        )
        _RAGAS_LLM = llm
        _RAGAS_EMB = emb
        
        # RAGAS Wrapper 설정 (SalesRAGAS 방식)
        global _RAGAS_LLM_WRAPPER, _RAGAS_EMB_WRAPPER
        _RAGAS_LLM_WRAPPER = LangchainLLMWrapper(_RAGAS_LLM)
        _RAGAS_EMB_WRAPPER = LangchainEmbeddingsWrapper(_RAGAS_EMB)
        
        print(f"   - 🔑 RAGAS 백엔드=OpenAI LLM + HF Embeddings · LLM={RAGAS_OPENAI_LLM}, EMB={RAGAS_OPENAI_EMB}")
    except Exception as e:
        print(f"   - ⚠️ RAGAS 백엔드 초기화 실패: {e}")

_init_ragas_backend()

KST = ZoneInfo("Asia/Seoul")

# =========[ 인터넷 연결 확인 ]=========
def check_internet_connection() -> bool:
    """인터넷 연결 상태를 확인합니다."""
    try:
        # DNS 조회로 연결 확인 (Google DNS 사용)
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        try:
            # 백업: 다른 DNS 서버 시도
            socket.create_connection(("1.1.1.1", 53), timeout=3)
            return True
        except OSError:
            return False

def check_kma_server_connectivity() -> bool:
    """기상청 API 서버 연결 상태를 확인합니다."""
    try:
        # 기상청 API 서버 연결 확인
        socket.create_connection(("apihub.kma.go.kr", 443), timeout=5)
        return True
    except OSError:
        return False

# =========[ 라이브(특보/단기) 매핑/유틸 ]=========
WRN_MAP = {"T":"태풍","W":"강풍","R":"호우","C":"한파","D":"건조","O":"해일","N":"지진해일","V":"풍랑","S":"대설","Y":"황사","H":"폭염","F":"안개"}
LVL_MAP = {"1":"예비특보","2":"주의보","3":"경보"}
CMD_MAP = {"1":"발표","2":"대치","3":"해제","4":"대치해제","5":"연장","6":"변경","7":"변경해제"}
REGION_CODE_RE = re.compile(r"^[A-Z]\d{7}$")
REGION_MAP: Dict[str, str] = {}
REGION_NAME_INDEX: Dict[str, List[str]] = {}
_text_embedder = None

# 단기예보 매핑
SKY_MAP = {"DB01": "맑음", "DB02": "구름조금", "DB03": "구름많음", "DB04": "흐림"}
PREP_MAP = {"0": "없음", "1": "비", "2": "비/눈", "3": "눈", "4": "눈/비"}
WIND_KO = {"N":"북","NNE":"북북동","NE":"북동","ENE":"동북동","E":"동","ESE":"동남동","SE":"남동","SSE":"남남동","S":"남","SSW":"남남서","SW":"남서","WSW":"서남서","W":"서","WNW":"서북서","NW":"북서","NNW":"북북서"}

def _norm_name(s: str) -> str:
    return re.sub(r"[()\s·ㆍ]", "", str(s or "")).lower()

def _pick_cols(df: pd.DataFrame) -> tuple:
    cols = {c.lower(): c for c in df.columns}
    code_col = next((cols[k] for k in ("code","region_code","지역코드","reg_id","regid","id") if k in cols), None)
    name_col = next((cols[k] for k in ("name","region_name","지역명","reg_name","regname","ko_name","한글명") if k in cols), None)
    if not code_col:
        best, hits = None, -1
        for c in df.columns:
            cnt = sum(bool(re.match(r"^[A-Za-z]?\d{5,}$", str(v).strip())) for v in df[c].astype(str))
            if cnt > hits: best, hits = c, cnt
        code_col = best
    if not name_col:
        best, hits = None, -1
        for c in df.columns:
            cnt = sum(bool(re.search(r"[가-힣]", str(v))) for v in df[c].astype(str))
            if cnt > hits: best, hits = c, cnt
        name_col = best
    return code_col, name_col

def _read_map_csv(path: Optional[str]) -> List[tuple]:
    if not path or not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    code_col, name_col = _pick_cols(df)
    pairs: List[tuple] = []
    for _, r in df.iterrows():
        code = str(r[code_col]).strip() if code_col else ""
        name = str(r[name_col]).strip() if name_col else ""
        if not code or not name or code.lower() == "nan" or name.lower() == "nan":
            continue
        pairs.append((code, name))
    return pairs

def _load_unified_region_map():
    """통합 CSV에서 지역코드 매핑 로드"""
    REGION_MAP.clear()
    REGION_NAME_INDEX.clear()
    
    if not os.path.exists(UNIFIED_REGIONS_CSV):
        print(f"⚠️ 통합 CSV 파일이 없습니다: {UNIFIED_REGIONS_CSV}")
        return
    
    try:
        # pandas로 읽기 (더 안정적)
        df = pd.read_csv(UNIFIED_REGIONS_CSV, encoding='utf-8-sig')
        
        # 컬럼 자동 감지
        code_col, name_col = _pick_cols(df)
        
        if not code_col or not name_col:
            print("❌ CSV에서 코드/이름 컬럼을 찾을 수 없습니다.")
            return
        
        print(f"✅ CSV 컬럼 감지: 코드={code_col}, 이름={name_col}")
        
        # 데이터 로드
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            
            if code and name and code.lower() != "nan" and name.lower() != "nan":
                REGION_MAP[code] = name
                REGION_NAME_INDEX.setdefault(_norm_name(name), []).append(code)
        
        print(f"✅ 통합 CSV 로드 완료: {len(REGION_MAP)}개 지역 (파일: {UNIFIED_REGIONS_CSV})")
        
    except Exception as e:
        print(f"❌ 통합 CSV 로드 실패: {e}")

_load_unified_region_map()

def resolve_region(token: str) -> str:
    if not token: return "N/A"
    t = token.strip()
    return REGION_MAP.get(t, t)

def fmt_kst(yyyymmddHHMM: str) -> str:
    try:
        dt = datetime.strptime(yyyymmddHHMM, "%Y%m%d%H%M").replace(tzinfo=KST)
        return dt.strftime("%Y-%m-%d %H:%M KST")
    except Exception:
        return yyyymmddHHMM

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

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

def minmax_norm(scores: List[float]) -> List[float]:
    if not scores: return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-8: return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

# =========[ 복합 질문 처리 함수들 ]=========
def extract_multiple_dates(q: str) -> List[datetime]:
    """복합 질문에서 여러 날짜 추출"""
    dates = []
    qn = re.sub(r"\s+", "", q or "").lower()
    now = datetime.now(tz=KST)
    
    # 날짜 패턴 매칭
    patterns = [
        ("오늘", lambda: now),
        ("내일", lambda: now + timedelta(days=1)),
        ("모레", lambda: now + timedelta(days=2)),
        ("글피", lambda: now + timedelta(days=3)),
    ]
    
    for keyword, date_func in patterns:
        if keyword in qn:
            dates.append(date_func())
    
    # 숫자 + 일 패턴
    m = re.search(r"(\d+)(일|주|달|년)(뒤|후)", qn)
    if m:
        num, unit = int(m.group(1)), m.group(2)
        if unit == "일":
            dates.append(now + timedelta(days=num))
        elif unit == "주":
            dates.append(now + timedelta(weeks=num))
        elif unit == "달":
            dates.append(now + timedelta(days=30*num))
        elif unit == "년":
            dates.append(now + timedelta(days=365*num))
    
    # 다음주 패턴
    if "다음주" in qn:
        days_to_next_mon = ((7 - now.weekday()) % 7) or 7
        dates.append(now + timedelta(days=days_to_next_mon))
    
    # 월일 패턴
    m2 = re.search(r"(\d{1,2})월(\d{1,2})일", qn)
    if m2:
        M, D = int(m2.group(1)), int(m2.group(2))
        try:
            dates.append(datetime(now.year, M, D, tzinfo=KST))
        except ValueError:
            pass
    
    return list(dict.fromkeys(dates))  # 중복 제거

def extract_multiple_regions(q: str) -> List[str]:
    """복합 질문에서 여러 지역 추출"""
    regions = []
    qn = re.sub(r"\s+", "", q or "").lower()
    
    # 지역명 매칭 (REGION_MAP이 로드된 후에 사용)
    if 'REGION_MAP' in globals():
        for code, name in REGION_MAP.items():
            if name in q:
                regions.append(name)
    
    # 일반적인 지역명
    common_regions = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "제주"]
    for region in common_regions:
        if region in q:
            regions.append(region)
    
    return list(dict.fromkeys(regions))  # 중복 제거

def decompose_complex_question(q: str) -> List[Dict[str, Any]]:
    """복합 질문을 단순 질문들로 분해"""
    dates = extract_multiple_dates(q)
    regions = extract_multiple_regions(q)
    
    # 기본값 설정
    if not dates:
        dates = [datetime.now(tz=KST)]
    if not regions:
        regions = ["서울"]
    
    # 모든 조합 생성
    sub_questions = []
    for date in dates:
        for region in regions:
            sub_questions.append({
                "date": date,
                "region": region,
                "question": f"{region} {date.strftime('%m월 %d일')} 날씨",
                "days_from_today": (date.date() - datetime.now(tz=KST).date()).days
            })
    
    return sub_questions

# =========[ 조건부 API 호출 함수들 ]=========
def should_fetch_advisories(q: str, days_from_today: int) -> bool:
    """특보 데이터를 가져올지 결정"""
    if not ENABLE_CONDITIONAL_API_CALLS:
        return True
    
    # 4일 이후면 특보 의미 없음
    if days_from_today >= 4:
        return False
    
    # 오늘 질문이 아니면 특보 의미 없음
    if days_from_today > 0:
        return False
    
    # 특보 관련 키워드가 있어야 함
    advisory_keywords = ["특보", "주의보", "경보", "해제", "발표", "현재", "지금"]
    return any(keyword in q for keyword in advisory_keywords)

def should_fetch_forecasts(q: str, days_from_today: int) -> bool:
    """단기예보 데이터를 가져올지 결정"""
    if not ENABLE_CONDITIONAL_API_CALLS:
        return True
    
    # 4일 이후면 단기예보 의미 없음
    if days_from_today >= 4:
        return False
    
    # 날씨 관련 키워드가 있어야 함
    weather_keywords = ["날씨", "기온", "강수", "하늘", "바람", "예보"]
    return any(keyword in q for keyword in weather_keywords)

def should_fetch_region_forecasts(q: str, days_from_today: int) -> bool:
    """권역예보 데이터를 가져올지 결정"""
    if not ENABLE_CONDITIONAL_API_CALLS:
        return True
    
    # 4일 이전이면 권역예보 의미 없음
    if days_from_today < 4:
        return False
    
    # 10일 이후면 권역예보 범위 벗어남
    if days_from_today > 10:
        return False
    
    # 권역예보 관련 키워드가 있어야 함
    region_keywords = ["권역", "주간", "장기", "전망", "예보"]
    return any(keyword in q for keyword in region_keywords)

# =========[ 병렬 처리 함수들 ]=========
def safe_api_call(func, *args, **kwargs):
    """안전한 API 호출을 위한 래퍼 함수"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"❌ API 호출 실패 ({func.__name__}): {e}")
        return None

def embed_texts_parallel(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """병렬로 임베딩 계산 (안전한 버전)"""
    # 병렬 처리 비활성화하거나 배치 크기가 작으면 순차 처리
    if not ENABLE_PARALLEL_PROCESSING or len(texts) < batch_size:
        return embed_texts(texts)
    
    # 간단하게 순차 처리로 폴백 (임베딩 모델의 thread-safety 문제 때문)
    print("   - ℹ️ 임베딩 순차 처리로 폴백")
    return embed_texts(texts)

def parallel_fetch_weather_data(sub_questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """병렬로 날씨 데이터 가져오기"""
    if not ENABLE_PARALLEL_PROCESSING:
        return fetch_weather_data_sequential(sub_questions)
    
    print("   - �� 병렬 처리로 데이터 수집 중...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        
        for i, sub_q in enumerate(sub_questions):
            q = sub_q["question"]
            date = sub_q["date"]
            region = sub_q["region"]
            days_from_today = sub_q["days_from_today"]
            
            # 라이브 데이터 (4일 이내일 때만)
            if days_from_today < 4:
                if should_fetch_advisories(q, days_from_today):
                    futures[f'advisories_{i}'] = executor.submit(
                        safe_api_call, 
                        fetch_kma_advisories, 
                        date.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M"),
                        date.strftime("%Y%m%d%H%M")
                    )
                
                if should_fetch_forecasts(q, days_from_today):
                    futures[f'forecasts_{i}'] = executor.submit(
                        safe_api_call, 
                        fetch_short_land_records
                    )
            
            # 권역 데이터 (4일 이후일 때만)
            if days_from_today >= 4 and should_fetch_region_forecasts(q, days_from_today):
                region_code = get_region_code(region)
                if region_code:
                    futures[f'region_forecasts_{i}'] = executor.submit(
                        safe_api_call,
                        fetch_mid_week_land,
                        reg_id=region_code,
                        target_date=date,
                        tmfc_range_days=3,
                        widen_days=0
                    )
        
        # 결과 수집
        results = {"advisories": [], "forecasts": [], "region_forecasts": []}
        for name, future in futures.items():
            try:
                result = future.result(timeout=API_TIMEOUT)
                if result:
                    if name.startswith('advisories_'):
                        results["advisories"].extend(result)
                    elif name.startswith('forecasts_'):
                        results["forecasts"].extend(result)
                    elif name.startswith('region_forecasts_'):
                        results["region_forecasts"].extend(result)
            except Exception as e:
                print(f"❌ 병렬 처리 실패 ({name}): {e}")
        
        return results

def fetch_weather_data_sequential(sub_questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """순차적으로 날씨 데이터 가져오기"""
    print("   - �� 순차 처리로 데이터 수집 중...")
    
    results = {"advisories": [], "forecasts": [], "mid_forecasts": []}
    
    for sub_q in sub_questions:
        q = sub_q["question"]
        date = sub_q["date"]
        region = sub_q["region"]
        days_from_today = sub_q["days_from_today"]
        
        # 라이브 데이터
        if days_from_today < 4:
            if should_fetch_advisories(q, days_from_today):
                advisories = fetch_kma_advisories(
                    date.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M"),
                    date.strftime("%Y%m%d%H%M")
                ) if WHEATHER_API_KEY_HUB else []
                results["advisories"].extend(advisories)
            
            if should_fetch_forecasts(q, days_from_today):
                forecasts = fetch_short_land_records() if WHEATHER_API_KEY_HUB else []
                results["forecasts"].extend(forecasts)
        
        # 권역 데이터
        if days_from_today >= 4 and should_fetch_region_forecasts(q, days_from_today):
            region_code = get_region_code(region)
            if region_code:
                region_forecasts = fetch_mid_week_land(
                    reg_id=region_code,
                    target_date=date,
                    tmfc_range_days=3,
                    widen_days=0
                ) if WHEATHER_API_KEY_HUB else []
                results["region_forecasts"].extend(region_forecasts)
    
    return results

def get_region_code(region_name: str) -> Optional[str]:
    """지역명으로 지역코드 찾기 (간단한 버전)"""
    # 간단한 매핑
    region_map = {
        "서울": "11B00000",
        "부산": "11H20000", 
        "대구": "11H10000",
        "인천": "11B00000",
        "광주": "11F20000",
        "대전": "11C20000",
        "울산": "11H20000",
        "세종": "11C10000",
        "제주": "11G00000"
    }
    return region_map.get(region_name)

# =========[ 실행부 ]=========

# =========[ 라이브(KMA) 호출 ]=========
def _format_kma_record(raw: List[str]) -> Dict[str, str]:
    tm_st = raw[0] if len(raw)>0 else "N/A"
    tm_ed = raw[1] if len(raw)>1 else "N/A"
    reg_token = raw[4].strip() if len(raw)>4 else "N/A"
    wrn = (raw[5].strip() if len(raw)>5 else "")
    lvl = (raw[6].strip() if len(raw)>6 else "")
    cmd = (raw[7].strip() if len(raw)>7 else "")
    grd = (raw[8].strip() if len(raw)>8 else "")
    region_name = resolve_region(reg_token)
    payload = {
        "source":"KMA_ADVISORY",
        "region_raw": reg_token, "region_name": region_name,
        "region_type":"code" if REGION_CODE_RE.match(reg_token or "") else "name",
        "hazard_code": wrn, "hazard_name": WRN_MAP.get(wrn, "알수없음"),
        "level_code": lvl, "level_name": LVL_MAP.get(lvl, "N/A"),
        "command_code": cmd, "command_name": CMD_MAP.get(cmd, cmd),
        "window_start": tm_st, "window_end": tm_ed,
        "window_start_kst": fmt_kst_with_ampm(tm_st) if tm_st!="N/A" else "N/A",
        "window_end_kst": fmt_kst_with_ampm(tm_ed) if tm_ed!="N/A" else "N/A",
        "announce_time_kst": fmt_kst_with_ampm(tm_st) if cmd=="1" and tm_st!="N/A" else None,
    }
    if wrn=="T" and grd: payload["typhoon_grade"] = grd
    time_bits = []
    if payload["window_start_kst"]!="N/A" and payload["window_end_kst"]!="N/A":
        time_bits.append(f"기간: {payload['window_start_kst']} ~ {payload['window_end_kst']}")
    elif payload["window_start_kst"]!="N/A":
        time_bits.append(f"시각: {payload['window_start_kst']}")
    if payload["announce_time_kst"]:
        time_bits.append(f"발표시각: {payload['announce_time_kst']}")
    parts = [
        f"지역: {region_name} ({reg_token})", *time_bits,
        f"현상: {payload['hazard_name']}({payload['hazard_code']})",
        f"수준: {payload['level_name']}({payload['level_code']})",
        f"명령: {payload['command_name']}({payload['command_code']})"
    ]
    if "typhoon_grade" in payload: parts.append(f"태풍 등급: {payload['typhoon_grade']}")
    return {"json": json.dumps(payload, ensure_ascii=False, separators=(",", ":")), "human": " | ".join(parts)}

def fetch_kma_advisories(start_time: str, end_time: str, disp: str="1") -> List[Dict[str, str]]:
    if not WHEATHER_API_KEY_HUB: return []
    
    # 인터넷 연결 확인
    if not check_internet_connection():
        print("❌ 인터넷 연결이 없습니다. 네트워크 상태를 확인해주세요.")
        return []
    
    # 기상청 서버 연결 확인
    if not check_kma_server_connectivity():
        print("❌ 기상청 API 서버에 연결할 수 없습니다. 서버 상태를 확인해주세요.")
        return []
    
    base = "https://apihub.kma.go.kr/api/typ01/url/wrn_met_data.php"
    params = {"authKey":WHEATHER_API_KEY_HUB, "wrn":"", "tmfc1":start_time, "tmfc2":end_time, "disp":disp}    
    for retry in range(3):  # 최대 3번 재시도
        try:
            r = requests.get(base, params=params, timeout=KMA_TIMEOUT)
            r.raise_for_status()
            text = r.content.decode("euc-kr", errors="ignore")
            docs, seen = [], set()
            for line in [ln for ln in text.strip().split("\n") if ln.strip() and not ln.startswith("#") and not ln.startswith("7777END")]:
                raw = line.strip().rstrip("=").split(",")
                if len(raw) < 9: continue
                rec = _format_kma_record(raw)
                key = re.sub(r"\s+", " ", rec["json"]).strip()
                if key in seen: continue
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

def _format_short_land_record(raw: list) -> Dict[str, str]:
    def g(i): return raw[i] if i < len(raw) else ""
    reg_id   = g(0); tm_fc = g(1); tm_ef = g(2); mod = g(3); ne = g(4)
    w1, w2   = g(9), g(11)
    ta, st   = g(12), g(13)
    sky, prep, wf = g(14), g(15), g(16)
    reg_name = resolve_region(reg_id)
    payload = {
        "source": "KMA_SHORT_LAND",
        "region_id": reg_id, "region_name": reg_name,
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
    return {"json": json.dumps(payload, ensure_ascii=False, separators=(",", ":")), "human": human}

def fetch_short_land_records() -> list:
    if not WHEATHER_API_KEY_HUB: return []
    
    # 인터넷 연결 확인
    if not check_internet_connection():
        print("❌ 인터넷 연결이 없습니다. 네트워크 상태를 확인해주세요.")
        return []
    
    # 기상청 서버 연결 확인
    if not check_kma_server_connectivity():
        print("❌ 기상청 API 서버에 연결할 수 없습니다. 서버 상태를 확인해주세요.")
        return []
    
    BASE = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_dl.php"
    params = {"reg": "", "tmfc": "0", "disp": "1", "authKey": WHEATHER_API_KEY_HUB}
    
    for retry in range(3):  # 최대 3번 재시도
        try:
            r = requests.get(f"{BASE}?{urlencode(params)}", timeout=KMA_TIMEOUT)
            r.raise_for_status()
            text = r.content.decode("euc-kr", errors="replace")
            docs = []
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("7777END"): continue
                if s.endswith("="): s = s[:-1]
                raw_row = [c.strip() for c in s.split(",")]
                if len(raw_row) < 17: continue
                docs.append(_format_short_land_record(raw_row))
            return docs
        except Exception as e:
            print(f"❌ 단기예보 API 오류 (시도 {retry+1}/3): {e}")
            if retry < 2:  # 마지막 시도가 아니면 잠시 대기
                import time
                time.sleep(2)
            else:
                print(f"❌ 단기예보 API 최종 실패")
                return []

def _reverse_region_map() -> Dict[str, List[str]]:
    return {k: list(dict.fromkeys(v)) for k, v in REGION_NAME_INDEX.items()}

def extract_region_from_question_live(q: str) -> Optional[str]:
    if not q: return None
    m = re.search(r"[A-Z]\d{7}", q)
    if m:
        code = m.group(0); name = REGION_MAP.get(code)
        if name: return name
    qn  = _norm_name(q)
    rev = _reverse_region_map()
    for norm_name in sorted(rev.keys(), key=len, reverse=True):
        if norm_name and norm_name in qn:
            for c in rev[norm_name]:
                nm = REGION_MAP.get(c)
                if nm: return nm
    commons = ["울진군","울진","영덕군","영덕","부산동부","울산동부","거제시","남해군","제주도북부","제주북부","사천시","파주시"]
    for token in commons:
        tn = _norm_name(token)
        if tn in qn:
            for code, name in REGION_MAP.items():
                if tn in _norm_name(name): return name
    return None

def summarize_region_alert(q: str, live_docs: List[Dict[str, str]], question_date: Optional[datetime] = None) -> str:
    region_name = extract_region_from_question_live(q)
    if not region_name: return ""
    
    # 질문 날짜가 4일 이후면 특보 요약 안 함
    if question_date:
        days_from_today = get_days_from_today(question_date)
        if days_from_today >= 4:
            return f"[LIVE_STATUS] {region_name}: 4일 이후 날짜로 특보 정보는 제공되지 않습니다. 중기예보를 참고하세요."
    
    rev = _reverse_region_map()
    target_codes = rev.get(_norm_name(region_name), [])  # ✅ 정규화하여 조회
    if not target_codes:
        return f"[LIVE_STATUS] {region_name}의 특보 정보를 찾을 수 없습니다. 지역 코드 매핑을 확인해주세요."
    
    matched = []
    current_time = now_kst()
    
    for d in live_docs or []:
        try:
            payload = json.loads(d["json"])
            if payload.get("region_raw") in target_codes:
                # 특보 시간 상태 확인
                start_time = payload.get('window_start', '')
                end_time = payload.get('window_end', '')
                is_active, status, start_dt, end_dt = is_forecast_active(start_time, end_time, current_time)
                payload['current_status'] = status
                matched.append(payload)
        except Exception:
            continue
    
    if not matched:
        return f"[LIVE_STATUS] 오늘 {region_name}에는 발효 중인 특보가 없습니다."
    
    matched.sort(key=lambda x: x.get("window_start", ""), reverse=True)
    p = matched[0]
    region, hz = p.get("region_name", region_name), p.get("hazard_name", "특보")
    lvl, cmd = p.get("level_name", ""), p.get("command_name", "")
    st, ed = p.get("window_start_kst", ""), p.get("window_end_kst", "")
    current_status = p.get("current_status", "알 수 없음")
    
    bits = [f"{region}에는 {hz} {lvl}가"]
    
    if current_status == "발효 중":
        if st and ed: 
            bits.append(f"{st}에 발효되어 {ed}에 해제 예정입니다.")
        elif st: 
            bits.append(f"{st}에 발효되었습니다.")
        else: 
            bits.append("발효 중입니다.")
    elif current_status == "종료됨":
        if ed:
            bits.append(f"{ed}에 종료되었습니다.")
        else:
            bits.append("이미 종료되었습니다.")
    elif current_status == "발효 예정":
        if st:
            bits.append(f"{st}에 발효 예정입니다.")
        else:
            bits.append("발효 예정입니다.")
    
    if cmd: bits.append(f"(명령: {cmd})")
    return "[LIVE_STATUS] " + " ".join(bits)

# =========[ 임베딩 ]=========
_text_embedder = SentenceTransformer(EMBED_MODEL_NAME)
def embed_texts(texts: List[str]) -> np.ndarray:
    embs = _text_embedder.encode(texts, show_progress_bar=False)
    embs = np.array([l2_normalize(e) for e in embs], dtype="float32")
    return embs

# =========[ 예보관 톤: 정규화/위험도 헬퍼 ]=========
def _norm_temp(val: str) -> Optional[float]:
    try:
        v = float(str(val).replace("°C","").strip())
        if v < -90 or v > 60: return None
        return v
    except:
        return None

def _risk_level_from_alert(hazard: str, level_name: str) -> str:
    if level_name in ("경보",): return "높음"
    if level_name in ("주의보",): return "보통"
    return "약"

def _risk_level_from_forecast(sky_status: str, precip_prob: str, temp: str) -> str:
    try:
        p = int(str(precip_prob).replace("%","")) if precip_prob else 0
    except:
        p = 0
    t = _norm_temp(temp)
    high_precip = p >= 60
    extreme_temp = (t is not None) and (t <= -10 or t >= 33)
    if high_precip and extreme_temp: return "높음"
    if high_precip or extreme_temp:  return "보통"
    return "약"

def _format_for_llm(src: str, payload_json: str, human: str) -> str:
    try:
        p = json.loads(payload_json)
    except:
        return f"[{src}] {human}"
    
    if src == "live_advisory":
        # 특보 시간 상태 확인
        start_time = p.get('window_start', '')
        end_time = p.get('window_end', '')
        is_active, status, start_dt, end_dt = is_forecast_active(start_time, end_time)
        
        line = (
            f"[{src}] 지역:{p.get('region_name','N/A')} | 현상:{p.get('hazard_name','N/A')} "
            f"| 수준:{p.get('level_name','N/A')} | 상태:{status} "
            f"| 발표:{p.get('window_start_kst','N/A')} | 해제:{p.get('window_end_kst','N/A')}"
        )
        risk = _risk_level_from_alert(p.get('hazard_name',''), p.get('level_name',''))
        return f"{line}\n[RISKS] 위험도(간이): {risk}\n[STATUS] {status}\n[NOTE] {human}"
        
    elif src == "live_forecast":
        temp = p.get("temp","N/A"); prob = p.get("precip_prob","N/A"); sky  = p.get("sky_status","N/A")
        
        # 예보 시간 확인 (단기예보는 보통 3일 이내)
        forecast_time = p.get('effective_time', '')
        forecast_dt = parse_kst_time(forecast_time)
        days_from_today = get_days_from_today(forecast_dt) if forecast_dt else 0
        
        forecast_type = "단기예보" if days_from_today < 4 else "중기예보_범위"
        
        line = (
            f"[{src}] 지역:{p.get('region_name','N/A')} | 시각:{p.get('forecast_time','N/A')} "
            f"| 하늘:{sky} | 기온:{temp} | 강수확률:{prob} | 구분:{forecast_type}"
        )
        risk = _risk_level_from_forecast(sky, prob, temp)
        return f"{line}\n[RISKS] 위험도(간이): {risk}\n[FORECAST_TYPE] {forecast_type}\n[NOTE] {human}"
        
    elif src == "region_forecast":
        line = (
            f"[{src}] 지역:{p.get('region_name','N/A')} | 대상:{p.get('forecast_time','N/A')} "
            f"| 하늘:{(p.get('sky_text') or 'N/A')} | 강수형:{(p.get('precip_type') or 'N/A')} | 강수확률:{(p.get('precip_prob') or 'N/A')}"
        )
        # 권역은 온도 수치 없음 → 위험도는 강수/현상 위주로 보수적으로
        risk = "보통" if (p.get('precip_type') not in (None,"","없음") or str(p.get('precip_prob','')).isdigit()) else "약"
        return f"{line}\n[RISKS] 위험도(간이): {risk}\n[FORECAST_TYPE] 권역예보\n[NOTE] {human}"
        
    return f"[{src}] {human}"

# =========[ 권역예보(wl) 전용: CSV 권역/질의 해석 ]=========
REGION_REGIONS_CSV = UNIFIED_REGIONS_CSV  # 통합 CSV 사용
MAX_RETRIES = 3
BACKOFF = 1.5
MERGE_BY_DAY = (os.getenv("MERGE_BY_DAY", "true").lower() in ("1", "true", "yes"))
WEEKDAY_IDX = {"월":0,"화":1,"수":2,"목":3,"금":4,"토":5,"일":6}

def now_kst() -> datetime:
    return datetime.now(tz=KST)

def parse_kst_time(time_str: str) -> Optional[datetime]:
    """YYYYMMDDHHMM 형식의 문자열을 KST datetime으로 변환"""
    try:
        if len(time_str) >= 12:
            return datetime.strptime(time_str[:12], "%Y%m%d%H%M").replace(tzinfo=KST)
        elif len(time_str) >= 10:
            return datetime.strptime(time_str[:10], "%Y%m%d%H").replace(tzinfo=KST)
        elif len(time_str) >= 8:
            return datetime.strptime(time_str[:8], "%Y%m%d").replace(tzinfo=KST)
    except Exception:
        pass
    return None

def is_forecast_active(start_time: str, end_time: str, current_time: Optional[datetime] = None) -> tuple:
    """특보/예보가 현재 활성 상태인지 확인
    Returns: (is_active: bool, status: str, start_dt: datetime, end_dt: datetime)
    """
    if current_time is None:
        current_time = now_kst()
    
    start_dt = parse_kst_time(start_time)
    end_dt = parse_kst_time(end_time)
    
    if not start_dt:
        return (False, "시간 정보 없음", None, None)
    
    if not end_dt:
        # 종료 시간이 없으면 시작 시간 기준으로만 판단
        if current_time >= start_dt:
            return (True, "발효 중", start_dt, None)
        else:
            return (False, "발효 예정", start_dt, None)
    
    if current_time < start_dt:
        return (False, "발효 예정", start_dt, end_dt)
    elif current_time > end_dt:
        return (False, "종료됨", start_dt, end_dt)
    else:
        return (True, "발효 중", start_dt, end_dt)

def get_days_from_today(target_date: Optional[datetime] = None) -> int:
    """오늘 기준으로 며칠 후인지 계산"""
    if target_date is None:
        return 0
    today = now_kst().date()
    target = target_date.date()
    return (target - today).days

def fmt_kst_any(s: str) -> str:
    s = (s or "").strip()
    patterns = [("%Y%m%d%H%M","%Y-%m-%d %H:%M KST"),("%Y%m%d%H","%Y-%m-%d %H:00 KST"),("%Y%m%d","%Y-%m-%d")]
    for fin, fout in patterns:
        try:
            dt = datetime.strptime(s, fin).replace(tzinfo=KST)
            return dt.strftime(fout)
        except Exception:
            pass
    return s

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", "", s or "")

# --- CSV 로딩 ---
def REGION_load_all_from_csv(path: str) -> Dict[str, str]:
    """통합 CSV에서 권역예보용 권역 정보 로드"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"권역 CSV를 찾을 수 없습니다: {path}")
    
    try:
        # pandas로 읽기 (더 안정적)
        df = pd.read_csv(path, encoding='utf-8-sig')
        
        # 컬럼 자동 감지
        code_col, name_col = _pick_cols(df)
        
        if not code_col or not name_col:
            raise ValueError("CSV에서 코드/이름 컬럼을 찾을 수 없습니다.")
        
        mapping: Dict[str, str] = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            
            if code and name and code.lower() != "nan" and name.lower() != "nan":
                mapping[code] = name
        
        if not mapping:
            raise ValueError("CSV에서 유효한 권역 정보를 읽지 못했습니다.")
        
        return mapping
        
    except Exception as e:
        print(f"❌ CSV 로딩 실패: {e}")
        raise

# 통합 CSV의 다양한 지역코드 형식 지원
REGION_REGION_CODE_RE = re.compile(r"^(11[A-Z](\d)?0{4,5}|L\d{7}|S\d{7})$")
def REGION_split_region_only(all_map: Dict[str, str]) -> Dict[str, str]:
    region_map = {rid: nm for rid, nm in all_map.items() if REGION_REGION_CODE_RE.match(rid)}
    if not region_map:
        known = ["11B00000","11D10000","11D20000","11C20000","11C10000","11F20000","11F10000","11H20000","11H10000","11G00000"]
        region_map = {rid: all_map[rid] for rid in known if rid in all_map}
    return region_map

def REGION_build_alias_map(region_map: Dict[str, str]) -> Dict[str, str]:
    alias: Dict[str, str] = {}
    norm = lambda x: re.sub(r"\s+", "", x or "")
    def add(a: str, full: str):
        a = norm(a)
        if a and a not in alias: alias[a] = full
    for _code, full in region_map.items():
        add(full, full)
        for p in full.split("·"): add(p, full)
        subs = {"충청북도":"충북","충청남도":"충남","전라북도":"전북","전라남도":"전남","경상북도":"경북","경상남도":"경남","제주도":"제주","경기도":"경기"}
        for long, short in subs.items():
            if long in full: add(short, full); add(long, full)
        if ("서울" in full) and ("인천" in full) and ("경기" in full or "경기도" in full):
            add("수도권", full)
        if "강원" in full and ("영서" in full or "영동" in full):
            add("강원", full)
            if "영서" in full: add("영서", full); add("강원영서", full)
            if "영동" in full: add("영동", full); add("강원영동", full)
    return alias

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
    # 기존 11X 형식
    (r"^11B",  "11B00000"), (r"^11D1", "11D10000"), (r"^11D2", "11D20000"),
    (r"^11C1", "11C10000"), (r"^11C2", "11C20000"), (r"^11F1", "11F10000"),
    (r"^11F2", "11F20000"), (r"^11H1", "11H10000"), (r"^11H2", "11H20000"),
    (r"^11G",  "11G00000"),
    # 통합 CSV의 L, S 형식 추가
    (r"^L100", "L1000000"), (r"^L101", "L1010000"), (r"^L102", "L1020000"),
    (r"^L103", "L1030000"), (r"^L104", "L1040000"), (r"^L105", "L1050000"),
    (r"^L106", "L1060000"), (r"^L107", "L1070000"), (r"^L108", "L1080000"),
    (r"^S100", "S1000000"), (r"^S110", "S1100000"), (r"^S120", "S1200000"),
    (r"^S130", "S1300000"), (r"^S140", "S1400000"), (r"^S150", "S1500000"),
    (r"^S160", "S1600000"), (r"^S170", "S1700000"), (r"^S180", "S1800000"),
]

def REGION_normalize_region_reg_code(code_like: str) -> Optional[str]:
    c = (code_like or "").strip()
    if not c: return None
    
    # 직접 매칭
    if c in REGION_LAND_MAP: return c
    
    # 패턴 매칭
    for pat, target in REGION_FAMILY_RULES:
        if re.match(pat, c): 
            return target if target in REGION_LAND_MAP else target
    
    # 통합 CSV의 다양한 형식 지원
    # L 형식: L1000000 -> L1000000, L1010200 -> L1010000
    if c.startswith('L') and len(c) >= 6:
        base_code = c[:4] + '0000'
        if base_code in REGION_LAND_MAP: return base_code
    
    # S 형식: S1000000 -> S1000000, S1100000 -> S1100000
    if c.startswith('S') and len(c) >= 6:
        base_code = c[:4] + '0000'
        if base_code in REGION_LAND_MAP: return base_code
    
    return None

def REGION_extract_datetime_from_question(q: str) -> Optional[datetime]:
    qn = normalize_spaces(q); now = now_kst()
    if any(k in qn for k in ["오늘","현재","지금"]): return now
    if "내일" in qn: return now + timedelta(days=1)
    if "모레" in qn: return now + timedelta(days=2)
    m = re.search(r"(\d+)(일|주|달|년)(뒤|후)", qn)
    if m:
        num, unit = int(m.group(1)), m.group(2)
        return (now + timedelta(days=num) if unit == "일" else
                now + timedelta(weeks=num) if unit == "주" else
                now + timedelta(days=30*num) if unit == "달" else
                now + timedelta(days=365*num))
    if "다음주" in qn:
        for k, idx in WEEKDAY_IDX.items():
            if k in qn:
                days_to_next_mon = ((7 - now.weekday()) % 7) or 7
                return now + timedelta(days=days_to_next_mon + idx)
        days_to_next_mon = ((7 - now.weekday()) % 7) or 7
        return now + timedelta(days=days_to_next_mon)
    m2 = re.search(r"(\d{1,2})월(\d{1,2})일", qn)
    if m2:
        M, D = int(m2.group(1)), int(m2.group(2))
        try: return datetime(now.year, M, D, tzinfo=KST)
        except ValueError: return None
    return None

def REGION_is_region_term_date(date: Optional[datetime]) -> Tuple[bool, Optional[int]]:
    if date is None: return (False, None)
    delta = (date.date() - now_kst().date()).days
    return (3 < delta <= 10, delta)

def REGION_extract_region_from_question(q: str) -> str:
    qn = normalize_spaces(q)
    
    # 통합 CSV의 다양한 지역코드 패턴 지원
    patterns = [
        r"(11[A-Z]\d{5,})",  # 기존 11X 형식
        r"(L\d{7})",         # L 형식
        r"(S\d{7})",         # S 형식
    ]
    
    for pattern in patterns:
        m = re.search(pattern, qn)
        if m:
            cand = REGION_normalize_region_reg_code(m.group(1))
            if cand: return REGION_LAND_MAP.get(cand, cand)
    
    # 지역명 매칭
    for _code, name in REGION_LAND_MAP.items():
        if normalize_spaces(name) in qn: return name
    
    # 별칭 매칭
    for alias, full in REGION_ALIASES.items():
        if alias in qn: return full
    
    # 수도권 특별 처리
    for code, name in REGION_LAND_MAP.items():
        if "서울" in name and "인천" in name and ("경기" in name or "경기도" in name):
            return name
    
    # 기본값
    first_code = next(iter(REGION_LAND_MAP.keys())) if REGION_LAND_MAP else None
    return REGION_LAND_MAP.get(first_code, "수도권") if first_code else "수도권"

def REGION_region_name_to_code(region_full_name: str) -> Optional[str]:
    for code, nm in REGION_LAND_MAP.items():
        if nm == region_full_name: return code
    return None

def request_with_retries(url: str, timeout: int, retries: int, backoff: float) -> requests.Response:
    # 인터넷 연결 확인
    if not check_internet_connection():
        raise RuntimeError("인터넷 연결이 없습니다. 네트워크 상태를 확인해주세요.")
    
    last = None
    for i in range(1, retries+1):
        try:
            print(f"[HTTP] GET {url} (try={i})")
            r = requests.get(url, timeout=timeout)
            print(f"[HTTP] -> {r.status_code}, {len(r.content)} bytes")
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if i < retries: time.sleep(backoff * i)
    raise last or RuntimeError("request failed")

def REGION_parse_wl_line(line: str) -> Optional[Dict[str, str]]:
    s = (line or "").strip()
    if not s or s.startswith("#") or s.startswith("7777END"): return None
    if s.endswith("="): s = s[:-1]
    cols = [c.strip() for c in s.split(",") if c.strip() != ""] if "," in s else [c.strip() for c in re.split(r"\s+", s) if c.strip()]
    if len(cols) < 3: return None
    looks_date = lambda x: bool(re.fullmatch(r"\d{8}(\d{2}(\d{2})?)?", x))
    if len(cols) >= 11 and looks_date(cols[1]) and looks_date(cols[2]):
        return {"reg_id": cols[0], "tmfc": cols[1], "tmef": cols[2], "sky_code": cols[6], "pre_code": cols[7], "conf": cols[8], "wf": cols[9], "rn_st": cols[10]}
    reg_id = cols[0]; tmfc = None; tmef = None; rest: List[str] = []
    for c in cols[1:]:
        if tmfc is None and looks_date(c): tmfc = c; continue
        if tmfc is not None and tmef is None and looks_date(c): tmef = c; continue
        rest.append(c)
    return {"reg_id": reg_id, "tmfc": tmfc or "", "tmef": tmef or "", "wf": " ".join(rest).strip(), "conf": "", "rn_st": "", "sky_code": "", "pre_code": ""}

def REGION_merge_by_day(latest_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    day_map: Dict[str, Dict[str, Optional[Dict[str, str]]]] = {}
    for it in latest_rows:
        tmef = it.get("tmef", "")
        if not tmef or len(tmef) < 8: continue
        day = tmef[:8]
        hh = (tmef + "  ")[8:10]
        slot = "am" if hh == "00" else ("pm" if hh == "12" else "etc")
        rec = day_map.setdefault(day, {"am": None, "pm": None, "etc": []})
        if slot in ("am", "pm"): rec[slot] = it
        else: rec["etc"].append(it)
    merged: List[Dict[str, str]] = []
    for day, slots in sorted(day_map.items()):
        am = slots["am"]; pm = slots["pm"]
        if am or pm:
            tmfc_candidates = [x.get("tmfc","") for x in (am, pm) if x]
            tmfc = max(tmfc_candidates) if tmfc_candidates else ""
            wf_am = (am or {}).get("wf",""); rn_am = (am or {}).get("rn_st","")
            wf_pm = (pm or {}).get("wf",""); rn_pm = (pm or {}).get("rn_st","")
            bits = []
            if wf_am: bits.append(f"오전 {wf_am}{f'({rn_am}%)' if rn_am.isdigit() else ''}")
            if wf_pm: bits.append(f"오후 {wf_pm}{f'({rn_pm}%)' if rn_pm.isdigit() else ''}")
            wf = " / ".join(bits) if bits else (wf_am or wf_pm or "")
            merged.append({"tmfc": tmfc, "tmef": day, "wf": wf, "rn_st": "", "conf": "", "sky_code": "", "pre_code": ""})
    return merged

def fetch_mid_week_land(reg_id: str, target_date: datetime, tmfc_range_days: int = 3, widen_days: int = 0,
                        disp: str = "1", help_flag: str = "0", merge_day: bool = MERGE_BY_DAY) -> List[Dict[str, str]]:
    if not WHEATHER_API_KEY_HUB:
        print("❌ API 키(WHEATHER_API_KEY_HUB)가 없습니다."); return []
    
    # 인터넷 연결 확인
    if not check_internet_connection():
        print("❌ 인터넷 연결이 없습니다. 네트워크 상태를 확인해주세요.")
        return []
    
    # 기상청 서버 연결 확인
    if not check_kma_server_connectivity():
        print("❌ 기상청 API 서버에 연결할 수 없습니다. 서버 상태를 확인해주세요.")
        return []
    
    now = now_kst()
    tmfc1 = now.astimezone(KST) - timedelta(days=tmfc_range_days)
    tmfc2 = now
    tmef1 = target_date - timedelta(days=widen_days)
    tmef2 = target_date + timedelta(days=widen_days)
    BASE = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_wl.php"
    params = {"reg": reg_id, "tmfc1": tmfc1.strftime("%Y%m%d%H"), "tmfc2": tmfc2.strftime("%Y%m%d%H"),
              "tmef1": tmef1.strftime("%Y%m%d"), "tmef2": tmef2.strftime("%Y%m%d"),
              "disp": disp, "help": help_flag, "authKey": WHEATHER_API_KEY_HUB}
    url = f"{BASE}?{urlencode(params)}"
    try:
        r = request_with_retries(url, timeout=KMA_TIMEOUT, retries=MAX_RETRIES, backoff=BACKOFF)
        text = r.content.decode("euc-kr", errors="replace")
        latest: Dict[str, Dict[str, str]] = {}
        want_d8 = target_date.strftime("%Y%m%d")
        for ln in text.splitlines():
            item = REGION_parse_wl_line(ln)
            if not item: continue
            if item.get("reg_id") != reg_id: continue
            tmef = item.get("tmef", "")
            if (tmef or "")[:8] != want_d8: continue
            prev = latest.get(tmef)
            if (prev is None) or (item.get("tmfc","") > prev.get("tmfc","")):
                latest[tmef] = item
        if not latest: return []
        rows = [latest[k] for k in sorted(latest.keys())]
        if merge_day: rows = REGION_merge_by_day(rows)
        docs: List[Dict[str, str]] = []
        region_name = REGION_LAND_MAP.get(reg_id, reg_id)
        for it in rows:
            is_day_level = len(it.get("tmef","")) == 8
            target_label = (fmt_kst_any(it["tmef"]) if not is_day_level else
                            datetime.strptime(it["tmef"], "%Y%m%d").replace(tzinfo=KST).strftime("%Y-%m-%d"))
            wf = (it.get("wf","") or "").strip(); conf = (it.get("conf","") or "").strip(); rn = (it.get("rn_st","") or "").strip()
            bits = []
            if wf: bits.append(wf)
            if conf and conf != "없음": bits.append(conf)
            if rn.isdigit(): bits.append(f"강수확률 {rn}%")
            summary = " · ".join(bits) if bits else (wf or "예보문 없음")
            payload = {
                "source": "KMA_REGION_WEEK_LAND",
                "region_id": reg_id, "region_name": region_name,
                "announce_time": it.get("tmfc",""),
                "forecast_time": it.get("tmef",""),
                "sky_text": wf, "precip_type": conf, "precip_prob": rn,
                "sky_code": it.get("sky_code",""), "pre_code": it.get("pre_code",""),
            }
            human = f"{region_name} 권역예보(주간) — 발표 {fmt_kst_any(payload['announce_time'])}, 대상 {target_label}: {summary}"
            docs.append({"json": json.dumps(payload, ensure_ascii=False, separators=(',', ':')), "human": human})
        return docs
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP 오류: {e}")
        return []
    except Exception as e:
        print(f"❌ API 호출/파싱 오류: {e}")
        return []

# =========[ 그래프 상태/LLM ]=========
class GraphState(TypedDict):
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    answer_draft: Optional[str]
    store_obj: Optional[Any]
    retry_count: int
    is_context_valid: Optional[bool]
    is_retrieval_sufficient: Optional[bool]
    is_answer_sufficient: Optional[bool]
    next_action: Optional[str]

def make_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY=REDACTED ValueError("OPENAI_API_KEY=REDACTED에 없습니다.")
    return ChatOpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY=REDACTED _should_use_live(q: str) -> bool:
    if FORCE_DISABLE_KMA_LIVE or not USE_KMA_LIVE: return False
    q = (q or "")
    live_kw = [
        "현재","지금","오늘","특보","주의보","경보","해제","발표","상향","하향",
        "시간","시각","내일","모레","새벽","오전","오후","야간","금일","당일","실시간",
        "날씨","기온","강수","소나기","호우","강풍","폭염","한파","대설","황사","풍랑","태풍"
    ]
    return any(k in q for k in live_kw)

# =========[ RAGAS 결과 파싱 헬퍼 ]=========
def _ragas_overall(result_obj: Any, metric_name: str) -> Optional[float]:
    try:
        val = None
        
        # 1. RAGAS 0.3.x _scores_dict 속성에서 직접 접근 (가장 일반적)
        if hasattr(result_obj, "_scores_dict") and isinstance(result_obj._scores_dict, dict):
            val = result_obj._scores_dict.get(metric_name)
            if val is not None:
                # 리스트인 경우 첫 번째 값 사용
                if isinstance(val, list) and len(val) > 0:
                    val = val[0]
                # JSON 문자열인 경우 파싱 시도
                elif isinstance(val, str) and val.startswith('{'):
                    try:
                        import json
                        json_data = json.loads(val)
                        if "statements" in json_data and isinstance(json_data["statements"], list):
                            # verdict 값들의 평균 계산
                            verdicts = []
                            for stmt in json_data["statements"]:
                                if isinstance(stmt, dict) and "verdict" in stmt:
                                    verdicts.append(float(stmt["verdict"]))
                            if verdicts:
                                val = sum(verdicts) / len(verdicts)
                                print(f"   - ✅ {metric_name}: {val:.4f} (JSON verdict 평균)")
                                return val
                    except:
                        pass
                val = float(val)
                if val == val:  # NaN 체크
                    print(f"   - ✅ {metric_name}: {val:.4f} (_scores_dict)")
                    return val
        
        # 2. RAGAS 0.3.x scores 속성에서 직접 접근
        if hasattr(result_obj, "scores") and hasattr(result_obj.scores, metric_name):
            val = getattr(result_obj.scores, metric_name)
            if val is not None:
                # JSON 문자열인 경우 파싱 시도
                if isinstance(val, str) and val.startswith('{'):
                    try:
                        import json
                        json_data = json.loads(val)
                        if "statements" in json_data and isinstance(json_data["statements"], list):
                            # verdict 값들의 평균 계산
                            verdicts = []
                            for stmt in json_data["statements"]:
                                if isinstance(stmt, dict) and "verdict" in stmt:
                                    verdicts.append(float(stmt["verdict"]))
                            if verdicts:
                                val = sum(verdicts) / len(verdicts)
                                print(f"   - ✅ {metric_name}: {val:.4f} (scores JSON verdict 평균)")
                                return val
                    except:
                        pass
                val = float(val)
                if val == val:  # NaN 체크
                    print(f"   - ✅ {metric_name}: {val:.4f} (scores 속성)")
                    return val
        
        # 3. to_dict() 시도
        if hasattr(result_obj, "to_dict"):
            d = result_obj.to_dict()
            if isinstance(d, dict):
                # scores 딕셔너리 내부 확인 (RAGAS 0.3.x)
                if "scores" in d and isinstance(d["scores"], dict):
                    val = d["scores"].get(metric_name)
                    if val is not None:
                        val = float(val)
                        if val == val:  # NaN 체크
                            print(f"   - ✅ {metric_name}: {val:.4f} (to_dict scores)")
                            return val
                
                # overall 딕셔너리 내부 확인 (구버전)
                if "overall" in d and isinstance(d["overall"], dict):
                    val = d["overall"].get(metric_name)
                    if val is not None:
                        val = float(val)
                        if val == val:  # NaN 체크
                            print(f"   - ✅ {metric_name}: {val:.4f} (to_dict overall)")
                            return val
                
                # 직접 키 접근
                if metric_name in d:
                    val = d[metric_name]
                    if val is not None:
                        val = float(val)
                        if val == val:  # NaN 체크
                            print(f"   - ✅ {metric_name}: {val:.4f} (to_dict 직접)")
                            return val
        
        # 4. __dict__ 시도
        if hasattr(result_obj, "__dict__"):
            d = result_obj.__dict__
            # _scores_dict 딕셔너리 내부 확인 (RAGAS 0.3.x)
            if "_scores_dict" in d and isinstance(d["_scores_dict"], dict):
                val = d["_scores_dict"].get(metric_name)
                if val is not None:
                    # 리스트인 경우 첫 번째 값 사용
                    if isinstance(val, list) and len(val) > 0:
                        val = val[0]
                    # JSON 문자열인 경우 파싱 시도
                    elif isinstance(val, str) and val.startswith('{'):
                        try:
                            import json
                            json_data = json.loads(val)
                            if "statements" in json_data and isinstance(json_data["statements"], list):
                                # verdict 값들의 평균 계산
                                verdicts = []
                                for stmt in json_data["statements"]:
                                    if isinstance(stmt, dict) and "verdict" in stmt:
                                        verdicts.append(float(stmt["verdict"]))
                                if verdicts:
                                    val = sum(verdicts) / len(verdicts)
                                    print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ JSON verdict 평균)")
                                    return val
                        except:
                            pass
                    val = float(val)
                    if val == val:  # NaN 체크
                        print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ _scores_dict)")
                        return val
            
            # scores 딕셔너리 내부 확인 (RAGAS 0.3.x)
            if "scores" in d and isinstance(d["scores"], dict):
                val = d["scores"].get(metric_name)
                if val is not None:
                    val = float(val)
                    if val == val:  # NaN 체크
                        print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ scores)")
                        return val
            
            if "overall" in d and isinstance(d["overall"], dict):
                val = d["overall"].get(metric_name)
                if val is not None:
                    val = float(val)
                    if val == val:  # NaN 체크
                        print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ overall)")
                        return val
            
            if metric_name in d:
                val = d[metric_name]
                if val is not None:
                    val = float(val)
                    if val == val:  # NaN 체크
                        print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ 직접)")
                        return val
        
        # 5. 직접 속성 접근
        if hasattr(result_obj, metric_name):
            val = getattr(result_obj, metric_name)
            if val is not None:
                val = float(val)
                if val == val:  # NaN 체크
                    print(f"   - ✅ {metric_name}: {val:.4f} (직접 속성)")
                    return val
        
        print(f"   - ❌ {metric_name} 값을 찾을 수 없음")
        return None
        
    except Exception as e:
        print(f"   - ⚠️ RAGAS 결과 파싱 실패 ({metric_name}): {e}")
        return None

# =========[ 프롬프트 (예보관 톤) ]=========
DRAFT_PROMPT = ChatPromptTemplate.from_template(
    """너는 대한민국 기상청 기준 용어를 사용하는 '현직 예보관'이야.
문맥에는 KMA 특보/단기/중기 예보 정보가 섞여 있다.
다음 '출력 규격'을 반드시 만족하는 한국어 자연문으로 초안 답변을 작성해.

[출력 규격 - 조건부 구조]

**질문이 오늘인 경우:**
1) 개요: 한 문장 핵심 요약
2) 기상특보 현황: 
   - 문맥의 [live_advisory] 데이터로 현재 특보 상태 표시
   - [STATUS] 기반: "발효 중"/"종료됨"/"발효 예정" 정확히 표시
   - 특보 데이터가 없으면 "특보 없음"
3) 단기예보 상세: 
   - 문맥의 [FORECAST_TYPE]="단기예보" 데이터만 사용
   - 하늘상태, 기온, 강수확률, 바람, 발표시각 포함해서 한줄로 작성해
4) 기상 정보 요약: 제공된 기상 데이터를 종합한 요약

**질문이 내일~3일 이내인 경우:**
1) 개요: 한 문장 핵심 요약
2) 단기예보 상세: 
   - 문맥의 [FORECAST_TYPE]="단기예보" 데이터만 사용
   - 하늘상태, 기온, 강수확률, 바람, 발표시각 포함해서 한줄로 작성해
3) 기상 정보 요약: 제공된 기상 데이터를 종합한 요약

**질문이 오늘+4일 이후인 경우:**
1) 개요: 한 문장 핵심 요약
2) 중기예보 전망:
   - 문맥의 [FORECAST_TYPE]이 "중기예보"인 정보만 사용
   - 향후 기간의 날씨 전망과 강수 가능성
   - 기온 변화 경향
3) 기상 정보 요약: 제공된 기상 데이터를 종합한 요약

금지: 표/코드/JSON/불릿(번호는 허용), 과도한 추측

[문맥]
{context}

질문: {question}
초안 답변(위 6구역 포함):"""
)

REFINE_PROMPT = ChatPromptTemplate.from_template(
    """다음은 초안과 근거 문맥이야.
예보관 관점에서 사실성(문맥 일치), 시간/지역 명시, 위험도 판단의 타당성, 조치의 구체성을 강화해 최종 답변을 작성해.
문맥 밖 정보 추가 금지. 표/코드/JSON 금지. 번호는 허용.

특히 다음 조건부 구조를 반드시 따르고 시간 기반 정보를 정확히 반영해:

**질문이 오늘인 경우:**
1) 개요
2) 기상특보 현황 ([live_advisory] 데이터로 현재 특보 상태)
3) 단기예보 상세 ([FORECAST_TYPE]="단기예보"만)
4) 기상 정보 요약

**질문이 내일~3일 이내인 경우:**
1) 개요
2) 단기예보 상세 ([FORECAST_TYPE]="단기예보"만)
3) 기상 정보 요약

**질문이 오늘+4일 이후인 경우:**
1) 개요
2) 중기예보 전망 ([FORECAST_TYPE]="중기예보"만)
3) 기상 정보 요약

※ 질문 날짜에 따라 섹션 개수와 번호가 달라짐
 
[문맥]
{context}

질문: {question}
초안: {answer_draft}

최종 답변(위 6구역 구조 유지):"""
)

# =========[ 노드 구현 ]=========
def load_store_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 초기화 (벡터DB 미사용)")
    return {**state, "store_obj": None, "retry_count": 0}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print(f"🧩 노드: 리트리브(라이브 + 중기) | 재시도: {state['retry_count']}")
    q = state["question"] or ""
    live_enabled = _should_use_live(q)

     # 복합 질문 처리 추가
    sub_questions = decompose_complex_question(q)
    if len(sub_questions) > 1:
        print(f"   - �� 복합 질문 감지: {len(sub_questions)}개 하위 질문으로 분해")
        # 병렬 처리로 데이터 수집
        weather_data = parallel_fetch_weather_data(sub_questions)
        # 기존 로직과 통합...
    
    live_enabled = _should_use_live(q)
    
    # 질문에서 날짜 추출하여 4일 이후면 라이브 데이터 제외
    question_date = REGION_extract_datetime_from_question(q)
    days_from_today = get_days_from_today(question_date) if question_date else 0
    
    # 오늘+4일부터는 라이브 데이터(특보/단기) 사용 안 함
    if days_from_today >= 4:
        print(f"ℹ️ 질문 날짜가 {days_from_today}일 후로, 라이브 데이터 제외하고 권역예보만 사용")
        live_enabled = False

    scored: List[tuple] = []
    live_status_msg = ""

    # --- 라이브 특보/단기 ---
    if live_enabled:
        try:
            now = datetime.now(tz=KST)
            tm1 = now.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M")
            tm2 = now.strftime("%Y%m%d%H%M")
            advisories = fetch_kma_advisories(tm1, tm2) if WHEATHER_API_KEY_HUB else []
            live_status_msg = summarize_region_alert(q, advisories, question_date)
            
            # 특보는 현재 발효 중인 것만 의미 있으므로 질문 날짜가 오늘일 때만 사용
            if advisories and days_from_today <= 0:
                human = [d["human"] for d in advisories]
                embs = embed_texts(human)
                idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
                qv = embed_texts([q])[0]
                topk = min(5, len(advisories))
                D, I = idx.search(np.array([qv], dtype="float32"), topk)
                for s, i in zip(D[0], I[0]):
                    if i == -1: continue
                    ctx = _format_for_llm("live_advisory", advisories[i]['json'], advisories[i]['human'])
                    scored.append((float(s) + 1.0, ctx, "live_advisory"))
        except Exception as e:
            print(f"❌ 특보 검색 오류: {e}")

        try:
            forecasts = fetch_short_land_records()
            # 단기예보는 3일까지 유효하므로 3일 이내일 때 사용
            if forecasts and days_from_today <= 3:
                human = [d["human"] for d in forecasts]
                embs = embed_texts(human)
                idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
                qv = embed_texts([q])[0]
                topk = min(5, len(forecasts))
                D, I = idx.search(np.array([qv], dtype="float32"), topk)
                for s, i in zip(D[0], I[0]):
                    if i == -1: continue
                    ctx = _format_for_llm("live_forecast", forecasts[i]['json'], forecasts[i]['human'])
                    scored.append((float(s) + 1.0, ctx, "live_forecast"))
        except Exception as e:
            print(f"❌ 단기예보 검색 오류: {e}")
    else:
        print("ℹ️ 라이브 기준 미충족 (질문에 실시간 키워드 없음)")

    # --- 권역예보 (오늘+4~10일) ---
    try:
        if REGION_LAND_MAP:  # CSV가 있을 때만
            q_date = REGION_extract_datetime_from_question(q)
            ok_region, delta = REGION_is_region_term_date(q_date)
            if ok_region:
                region_name = REGION_extract_region_from_question(q)
                region_code = REGION_region_name_to_code(region_name)
                m = re.search(r"(11[A-Z]\d{5,})", normalize_spaces(q))
                if m:
                    cand = REGION_normalize_region_reg_code(m.group(1))
                    if cand: region_code, region_name = cand, REGION_LAND_MAP.get(cand, region_name)
                if region_code:
                    region_docs = fetch_mid_week_land(reg_id=region_code, target_date=q_date,
                                                   tmfc_range_days=3, widen_days=0, disp="1", help_flag="0",
                                                   merge_day=MERGE_BY_DAY)
                    if region_docs:
                        human = [d["human"] for d in region_docs]
                        embs = embed_texts(human)
                        idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
                        qv = embed_texts([q])[0]
                        topk = min(5, len(region_docs))
                        D, I = idx.search(np.array([qv], dtype="float32"), topk)
                        for s, i in zip(D[0], I[0]):
                            if i == -1: continue
                            ctx = _format_for_llm("region_forecast", region_docs[i]['json'], region_docs[i]['human'])
                            scored.append((float(s) + 1.0, ctx, "region_forecast"))
                else:
                    print("⚠️ 권역 코드를 찾지 못했습니다.")
            else:
                if delta is not None:
                    print(f"ℹ️ 권역 대상 아님(오늘 기준 {delta:+d}일).")
    except Exception as e:
        print(f"❌ 권역예보 처리 오류: {e}")

    # --- 스코어 정규화 & dedup ---
    normalized: List[tuple] = []
    by_src = {
        "live_advisory": [h for h in scored if h[2]=="live_advisory"],
        "live_forecast": [h for h in scored if h[2]=="live_forecast"],
        "region_forecast":  [h for h in scored if h[2]=="region_forecast"],
    }
    for src, hits in by_src.items():
        if not hits: continue
        scores = [h[0] for h in hits]
        normed = minmax_norm(scores)
        for (orig_s, text, sname), ns in zip(hits, normed):
            normalized.append((ns, text, sname, orig_s))

    seen = set(); dedup = []
    for s, t, src, o in normalized:
        key = re.sub(r"\s+", " ", t.strip())
        if key in seen: continue
        seen.add(key); dedup.append((s, t, src, o))
    dedup.sort(key=lambda x: x[0], reverse=True)
    top = dedup[:7]

    ctx_parts = []
    # 라이브 상태 메시지는 질문 날짜가 오늘일 때만 추가 (특보는 현재 상태만 의미)
    if live_status_msg and days_from_today <= 0: 
        ctx_parts.append(live_status_msg)
    # RAGAS faithfulness 향상을 위해 메타데이터 제거하고 순수 텍스트만 사용
    ctx_parts += [txt for s, txt, src, _ in top]
    context = "\n\n".join(ctx_parts) or "관련 문서를 찾을 수 없습니다."

    return {**state, "context": context}

def _has_minimum_fields(ctx: str) -> bool:
    need_any = ["지역:", "현상:", "수준:", "하늘:", "강수확률:", "기온:"]
    hit = sum(1 for k in need_any if k in ctx)
    return hit >= 2

# ===== 검증 노드들 =====
# 1차 검증 (검색 품질) 임계값
CONTEXT_PRECISION_THRESHOLD = 0.7
CONTEXT_RECALL_THRESHOLD = 0.7

# 2차 검증 (답변 품질) 임계값
FAITHFULNESS_THRESHOLD = 0.7  # RAGAS faithfulness는 일반적으로 낮게 나옴 (0.4 → 0.35로 더 완화)
ANSWER_RELEVANCY_THRESHOLD = 0.7

MAX_RETRIES = 3

def retrieval_validation_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 1차 검증 (검색 품질)")
    question = state.get("question") or ""
    context = state.get("context") or ""

    if not context or "관련 문서를 찾을 수 없습니다." in context:
        print("   - ❌ 검색된 문서가 없어 불충분으로 판단합니다.")
        return {**state, "is_retrieval_sufficient": False}

    # RAGAS 평가
    ragas_scores = {"context_precision": 0.0}
    if _HAS_RAGAS and _RAGAS_LLM_WRAPPER:
        try:
            print("   - 📊 RAGAS 검색 품질 평가 중...")

            # 컨텍스트 최적화
            max_context_length = 2500
            optimized_context = context[:max_context_length] if len(context) > max_context_length else context

            # 임시 답변 생성 (LLMContextPrecisionWithoutReference용)
            temp_answer = optimized_context[:1200] if len(optimized_context) > 0 else "정보 부족"

            print(f"   - 📝 SingleTurnSample 준비: 질문={len(question)}자, 컨텍스트={len(optimized_context)}자")

            # SalesRAGAS 방식: SingleTurnSample 사용
            context_precision_scorer = LLMContextPrecisionWithoutReference(llm=_RAGAS_LLM_WRAPPER)
            
            # SingleTurnSample 생성
            context_sample = SingleTurnSample(
                user_input=question,
                response=temp_answer,
                retrieved_contexts=[optimized_context] if optimized_context else [""]
            )

            print("   - 🔄 RAGAS 평가 실행 중...")
            
            # SingleTurnSample 방식으로 평가 (동기 방식으로 변경)
            import asyncio
            context_precision_score = asyncio.run(context_precision_scorer.single_turn_ascore(context_sample))
            ragas_scores["context_precision"] = float(context_precision_score)
            
            print(f"   - 📈 검색 품질 지표:")
            print(f"     • Context Precision (LLM-based): {ragas_scores['context_precision']:.3f}")

        except Exception as e:
            print(f"   - ⚠️ RAGAS 검색 평가 실패: {e}")
    else:
        print("   - ⚠️ RAGAS 백엔드가 준비되지 않아 평가를 건너뜁니다.")

    # 개별 임계값 평가
    precision_sufficient = ragas_scores["context_precision"] >= CONTEXT_PRECISION_THRESHOLD
    is_sufficient = precision_sufficient
    
    print(f"   - 🎯 개별 평가 결과:")
    print(f"     • Context Precision: {ragas_scores['context_precision']:.3f} (임계값: {CONTEXT_PRECISION_THRESHOLD}) {'✅' if precision_sufficient else '❌'}")
    print(f"     • 최종 결과: {'✅ 충분' if is_sufficient else '⚠️ 불충분'}")
    
    return {**state, "is_retrieval_sufficient": is_sufficient}

def answer_validation_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 2차 검증 (답변 품질)")
    retry_count = state.get("retry_count", 0) + 1

    if not _HAS_RAGAS or not (_RAGAS_LLM and _RAGAS_EMB):
        print("   - ⚠️ RAGAS 백엔드가 준비되지 않아 검증을 건너뜁니다.")
        return {**state, "is_answer_sufficient": True, "retry_count": retry_count}

    question = state.get("question") or ""
    context = state.get("context") or ""
    answer = state.get("answer") or ""

    if not all([question, context, answer]):
        print("   - ❌ 평가 정보가 부족하여 검증을 건너뜁니다.")
        return {**state, "is_answer_sufficient": True, "retry_count": retry_count}

    # 컨텍스트 및 답변 최적화 (RAGAS faithfulness 향상을 위해 길이 증가)
    max_context_length = 5000  # 3000 → 5000으로 증가
    optimized_context = context[:max_context_length] if len(context) > max_context_length else context
    max_answer_length = 2000   # 1200 → 2000으로 증가
    optimized_answer = answer[:max_answer_length] if len(answer) > max_answer_length else answer
    
    # RAGAS faithfulness 향상을 위해 컨텍스트 정리 (메타데이터 제거)
    import re
    # [유사도:0.1234][live_forecast] 같은 메타데이터 제거
    cleaned_context = re.sub(r'\[유사도:[^\]]+\]\[[^\]]+\]\n?', '', optimized_context)
    # [LIVE_STATUS], [RISKS] 같은 태그도 제거
    cleaned_context = re.sub(r'\[[A-Z_]+\]', '', cleaned_context)
    # 연속된 공백 정리
    cleaned_context = re.sub(r'\n\s*\n', '\n\n', cleaned_context).strip()
    optimized_context = cleaned_context

    if len(optimized_context.strip()) < 50 or len(optimized_answer.strip()) < 20:
        print("   - ⚠️ 컨텍스트/답변이 너무 짧아 RAGAS 평가 생략")
        return {**state, "is_answer_sufficient": True, "retry_count": retry_count}

    print(f"   - 📝 답변 품질 평가 준비: 질문={len(question)}자, 컨텍스트={len(optimized_context)}자, 답변={len(optimized_answer)}자")

    try:
        print("   - 📊 RAGAS 답변 품질 평가 중...")
        
        scores = {}
        
        try:
            # Faithfulness (SalesRAGAS 방식)
            faithfulness_scorer = Faithfulness(llm=_RAGAS_LLM_WRAPPER)
            
            # SingleTurnSample 생성 (Faithfulness용)
            faithfulness_sample = SingleTurnSample(
                user_input=question,
                response=optimized_answer,
                retrieved_contexts=[optimized_context] if optimized_context else [""]
            )
            
            # SingleTurnSample 방식으로 평가 (동기 방식으로 변경)
            import asyncio
            faithfulness_score = asyncio.run(faithfulness_scorer.single_turn_ascore(faithfulness_sample))
            scores['faithfulness'] = float(faithfulness_score)
            
        except Exception as e:
            scores['faithfulness'] = 0.0
        
        try:
            # Answer Relevancy (SalesRAGAS 방식)
            answer_relevancy_scorer = ResponseRelevancy(
                llm=_RAGAS_LLM_WRAPPER, 
                embeddings=_RAGAS_EMB_WRAPPER
            )
            
            # SingleTurnSample 생성 (Answer Relevancy용)
            relevancy_sample = SingleTurnSample(
                user_input=question,
                response=optimized_answer,
                retrieved_contexts=[optimized_context] if optimized_context else [""]
            )
            
            # SingleTurnSample 방식으로 평가 (동기 방식으로 변경)
            import asyncio
            answer_relevancy_score = asyncio.run(answer_relevancy_scorer.single_turn_ascore(relevancy_sample))
            scores['answer_relevancy'] = float(answer_relevancy_score)
            
        except Exception as e:
            scores['answer_relevancy'] = 0.0

        f_val = scores.get('faithfulness', 0.0)
        r_val = scores.get('answer_relevancy', 0.0)

        if f_val is None or r_val is None:
            print("   - ⚠️ RAGAS 점수 NaN/None → 이번 라운드 통과로 처리")
            return {**state, "is_answer_sufficient": True, "retry_count": retry_count}

        # 개별 임계값 평가
        faithfulness_sufficient = f_val >= FAITHFULNESS_THRESHOLD
        relevancy_sufficient = r_val >= ANSWER_RELEVANCY_THRESHOLD
        is_sufficient = faithfulness_sufficient and relevancy_sufficient

        print(f"   - 📈 답변 품질 지표:")
        print(f"     • Faithfulness: {f_val:.3f} (임계값: {FAITHFULNESS_THRESHOLD}) {'✅' if faithfulness_sufficient else '❌'}")
        print(f"     • Answer Relevancy: {r_val:.3f} (임계값: {ANSWER_RELEVANCY_THRESHOLD}) {'✅' if relevancy_sufficient else '❌'}")
        print(f"     • 최종 결과: {'✅ 충분' if is_sufficient else '⚠️ 불충분'}")

        return {**state, "is_answer_sufficient": is_sufficient, "retry_count": retry_count}

    except Exception as e:
        print(f"   - ❌ 2차 검증 중 오류 발생: {e}")
        return {**state, "is_answer_sufficient": True, "retry_count": retry_count}

def validate_context_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 1차 컨텍스트 검증(예보관 체크)")
    context = state.get("context") or ""
    is_valid = _has_minimum_fields(context) and ("관련 문서를 찾을 수 없습니다." not in context)
    if not is_valid:
        print("⚠️ 컨텍스트 불충분 → 웹 검색 보강")
        return {**state, "is_context_valid": False, "retry_count": state["retry_count"] + 1}
    print("✅ 컨텍스트 충분")
    return {**state, "is_context_valid": True}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    print(f"🧩 노드: 웹 검색 (Tavily) | 재시도: {state['retry_count']}")
    if state["retry_count"] >= 2:
        print("⚠️ 웹 검색 재시도 초과.")
        return state
    
    # 인터넷 연결 확인
    if not check_internet_connection():
        print("❌ 인터넷 연결이 없어 웹 검색을 수행할 수 없습니다.")
        return {**state, "context": state["context"] + "\n\n[웹 검색 결과] 인터넷 연결이 없어 웹 검색을 수행할 수 없습니다."}
    
    question = state["question"] or ""
    if not TAVILY_API_KEY:
        print("⚠️ TAVILY_API_KEY 없음 → 스킵")
        return {**state, "context": state["context"] + "\n\n[웹 검색 결과] Tavily API 키가 없어 검색을 수행할 수 없습니다."}
    from tavily import TavilyClient
    client = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        results = client.search(query=question, max_results=TAVILY_MAX_RESULTS)
        web_search_results = "\n".join([f"- 출처: {res['url']}\n  내용: {res['content']}" for res in results['results']])
        web_result = f"""[웹 검색 결과]
질문 "{question}"에 대한 최신 정보입니다.
{web_search_results}
"""
        return {**state, "context": state["context"] + "\n\n" + web_result}
    except Exception as e:
        print(f"❌ Tavily 오류: {e}")
        return {**state, "context": state["context"] + "\n\n[웹 검색 결과] 웹 검색 실패."}

# ===== 대체 답변 노드 =====
def fallback_answer_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 대체 답변 생성")
    fallback_message = "죄송합니다. 해당 질문에 대한 충분한 정보를 찾을 수 없습니다. 다른 질문을 해주세요."
    return {**state, "answer": fallback_message}

def generate_draft_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 초안 생성")
    if not state.get("question"): raise ValueError("question 누락")
    if not state.get("context"): raise ValueError("context 누락")
    chain = (
        {"context": itemgetter("context"), "question": itemgetter("question")}
        | DRAFT_PROMPT
        | make_llm()
        | StrOutputParser()
    )
    ans = chain.invoke({"context": state["context"], "question": state["question"]})
    txt = re.sub(r'\n{3,}', '\n\n', ans or "").strip()
    return {**state, "answer_draft": txt}

def refine_answer_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 답변 개선/최종")
    if not state.get("question"): raise ValueError("question 누락")
    if not state.get("context"): raise ValueError("context 누락")
    if not state.get("answer_draft"): raise ValueError("answer_draft 누락")
    chain = (
        {"context": itemgetter("context"), "question": itemgetter("question"), "answer_draft": itemgetter("answer_draft")}
        | REFINE_PROMPT
        | make_llm()
        | StrOutputParser()
    )
    ans = chain.invoke({"context": state["context"], "question": state["question"], "answer_draft": state["answer_draft"]})
    txt = re.sub(r'\n{3,}', '\n\n', ans or "").strip()
    return {**state, "answer": txt}

# =========[ 그래프 빌드 ]=========
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("load_store", load_store_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("retrieval_validation", retrieval_validation_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate_draft", generate_draft_node)
    g.add_node("refine_answer", refine_answer_node)
    g.add_node("answer_validation", answer_validation_node)
    g.add_node("fallback_answer", fallback_answer_node)

    g.set_entry_point("load_store")
    g.add_edge("load_store", "retrieve")
    g.add_edge("retrieve", "retrieval_validation")

    # 1차 검증 결과에 따라 웹 검색 여부 결정
    g.add_conditional_edges(
        "retrieval_validation",
        lambda state: "sufficient" if state["is_retrieval_sufficient"] else "insufficient",
        {"sufficient": "generate_draft", "insufficient": "web_search"}
    )
    g.add_edge("web_search", "generate_draft")
    g.add_edge("generate_draft", "refine_answer")
    g.add_edge("refine_answer", "answer_validation")

    # 2차 검증 결과에 따라 종료/재시도/대체 답변 결정
    def decide_after_answer_validation(state: GraphState) -> str:
        if state["is_answer_sufficient"]:
            return "end"
        elif state["retry_count"] >= MAX_RETRIES:
            return "fallback"
        else:
            return "retry"

    g.add_conditional_edges(
        "answer_validation",
        decide_after_answer_validation,
        {"end": END, "fallback": "fallback_answer", "retry": "web_search"}
    )
    g.add_edge("fallback_answer", END)

    app = g.compile()
    # try:
    #     graph_image_path = "agent_workflow_v3.png"
    #     with open(graph_image_path, "wb") as f:
    #         f.write(app.get_graph().draw_mermaid_png())
    #     print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    # except Exception as e:
    #     print(f"그래프 시각화 중 오류: {e}")
    return app

# # =========[ 평가 유틸(선택) ]=========
# def _ensure_embedder():
#     global _text_embedder
#     if _text_embedder is None: _text_embedder = SentenceTransformer(EMBED_MODEL_NAME)
#     return _text_embedder

# def evaluate_goldenset(app, csv_path: str, limit: int = 50, out_path: str = "evaluation_results.csv"):
#     if not os.path.exists(csv_path): raise FileNotFoundError(f"골든셋 CSV를 찾을 수 없습니다: {csv_path}")
#     df = pd.read_csv(csv_path)
#     def _find_col(cands):
#         lc = {c.lower(): c for c in df.columns}
#         for k in cands:
#             if k in lc: return lc[k]
#         for c in df.columns:
#             if any(k in c for k in ["질문","question"]): return c
#         return None
#     q_col = _find_col(["question"]); a_col = _find_col(["answer","ground_truth","gt"])
#     if not q_col or not a_col: raise ValueError(f"CSV에 question/answer 컬럼을 찾을 수 없습니다. (발견된 컬럼: {list(df.columns)})")
#     eval_df = df.head(limit).copy()
#     preds, scores = [], []
#     emb_model = _ensure_embedder()
#     for idx, row in eval_df.iterrows():
#         q = str(row[q_col]).strip(); gt = str(row[a_col]).strip()
#         try:
#             out = app.invoke({"question": q})
#             pred = (out.get("answer") or "").strip()
#         except Exception as e:
#             pred = f"[오류] {e}"
#         if gt and pred and not pred.startswith("[오류]"):
#             vecs = emb_model.encode([gt, pred], show_progress_bar=False)
#             vecs = np.array([l2_normalize(v) for v in vecs], dtype="float32")
#             sim = float((vecs[0] * vecs[1]).sum())
#         else: sim = 0.0
#         preds.append(pred); scores.append(sim)
#         print(f"\n[{idx+1}/{limit}]")
#         print(f"질문: {q}")
#         print(f"정답: {gt}")
#         print(f"답변: {pred}")
#         print(f"유사도: {sim:.4f}")
#         print("-"*50)
#     eval_df["prediction"] = preds
#     eval_df["cosine_similarity"] = scores
#     eval_df["passed@0.75"] = eval_df["cosine_similarity"] >= 0.75
#     eval_df.to_csv(out_path, index=False, encoding="utf-8-sig")
#     print(f"\n전체 결과 저장: {out_path}")

# =========[ OchestratorTest.py 호환 함수 ]=========
def run(state: dict) -> dict:
    """
    OchestratorTest.py에서 호출되는 재해대응 에이전트 실행 함수
    
    Args:
        state: OchestratorTest.py에서 전달받은 상태 딕셔너리
               - query: 사용자 질문 (필수)
    
    Returns:
        dict: 실행 결과
            - agent_answer: 최종 답변
    """
    try:
        # 질문 추출
        query = state.get("query", "")
        if not query:
            return {"agent_answer": "질문이 제공되지 않았습니다. 재해 관련 질문을 해주세요."}
        
        print(f"[날씨_agent] 질문 처리 시작: {query}")
        
        # 그래프 빌드 및 실행
        app = build_graph()
        
        # 그래프 실행
        result = app.invoke({"question": query})
        
        # 답변 추출
        answer = result.get("answer", "답변을 생성할 수 없습니다.")
        
        print(f"[날씨_agent] 답변 생성 완료: {len(answer)}자")
        
        return {"agent_answer": answer}
        
    except Exception as e:
        error_msg = f"재해대응 에이전트 실행 중 오류가 발생했습니다: {e}"
        print(f"[날씨_agent] 오류: {e}")
        return {"agent_answer": error_msg}

# =========[ 실행부 ]=========
if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser
    parser = ArgumentParser(description="기상 전문가 통합 그래프 (라이브+중기, No VectorDB, No 2nd Validation)")
    parser.add_argument("-q", "--question", default=None, help="질문 1회 실행 후 종료")
    parser.add_argument("--show-context", action="store_true", help="컨텍스트(근거) 출력")
    args = parser.parse_args()

    print("💬 기상 전문가 그래프 (라이브+중기, API-only, 2차 검증 제거)")
    app = build_graph()

    if args.question:
        q = args.question.strip()
        if not q: raise ValueError("질문이 비어 있습니다.")
        try:
            out = app.invoke({"question": q})
            if args.show_context:
                print("\n=== 컨텍스트 ===")
                print(out.get("context", ""))
            print("\n=== 답변 ===")
            print(out.get("answer", ""))
            print()
        except Exception as e:
            print(f"❌ 오류: {e}\n")
    else:
        print("질문을 입력하세요. (종료: exit/quit)")
        while True:
            q = input("질문> ").strip()
            if q.lower() in ("exit", "quit"): break
            if not q: continue
            try:
                out = app.invoke({"question": q})
                if args.show_context:
                    print("\n=== 컨텍스트 ===")
                    print(out.get("context", ""))
                print("\n=== 답변 ===")
                print(out.get("answer", ""))
                print()
            except Exception as e:
                print(f"❌ 오류: {e}\n")
