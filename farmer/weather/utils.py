# -*- coding: utf-8 -*-
"""
공통 유틸리티 함수들
날씨 에이전트에서 사용하는 공통 함수들
"""

import os
import re
import json
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
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

def load_region_map():
    """CSV 파일에서 지역 매핑 로드"""
    global REGION_MAP
    csv_path = os.getenv("REGION_CSV_PATH", "farmer/all_regions_combined.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV 파일이 없습니다: {csv_path}")
        REGION_MAP = {}
        return
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # 컬럼 자동 감지
        code_col = None
        name_col = None
        
        for col in df.columns:
            if col.upper() in ['REG_ID', 'CODE', 'ID']:
                code_col = col
            elif col.upper() in ['REG_NAME', 'NAME', 'REGION_NAME']:
                name_col = col
        
        if not code_col or not name_col:
            print("❌ CSV에서 코드/이름 컬럼을 찾을 수 없습니다.")
            REGION_MAP = {}
            return
        
        # 데이터 로드
        REGION_MAP = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            if code and name and code != 'nan' and name != 'nan':
                REGION_MAP[name] = code
        
        print(f"✅ 지역 매핑 로드 완료: {len(REGION_MAP)}개 지역")
        
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
    
    # 문서 텍스트 추출
    human_texts = [doc["human"] for doc in documents]
    
    # 임베딩 계산
    doc_embeddings = embed_texts(human_texts)
    query_embedding = embed_texts([query])
    
    # FAISS 인덱스 생성
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    
    # 유사도 검색
    scores, indices = index.search(query_embedding, min(top_k, len(documents)))
    
    # 결과 반환
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((float(score), documents[idx]))
    
    return results

def combine_weather_data(state: Dict[str, Any], max_docs_per_type: int = 3) -> str:
    """날씨 데이터 통합 (토큰 제한 고려)"""
    context_parts = []
    
    # 기상특보 데이터 (최대 3개)
    if "advisory_data" in state and state["advisory_data"]:
        for doc in state["advisory_data"][:max_docs_per_type]:
            formatted = _format_for_llm("live_advisory", doc["json"], doc["human"])
            context_parts.append(formatted)
    
    # 단기예보 데이터 (최대 3개)
    if "short_forecast_data" in state and state["short_forecast_data"]:
        for doc in state["short_forecast_data"][:max_docs_per_type]:
            formatted = _format_for_llm("live_forecast", doc["json"], doc["human"])
            context_parts.append(formatted)
    
    # 중기예보 데이터 (최대 3개)
    if "mid_forecast_data" in state and state["mid_forecast_data"]:
        for doc in state["mid_forecast_data"][:max_docs_per_type]:
            formatted = _format_for_llm("region_forecast", doc["json"], doc["human"])
            context_parts.append(formatted)
    
    if context_parts:
        return "\n\n".join(context_parts)
    else:
        return "NO_DATA_AVAILABLE"

# 지역 매핑 (CSV 파일에서 로드)
REGION_MAP = {}

def load_region_map():
    """CSV 파일에서 지역 매핑 로드"""
    global REGION_MAP
    csv_path = os.getenv("REGION_CSV_PATH", "farmer/all_regions_combined.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV 파일이 없습니다: {csv_path}")
        REGION_MAP = {}
        return
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # 컬럼 자동 감지
        code_col = None
        name_col = None
        
        for col in df.columns:
            if col.upper() in ['REG_ID', 'CODE', 'ID']:
                code_col = col
            elif col.upper() in ['REG_NAME', 'NAME', 'REGION_NAME']:
                name_col = col
        
        if not code_col or not name_col:
            print("❌ CSV에서 코드/이름 컬럼을 찾을 수 없습니다.")
            REGION_MAP = {}
            return
        
        # 데이터 로드
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            
            if code and name and code.lower() != "nan" and name.lower() != "nan":
                REGION_MAP[code] = name
        
        print(f"✅ 지역 매핑 로드 완료: {len(REGION_MAP)}개 지역")
        
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        REGION_MAP = {}

# 모듈 로드 시 지역 매핑 초기화
load_region_map()

def extract_region_from_question(question: str) -> str:
    """질문에서 지역 추출 (CSV 기반)"""
    if not question:
        return "서울"
    
    # CSV에서 로드된 지역 매핑 사용
    if REGION_MAP:
        # 질문에서 지역 코드나 지역명 찾기
        for code, name in REGION_MAP.items():
            if code in question or name in question:
                return name
    
    # CSV가 없으면 기본값 반환
    return "서울"

def extract_date_from_question(question: str) -> datetime:
    """질문에서 날짜 추출 (간단한 버전)"""
    if not question:
        return datetime.now(tz=KST)
    
    now = datetime.now(tz=KST)
    
    if "오늘" in question or "현재" in question or "지금" in question:
        return now
    elif "내일" in question:
        return now + timedelta(days=1)
    elif "모레" in question:
        return now + timedelta(days=2)
    elif "3일" in question:
        return now + timedelta(days=3)
    elif "4일" in question:
        return now + timedelta(days=4)
    elif "5일" in question:
        return now + timedelta(days=5)
    elif "6일" in question:
        return now + timedelta(days=6)
    elif "7일" in question:
        return now + timedelta(days=7)
    elif "1주" in question or "일주일" in question:
        return now + timedelta(days=7)
    elif "2주" in question:
        return now + timedelta(days=14)
    
    return now  # 기본값
