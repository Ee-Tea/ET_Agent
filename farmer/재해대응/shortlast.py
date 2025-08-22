# -*- coding: utf-8 -*-
import os
import re
import json
from typing import TypedDict, Optional, Any, Dict, List
from datetime import datetime, timedelta
from operator import itemgetter
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import faiss
import requests
from urllib.parse import urlencode

from dotenv import load_dotenv
from tavily import TavilyClient
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# =========[ 환경설정 ]=========
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "faiss_kma_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
KMA_TIMEOUT = int(os.getenv("KMA_TIMEOUT", "10"))
WHEATHER_API_KEY_HUB = os.getenv("WHEATHER_API_KEY_HUB")
USE_KMA_LIVE = os.getenv("USE_KMA_LIVE", "true").lower() in ("1", "true", "yes")
FORCE_DISABLE_KMA_LIVE = False

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# =========[ LangChain/LangGraph/LLM ]=========
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

# =========[ 매핑/유틸 ]=========
WRN_MAP = {"T":"태풍","W":"강풍","R":"호우","C":"한파","D":"건조","O":"해일","N":"지진해일","V":"풍랑","S":"대설","Y":"황사","H":"폭염","F":"안개"}
LVL_MAP = {"1":"예비특보","2":"주의보","3":"경보"}
CMD_MAP = {"1":"발표","2":"대치","3":"해제","4":"대치해제","5":"연장","6":"변경","7":"변경해제"}
REGION_CODE_RE = re.compile(r"^[A-Z]\d{7}$")
REGION_MAP = {}
_text_embedder = None

# 단기예보 매핑
SKY_MAP = {"DB01": "맑음", "DB02": "구름조금", "DB03": "구름많음", "DB04": "흐림"}
PREP_MAP = {"0": "없음", "1": "비", "2": "비/눈", "3": "눈", "4": "눈/비"}
WIND_KO = {
    "N":"북","NNE":"북북동","NE":"북동","ENE":"동북동","E":"동","ESE":"동남동","SE":"남동","SSE":"남남동",
    "S":"남","SSW":"남남서","SW":"남서","WSW":"서남서","W":"서","WNW":"서북서","NW":"북서","NNW":"북북서"
}

# =========[ 지역 코드 매핑: 두 CSV 병합 ]=========
# .env 예시:
# KMA_ADVISORY_MAP_CSV=./warn_regions.csv   # 기상특보용
# KMA_FCT_MAP_CSV=./regions.csv             # 단기예보용

REGION_MAP: Dict[str, str] = {}          # code -> name
REGION_NAME_INDEX: Dict[str, List[str]] = {}  # normalized_name -> [codes]

def _norm_name(s: str) -> str:
    # 공백/ㆍ/·/괄호 제거 후 소문자화
    return re.sub(r"[()\s·ㆍ]", "", str(s or "")).lower()

def _pick_cols(df: pd.DataFrame) -> tuple:
    cols = {c.lower(): c for c in df.columns}
    code_col = next((cols[k] for k in ("code","region_code","지역코드","reg_id","regid","id") if k in cols), None)
    name_col = next((cols[k] for k in ("name","region_name","지역명","reg_name","regname","ko_name","한글명") if k in cols), None)

    # 못 찾으면 패턴으로 추정
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

def _load_region_maps_from_two_csvs():
    REGION_MAP.clear()
    REGION_NAME_INDEX.clear()

    paths = [os.getenv("KMA_ADVISORY_MAP_CSV"), os.getenv("KMA_FCT_MAP_CSV")]
    total = 0
    for p in paths:
        for code, name in _read_map_csv(p):
            REGION_MAP[code] = name   # 중복 코드는 마지막 항목으로 덮어씀 (최신/정확 CSV가 우선)
            REGION_NAME_INDEX.setdefault(_norm_name(name), []).append(code)
            total += 1

    used = ", ".join([p for p in paths if p]) or "N/A"
    print(f"✅ 지역코드 매핑 로드: {len(REGION_MAP)}개 (from: {used})")

_load_region_maps_from_two_csvs()  # <-- 기존 _load_region_map_from_csv() 호출을 이걸로 교체

def resolve_region(token: str) -> str:
    if not token: return "N/A"
    t = token.strip()
    return REGION_MAP.get(t, t)

def fmt_kst(yyyymmddHHMM: str) -> str:
    try:
        dt = datetime.strptime(yyyymmddHHMM, "%Y%m%d%H%M")
        return dt.strftime("%Y-%m-%d %H:%M KST")
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

# =========[ KMA 특보/예보 가져오기 ]=========
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
        "window_start_kst": fmt_kst(tm_st) if tm_st!="N/A" else "N/A",
        "window_end_kst": fmt_kst(tm_ed) if tm_ed!="N/A" else "N/A",
        "announce_time_kst": fmt_kst(tm_st) if cmd=="1" and tm_st!="N/A" else None,
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
        f"종류: {payload['hazard_name']}({payload['hazard_code']})",
        f"수준: {payload['level_name']}({payload['level_code']})",
        f"명령: {payload['command_name']}({payload['command_code']})"
    ]
    if "typhoon_grade" in payload: parts.append(f"태풍 등급: {payload['typhoon_grade']}")
    return {
        "json": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "human": " | ".join(parts)
    }

def fetch_kma_advisories(start_time: str, end_time: str, disp: str="1") -> List[Dict[str, str]]:
    if not WHEATHER_API_KEY_HUB: return []
    base = "https://apihub.kma.go.kr/api/typ01/url/wrn_met_data.php"
    params = {"authKey":WHEATHER_API_KEY_HUB, "wrn":"", "tmfc1":start_time, "tmfc2":end_time, "disp":disp}
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
        print(f"❌ 기상특보 API 호출 중 오류 발생: {e}")
        return []

def _format_short_land_record(raw: list) -> Dict[str, str]:
    def g(i): return raw[i] if i < len(raw) else ""
    reg_id   = g(0)
    tm_fc    = g(1)
    tm_ef    = g(2)
    mod      = g(3)
    ne       = g(4)
    w1, w2   = g(9), g(11)
    ta, st   = g(12), g(13)
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

    human_summary = f"{reg_name} 지역의 날씨는 {payload['sky_status']} 상태이며, 기온은 {payload['temp']}이고 강수확률은 {payload['precip_prob']}입니다."
    if payload['wind_direction']:
        human_summary += f" 바람은 {payload['wind_direction']}으로 붑니다."
    
    return {
        "json": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "human": human_summary
    }

def fetch_short_land_records() -> list:
    if not WHEATHER_API_KEY_HUB: return []
    BASE = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_dl.php"
    params = {"reg": "", "tmfc": "0", "disp": "1", "authKey": WHEATHER_API_KEY_HUB}
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
            if len(raw_row) < 17: continue
            formatted_record = _format_short_land_record(raw_row)
            docs.append(formatted_record)
        return docs
    except Exception as e:
        print(f"❌ 단기예보 API 호출 중 오류 발생: {e}")
        return []

def _reverse_region_map() -> Dict[str, List[str]]:
    """정규화된 지역명 -> 코드목록 (두 CSV 합본 기반)"""
    return {k: list(dict.fromkeys(v)) for k, v in REGION_NAME_INDEX.items()}

def extract_region_from_question(q: str) -> Optional[str]:
    """질문에서 지역명을 찾아 '정식 지역명'으로 반환 (코드/한글 모두 인식)"""
    if not q:
        return None

    # 코드 직접 입력 처리 (예: 11B00000)
    m = re.search(r"[A-Z]\d{7}", q)
    if m:
        code = m.group(0)
        name = REGION_MAP.get(code)
        if name:
            return name

    # 한글 지역명(정규화) 매칭
    qn  = _norm_name(q)
    rev = _reverse_region_map()
    for norm_name in sorted(rev.keys(), key=len, reverse=True):
        if norm_name and norm_name in qn:
            for c in rev[norm_name]:
                nm = REGION_MAP.get(c)
                if nm:
                    return nm

    # 관용/약칭 보정
    commons = ["울진군", "울진", "영덕군", "영덕", "부산동부", "울산동부", "거제시",
               "남해군", "제주도북부", "제주북부", "사천시", "파주시"]
    for token in commons:
        tn = _norm_name(token)
        if tn in qn:
            for code, name in REGION_MAP.items():
                if tn in _norm_name(name):
                    return name
    return None


def summarize_region_alert(q: str, live_docs: List[Dict[str, str]]) -> str:
    region_name = extract_region_from_question(q)
    if not region_name: return "" 
    rev = _reverse_region_map()
    target_codes = rev.get(region_name, [])
    if not target_codes:
        return f"[LIVE_STATUS] {region_name}의 특보 정보를 찾을 수 없습니다. 지역 코드 매핑을 확인해주세요."
    matched = []
    for d in live_docs or []:
        try:
            payload = json.loads(d["json"])
            if payload.get("region_raw") in target_codes: matched.append(payload)
        except Exception: continue
    if not matched:
        return f"[LIVE_STATUS] 오늘 {region_name}에는 발효 중인 특보가 없습니다. 평소와 같이 관리하시면 됩니다."
    matched.sort(key=lambda x: x.get("window_start", ""), reverse=True)
    p = matched[0]
    region, hz = p.get("region_name", region_name), p.get("hazard_name", "특보")
    lvl, cmd = p.get("level_name", ""), p.get("command_name", "")
    st, ed = p.get("window_start_kst", ""), p.get("window_end_kst", "")
    bits = [f"{region}에는 {hz} {lvl}가"]
    if st and ed: bits.append(f"{st}에 발효되어 {ed}에 해제됩니다.")
    elif st: bits.append(f"{st}에 발효되었습니다.")
    else: bits.append("발효 중입니다.")
    if cmd: bits.append(f"(명령: {cmd})")
    return "[LIVE_STATUS] " + " ".join(bits)

# =========[ 임베딩 (라이브 검색용) ]=========
_text_embedder = SentenceTransformer(EMBED_MODEL_NAME)
def embed_texts(texts: List[str]) -> np.ndarray:
    embs = _text_embedder.encode(texts, show_progress_bar=False)
    embs = np.array([l2_normalize(e) for e in embs], dtype="float32")
    return embs

# =========[ 상태/프롬프트/LLM ]=========
class GraphState(TypedDict):
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    answer_draft: Optional[str]
    store_obj: Optional[Any]
    retry_count: int
    is_context_valid: Optional[bool]
    next_action: Optional[str]

class ValidationResult(BaseModel):
    is_valid: bool
    is_grounded: bool
    reasoning: str

def make_llm() -> ChatGroq:
    if not GROQ_API_KEY: raise ValueError("GROQ_API_KEY가 .env에 없습니다.")
    return ChatGroq(model_name=GROQ_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY)

def _should_use_live(q: str) -> bool:
    if FORCE_DISABLE_KMA_LIVE or not USE_KMA_LIVE: return False
    q = (q or "")
    live_kw = ["현재", "지금", "오늘", "특보", "실시간", "경보", "주의보", "해제", "발표", "날씨", "기온", "강수"]
    return any(k in q for k in live_kw)

# =========[ LangGraph 노드 ]=========
def load_store_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 벡터스토어 로드 (LangChain-FAISS)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True}
    )
    vs = LCFAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return {**state, "store_obj": vs, "retry_count": 0}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print(f"🧩 노드: 검색(MMR + KMA 라이브 병합) | 재시도 횟수: {state['retry_count']}")
    q = state["question"] or ""
    live_enabled = _should_use_live(q)
    
    retriever = state["store_obj"].as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
    docs = retriever.invoke(q)
    persist_scored = [(0.2, d.page_content, "persist") for d in docs]
    
    live_scored: List[tuple] = []
    live_status_msg = ""

    if live_enabled:
        # 기상특보 데이터 가져오기 및 임베딩
        try:
            now = datetime.now()
            tm1 = now.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M")
            tm2 = now.strftime("%Y%m%d%H%M")
            advisories = fetch_kma_advisories(tm1, tm2) if WHEATHER_API_KEY_HUB else []
            live_status_msg = summarize_region_alert(q, advisories)
            if advisories:
                human_texts = [d["human"] for d in advisories]
                embs = embed_texts(human_texts)
                live_idx = faiss.IndexFlatIP(embs.shape[1])
                live_idx.add(embs)
                qv = embed_texts([q])[0]
                topk = min(5, len(advisories))
                D, I = live_idx.search(np.array([qv], dtype="float32"), topk)
                for s, i in zip(D[0], I[0]):
                    if i == -1: continue
                    ctx = f"JSON: {advisories[i]['json']}\n요약: {advisories[i]['human']}"
                    live_scored.append((float(s) + 1.0, ctx, "live_advisory"))
        except Exception as e:
            print(f"❌ 기상특보 검색 중 오류 발생: {e}")
        
        # 단기예보 데이터 가져오기 및 임베딩
        try:
            forecasts = fetch_short_land_records()
            if forecasts:
                human_texts = [d["human"] for d in forecasts]
                embs = embed_texts(human_texts)
                live_idx = faiss.IndexFlatIP(embs.shape[1])
                live_idx.add(embs)
                qv = embed_texts([q])[0]
                topk = min(5, len(forecasts))
                D, I = live_idx.search(np.array([qv], dtype="float32"), topk)
                for s, i in zip(D[0], I[0]):
                    if i == -1: continue
                    ctx = f"JSON: {forecasts[i]['json']}\n요약: {forecasts[i]['human']}"
                    live_scored.append((float(s) + 1.0, ctx, "live_forecast"))
        except Exception as e:
            print(f"❌ 단기예보 검색 중 오류 발생: {e}")
            
    else:
        print("ℹ️ KMA 라이브 호출 건너뜀 (실시간 의도 아님 또는 USE_KMA_LIVE=False)")
    
    merged = live_scored + persist_scored
    normalized: List[tuple] = []
    by_src = {"persist": [h for h in merged if h[2]=="persist"], 
              "live_advisory": [h for h in merged if h[2]=="live_advisory"],
              "live_forecast": [h for h in merged if h[2]=="live_forecast"]}
    
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
    top = dedup[:5]
    
    ctx_parts = []
    if live_status_msg: ctx_parts.append(live_status_msg)
    ctx_parts += [f"[유사도:{s:.4f}][{src}]\n{txt}" for s, txt, src, _ in top]
    context = "\n\n".join(ctx_parts) or "관련 문서를 찾을 수 없습니다."
    
    return {**state, "context": context}

def validate_context_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 1차 컨텍스트 검증")
    context = state["context"] or ""
    is_valid = not (
        "관련 문서를 찾을 수 없습니다." in context or
        (state.get("is_context_valid") is None and "LIVE_STATUS" in context and len(context.split('\n')) < 3)
    )
    if not is_valid:
        print("⚠️ 컨텍스트가 불충분하여 웹 검색이 필요합니다.")
        return {**state, "is_context_valid": False, "retry_count": state["retry_count"] + 1}
    print("✅ 컨텍스트가 충분합니다.")
    return {**state, "is_context_valid": True}

def web_search_node(state: GraphState) -> Dict[str, Any]:
    print(f"🧩 노드: 웹 검색 (Tavily Search API 사용) | 현재 재시도 횟수: {state['retry_count']}")
    if state["retry_count"] >= 2:
        print("⚠️ 웹 검색 재시도 횟수 초과. 최종 실패 노드로 이동합니다.")
        return {**state, "next_action": "fallback"}
    
    question = state["question"] or ""
    if not TAVILY_API_KEY:
        print("⚠️ TAVILY_API_KEY가 .env에 없습니다. 웹 검색을 건너뜁니다.")
        return {**state, "context": state["context"] + "\n\n[웹 검색 결과] Tavily API 키가 없어 검색을 수행할 수 없습니다."}

    client = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        results = client.search(query=question, max_results=TAVILY_MAX_RESULTS)
        web_search_results = "\n".join([
            f"- 출처: {res['url']}\n 내용: {res['content']}"
            for res in results['results']
        ])
        web_result = f"""[웹 검색 결과]
질문 "{question}"에 대한 최신 정보입니다.
{web_search_results}
"""
        return {**state, "context": state["context"] + "\n\n" + web_result}
    except Exception as e:
        print(f"❌ Tavily 웹 검색 중 오류 발생: {e}")
        return {**state, "context": state["context"] + "\n\n[웹 검색 결과] 웹 검색에 실패했습니다. 다음을 참고하세요."}

DRAFT_PROMPT = ChatPromptTemplate.from_template(
    """너는 농작물 재해 대응 전문가야.
아래 문맥에는 JSON 특보와 일반 문서가 섞여 있어. 문맥에서 사실만 선별하여 질문에 대한 초안 답변을 작성해.
불릿, 번호 목록, 코드블록, JSON, 표는 사용하지 마. 중복되거나 상충되는 내용은 정리하고, 특보는 지역, 종류, 수준, 발표시각, 기간을 포함해서 자연스러운 문장으로 설명해. 단기예보는 지역, 기온, 하늘 상태, 강수 확률을 포함해서 자연스러운 문장으로 설명해.

[문맥]
{context}

질문: {question}
초안 답변:"""
)

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

REFINE_PROMPT = ChatPromptTemplate.from_template(
    """다음은 질문과, 그에 대한 초안 답변, 그리고 답변의 근거가 된 문맥이야.
너는 농작물 재해 대응 전문가로서, 초안 답변이 질문의 의도를 완전히 충족하는지, 그리고 문맥에 없는 내용(환각)을 포함하고 있지는 않은지 엄격하게 검토하고 최종 답변을 작성해줘.
만약 초안 답변이 불완전하거나 문맥과 맞지 않는다면, 문맥을 기반으로 더 정확하고 완전한 답변으로 수정해야 해.
최종 답변은 한국어 자연어 문장으로만 작성하고, 불필요한 서식은 사용하지 마.

[문맥]
{context}

질문: {question}
초안 답변: {answer_draft}

최종 답변:"""
)

def refine_answer_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 답변 개선 및 최종 생성")
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

VALIDATION_PROMPT = ChatPromptTemplate.from_template(
    """주어진 질문, 답변, 그리고 답변의 근거가 된 문맥을 바탕으로 다음 세 가지를 엄격하게 평가해줘.
너는 오직 **JSON 객체**로만 응답해야 하며, 어떠한 추가 설명이나 텍스트도 포함해서는 안 돼. 응답은 반드시 다음 스키마를 따라야 해:

```json
{{
    "is_valid": boolean,
    "is_grounded": boolean,
    "reasoning": string
}}

질문: {question}
문맥: {context}
답변: {answer}
응답:"""
)

def post_generate_validate_node(state: GraphState) -> Dict[str, Any]:
    print(f"🧩 노드: 2차 답변 검증 (LLM 활용) | 재시도 횟수: {state['retry_count']}")
    q = state["question"] or ""
    a = state["answer"] or ""
    ctx = state["context"] or ""

    if state["retry_count"] >= 3:
        print("⚠️ 재시도 횟수 초과. 최종 실패 노드로 이동합니다.")
        return {**state, "next_action": "fallback"}

    validation_chain = (
        VALIDATION_PROMPT
        | make_llm()
        | StrOutputParser()
    )

    is_valid_answer, is_grounded = False, False

    try:
        raw_output = validation_chain.invoke({"question": q, "context": ctx, "answer": a})
        print(f"LLM 원본 응답: {raw_output[:100]}...")

        match = re.search(r'```json\s*(\{.*\})\s*```', raw_output, re.DOTALL)
        if not match:
            match = re.search(r'(\{.*\})', raw_output, re.DOTALL)
        
        if not match:
            raise ValueError("LLM 응답에서 유효한 JSON 객체를 찾을 수 없습니다.")

        json_str = match.group(1)
        json_obj = json.loads(json_str)

        validation_result = ValidationResult(**json_obj)
        
        is_valid_answer = validation_result.is_valid
        is_grounded = validation_result.is_grounded
        print(f"➡️ LLM 평가 결과: 유효성={is_valid_answer}, 근거 기반={is_grounded}, 이유: {validation_result.reasoning}")

    except Exception as e:
        print(f"❌ JSON 파싱 또는 모델 유효성 검사 중 오류 발생: {e}")
        is_valid_answer, is_grounded = False, False

    if is_valid_answer and is_grounded:
        print("✅ 답변이 충분하고 근거가 명확합니다. 종료합니다.")
        return {**state, "next_action": "END"}
    elif not is_grounded and state["retry_count"] < 2:
        print("⚠️ 답변이 문맥에 근거하지 않아 재검색이 필요합니다.")
        return {**state, "next_action": "re_search", "retry_count": state["retry_count"] + 1}
    elif not is_valid_answer:
        print("⚠️ 답변이 불완전하여 다시 생성합니다.")
        return {**state, "next_action": "refine_again", "retry_count": state["retry_count"] + 1}
    else:
        print("⚠️ 답변이 불충분하지만, 더 이상 재시도하지 않고 최종 실패 노드로 이동합니다.")
        return {**state, "next_action": "fallback"}

def fallback_answer_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 최종 실패 답변")
    fallback_message = (
        "죄송합니다. 질문에 대한 정확한 정보를 찾기 위해 여러 번 웹 검색을 시도했으나, 신뢰할 수 있는 정보를 확보하는 데 실패했습니다. "
        "다른 질문을 해주시면 최선을 다해 답변해 드리겠습니다."
    )
    return {**state, "answer": fallback_message}

# =========[ 그래프 빌드 ]=========
def build_graph():
    g = StateGraph(GraphState)
    
    g.add_node("load_store", load_store_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("validate_context", validate_context_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate_draft", generate_draft_node)
    g.add_node("refine_answer", refine_answer_node)
    g.add_node("post_generate_validate", post_generate_validate_node)
    g.add_node("fallback_answer", fallback_answer_node)

    g.set_entry_point("load_store")
    g.add_edge("load_store", "retrieve")
    g.add_edge("retrieve", "validate_context")

    g.add_conditional_edges(
        "validate_context",
        lambda state: "continue" if state["is_context_valid"] else "fallback",
        {"continue": "generate_draft", "fallback": "web_search"}
    )
    
    g.add_edge("web_search", "generate_draft")
    g.add_edge("generate_draft", "refine_answer")
    g.add_edge("refine_answer", "post_generate_validate")

    g.add_conditional_edges(
        "post_generate_validate",
        lambda state: state["next_action"],
        {
            "re_search": "web_search",
            "refine_again": "refine_answer",
            "END": END,
            "fallback": "fallback_answer"
        }
    )
    
    g.add_edge("fallback_answer", END)
    
    app = g.compile()
    
    try:
        graph_image_path = "agent_workflow.png"
        with open(graph_image_path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")

    return app

# =========[ 평가 유틸 ]=========
def _ensure_embedder():
    global _text_embedder
    if _text_embedder is None: _text_embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _text_embedder

def evaluate_goldenset(app, csv_path: str, limit: int = 50, out_path: str = "evaluation_results.csv"):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"골든셋 CSV를 찾을 수 없습니다: {csv_path}")
    df = pd.read_csv(csv_path)
    def _find_col(cands):
        lc = {c.lower(): c for c in df.columns}
        for k in cands:
            if k in lc: return lc[k]
        for c in df.columns:
            if any(k in c for k in ["질문","question"]): return c
        return None
    q_col = _find_col(["question"])
    a_col = _find_col(["answer","ground_truth","gt"])
    if not q_col or not a_col: raise ValueError(f"CSV에 question/answer 컬럼을 찾을 수 없습니다. (발견된 컬럼: {list(df.columns)})")
    eval_df = df.head(limit).copy()
    preds, scores = [], []
    emb_model = _ensure_embedder()
    for idx, row in eval_df.iterrows():
        q = str(row[q_col]).strip()
        gt = str(row[a_col]).strip()
        try:
            out = app.invoke({"question": q})
            pred = (out.get("answer") or "").strip()
        except Exception as e:
            pred = f"[오류] {e}"
        if gt and pred and not pred.startswith("[오류]"):
            vecs = emb_model.encode([gt, pred], show_progress_bar=False)
            vecs = np.array([l2_normalize(v) for v in vecs], dtype="float32")
            sim = float((vecs[0] * vecs[1]).sum())
        else: sim = 0.0
        preds.append(pred)
        scores.append(sim)
        print(f"\n[{idx+1}/{limit}]")
        print(f"질문: {q}")
        print(f"정답: {gt}")
        print(f"답변: {pred}")
        print(f"유사도: {sim:.4f}")
        print("-"*50)
    eval_df["prediction"] = preds
    eval_df["cosine_similarity"] = scores
    eval_df["passed@0.75"] = eval_df["cosine_similarity"] >= 0.75
    eval_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n전체 결과 저장: {out_path}")

# =========[ 실행부 ]=========
if __name__ == "__main__":
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser(description="Live+Persist RAG: 대화형 기본 / --eval 시 평가")
    parser.add_argument("-q", "--question", default=None, help="한 번만 질문하고 종료")
    parser.add_argument("--show-context", action="store_true", help="컨텍스트(근거)도 함께 출력")
    parser.add_argument("--eval", dest="eval_csv", default=None, help="골든셋 CSV 경로(지정 시 평가 모드)")
    parser.add_argument("--n", dest="limit", type=int, default=50, help="평가 표본 개수(최대)")
    parser.add_argument("--out", dest="out_path", default="evaluation_results.csv", help="평가 결과 저장 경로")
    args = parser.parse_args()
    print("💬 LangGraph Live+Persist (LC-FAISS only)")
    app = build_graph()
    if args.eval_csv:
        evaluate_goldenset(app, csv_path=args.eval_csv, limit=args.limit, out_path=args.out_path)
        sys.exit(0)
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