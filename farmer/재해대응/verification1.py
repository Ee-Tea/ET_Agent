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

from dotenv import load_dotenv
load_dotenv()

# =========[ 환경설정 ]=========
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "faiss_kma_db") # LangChain-FAISS 디렉토리 필수
GROQ_API_KEY=REDACTED("GROQ_API_KEY=REDACTED = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
KMA_TIMEOUT = int(os.getenv("KMA_TIMEOUT", "10"))
WHEATHER_API_KEY_HUB=REDACTED("WHEATHER_API_KEY_HUB")
USE_KMA_LIVE = os.getenv("USE_KMA_LIVE", "true").lower() in ("1", "true", "yes")
FORCE_DISABLE_KMA_LIVE = False

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

def _load_region_map_from_csv():
    path = os.getenv("REGION_MAP_CSV")
    if not path or not os.path.exists(path):
        return
    try:
        df = pd.read_csv(path)
        code_col = next((c for c in df.columns if c.lower() in ("code","region_code","지역코드")), None)
        name_col = next((c for c in df.columns if c.lower() in ("name","region_name","지역명")), None)
        if not code_col or not name_col: return
        for _, r in df.iterrows():
            code = str(r[code_col]).strip()
            name = str(r[name_col]).strip()
            if code and name: REGION_MAP[code] = name
    except Exception:
        pass
_load_region_map_from_csv()

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

# =========[ KMA 가져오기/요약 ]=========
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
        "source":"KMA",
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

def fetch_kma_alerts(start_time: str, end_time: str, disp: str="1") -> List[Dict[str, str]]:
    if not WHEATHER_API_KEY_HUB: return []
    base = "https://apihub.kma.go.kr/api/typ01/url/wrn_met_data.php"
    params = {"authKey":WHEATHER_API_KEY_HUB, "wrn":"", "tmfc1":start_time, "tmfc2":end_time, "disp":disp}
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

def _reverse_region_map() -> Dict[str, List[str]]:
    rev: Dict[str, List[str]] = {}
    for code, name in REGION_MAP.items():
        rev.setdefault(name, []).append(code)
        alias = name.replace(" ", "")
        rev.setdefault(alias, []).append(code)
    return rev

def extract_region_from_question(q: str) -> Optional[str]:
    if not q: return None
    q_norm = re.sub(r"\s+", "", q)
    cand_names = set(REGION_MAP.values())
    for nm in cand_names:
        if nm in q or nm.replace(" ", "") in q_norm: return nm
    commons = ["울진군", "울진", "영덕군", "영덕", "부산동부", "울산동부", "거제시", "남해군", "제주도북부", "제주북부", "사천시", "파주시"]
    for token in commons:
        if token in q or token.replace(" ", "") in q_norm:
            for code, name in REGION_MAP.items():
                if token.replace(" ", "") in name.replace(" ", ""): return name
    return None

def summarize_region_alert(q: str, live_docs: List[Dict[str, str]]) -> str:
    """질문에서 지역을 뽑아 매핑 실패/특보 없음/특보 있음 문구를 결정적으로 만들어줌."""
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
    store_obj: Optional[Any]
    retry_count: int
    is_context_valid: Optional[bool]
    next_action: Optional[str]

def make_llm() -> ChatGroq:
    if not GROQ_API_KEY=REDACTED ValueError("GROQ_API_KEY=REDACTED에 없습니다.")
    return ChatGroq(model_name=GROQ_MODEL, temperature=TEMPERATURE, api_key=GROQ_API_KEY=REDACTED _should_use_live(q: str) -> bool:
    if FORCE_DISABLE_KMA_LIVE or not USE_KMA_LIVE: return False
    q = (q or "")
    live_kw = ["현재", "지금", "오늘", "특보", "실시간", "경보", "주의보", "해제", "발표"]
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
    live_docs: List[Dict[str, str]] = []
    live_status_msg = ""
    if live_enabled:
        try:
            now = datetime.now()
            tm1 = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M")
            tm2 = now.strftime("%Y%m%d%H%M")
            live_docs = fetch_kma_alerts(tm1, tm2) if WHEATHER_API_KEY_HUB else []
            live_status_msg = summarize_region_alert(q, live_docs)
            if live_docs:
                human_texts = [d["human"] for d in live_docs]
                embs = embed_texts(human_texts)
                live_idx = faiss.IndexFlatIP(embs.shape[1])
                live_idx.add(embs)
                qv = embed_texts([q])[0]
                topk = min(5, len(live_docs))
                D, I = live_idx.search(np.array([qv], dtype="float32"), topk)
                for s, i in zip(D[0], I[0]):
                    if i == -1: continue
                    ctx = f"JSON: {live_docs[i]['json']}\n요약: {live_docs[i]['human']}"
                    live_scored.append((float(s) + 1.0, ctx, "live"))
        except Exception:
            pass
    else:
        print("ℹ️ KMA 라이브 호출 건너뜀 (실시간 의도 아님 또는 USE_KMA_LIVE=False)")
    merged = live_scored + persist_scored
    normalized: List[tuple] = []
    bysrc = {"persist": [h for h in merged if h[2]=="persist"], "live": [h for h in merged if h[2]=="live"]}
    for src, hits in bysrc.items():
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
    print("🧩 노드: 웹 검색 (실제 웹 검색 API 호출 대신 더미 데이터 반환)")
    question = state["question"] or ""
    if state["retry_count"] > 2:
        print("⚠️ 재시도 횟수 초과. 웹 검색 실패로 간주합니다.")
        return {**state, "context": "웹 검색 결과가 없습니다. 기존 컨텍스트를 사용합니다."}
    
    # 💡 실제 웹 검색 로직 대신, 더미 데이터 생성
    #  - 사용자 질문 기반으로 적절한 웹 검색 결과를 시뮬레이션합니다.
    dummy_web_result = f"""[웹 검색 결과]
    - 질문 "{question}"에 대한 최신 정보입니다.
    - 웹 검색 결과는 다음과 같습니다:
    - 2025년 8월 12일자 뉴스 기사에 따르면, 전국적인 폭염 특보가 발령되었으며 특히 서울과 경기도 일부 지역에 한파가 발생하고 있습니다.
    - 농작물 재해 예방을 위한 정부 지침: "영농현장 재해예방 관리 지침"
    - 기상청 웹사이트에 따르면, 폭염주의보는 1주일째 지속되고 있습니다.
    - 추가 정보는 관련 문서를 참고하세요.
    """
    return {**state, "context": state["context"] + "\n\n" + dummy_web_result}

def generate_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 생성")
    if not state.get("question"): raise ValueError("question 누락")
    if not state.get("context"): raise ValueError("context 누락")

    prompt = ChatPromptTemplate.from_template(
        """너는 농작물 재해 대응 전문가야.
아래 문맥에는 JSON 특보와 일반 문서가 섞여 있어. JSON에서 사실만 선별하되, 최종 출력은 한국어 자연어 문장으로만 작성해.
불릿, 번호 목록, 코드블록, JSON, 표는 사용하지 마. 중복/상충은 정리하고,
특보는 지역·종류·수준·명령·발표시각·기간을 자연스러운 문장으로 설명해.

[문맥]
{context}

질문: {question}
답변:"""
    )
    chain = (
        {"context": itemgetter("context"), "question": itemgetter("question")}
        | prompt
        | make_llm()
        | StrOutputParser()
    )
    ans = chain.invoke({"context": state["context"], "question": state["question"]})
    txt = re.sub(r'\n{3,}', '\n\n', ans or "").strip()
    return {**state, "answer": txt}

def post_generate_validate_node(state: GraphState) -> Dict[str, Any]:
    print(f"🧩 노드: 2차 답변 검증 | 재시도 횟수: {state['retry_count']}")
    q = state["question"] or ""
    a = state["answer"] or ""
    ctx = state["context"] or ""

    # ✅ LLM을 사용해 답변이 질문에 충분한지, 컨텍스트를 기반으로 하는지 평가 (더미 로직)
    is_valid_answer = "알수없습니다" not in a and "부족합니다" not in a
    is_accurate_based_on_context = True # 실제로는 LLM 호출 필요

    if not is_valid_answer and state["retry_count"] < 2:
        if _should_use_live(q) and "LIVE_STATUS" not in ctx:
            print("⚠️ 답변이 불충분하고 실시간 정보가 누락되어 웹 검색 재시도")
            return {**state, "next_action": "re_search", "retry_count": state["retry_count"] + 1}
        else:
            print("⚠️ 답변이 불충분하여 검색 재시도")
            return {**state, "next_action": "re_retrieve", "retry_count": state["retry_count"] + 1}
    
    print("✅ 답변이 충분합니다. 종료합니다.")
    return {**state, "next_action": "END"}

def fallback_answer_node(state: GraphState) -> Dict[str, Any]:
    print("🧩 노드: 최종 실패 답변")
    question = state["question"] or ""
    fallback_message = (
        "죄송합니다. 현재 시스템은 여러 차례 재시도했음에도 질문에 대한 정확한 정보를 찾거나 생성하는 데 실패했습니다. "
        "질문을 더 구체적으로 바꿔주시거나 잠시 후에 다시 시도해주시기 바랍니다."
    )
    return {**state, "answer": fallback_message}

# =========[ 그래프 빌드 ]=========
def build_graph():
    g = StateGraph(GraphState)
    
    g.add_node("load_store", load_store_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("validate_context", validate_context_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate", generate_node)
    g.add_node("post_generate_validate", post_generate_validate_node)
    g.add_node("fallback_answer", fallback_answer_node)

    g.set_entry_point("load_store")
    g.add_edge("load_store", "retrieve")
    g.add_edge("retrieve", "validate_context")

    # 1차 검증 결과에 따라 흐름 제어
    g.add_conditional_edges(
        "validate_context",
        lambda state: "continue" if state["is_context_valid"] else "fallback",
        {"continue": "generate", "fallback": "web_search"}
    )

    # 웹 검색 후 다시 생성으로 이동
    g.add_edge("web_search", "generate")

    g.add_edge("generate", "post_generate_validate")

    # 2차 검증 결과에 따라 재시도 또는 종료
    g.add_conditional_edges(
        "post_generate_validate",
        lambda state: state["next_action"],  # "re_retrieve" | "re_search" | "END"
        {"re_retrieve": "retrieve", "re_search": "web_search", "END": END}
    )
    
    # 재시도 초과 등 최종 실패 시
    g.add_edge("web_search", "fallback_answer")
    g.add_edge("fallback_answer", END)

    # ✅ 여기서 컴파일하고, 컴파일된 객체(app)로 그래프 이미지 생성
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