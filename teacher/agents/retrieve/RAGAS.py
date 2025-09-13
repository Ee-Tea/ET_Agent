# -*- coding: utf-8 -*-
"""
RAGAS 평가 스크립트 (retrieve_agent 전용)
- 입력: 골든셋 JSONL (각 줄: {"question","ground_truth","contexts"})
- 절차: 골든셋 로드 → retrieve_agent로 예측(answer) & 컨텍스트 추출 → RAGAS 평가 → 결과 저장
- 참고: retrieve_agent.invoke(input) -> {"retrieve_answer": str, "retrieval": {"merged_context": str, ...}}
"""

import os, json, argparse, time, re
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

# ================== (1) 에이전트 로드 ==================
# 프로젝트 경로에 맞게 import 우선 시도
try:
    from teacher.agents.retrieve.retrieve_agent import retrieve_agent as RetrieveAgent
except Exception:
    # 같은 디렉토리에 파일이 있는 경우
    from retrieve_agent import retrieve_agent as RetrieveAgent

# ================== (2) RAGAS 준비물 ==================
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


# ---------- 유틸: JSONL 로드 ----------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            obj = json.loads(line)
            q  = obj.get("question") or obj.get("user_input") or ""
            gt = obj.get("ground_truth") or obj.get("reference") or ""
            # gold contexts는 평가지표에 직접 쓰지 않고 참고/디버그 용도로만 저장
            ctx = obj.get("contexts") or obj.get("reference_contexts") or []
            if isinstance(ctx, str):
                ctx = [ctx]
            rows.append({"question": q, "ground_truth": gt, "gold_contexts": ctx})
    return rows


# ---------- 유틸: agent의 merged_context 문자열 → 리스트 ----------
def split_contexts_from_merged(merged: str) -> List[str]:
    """
    retrieve_agent는 병합 컨텍스트를 큰 문자열로 반환하므로
    - 두 줄 공백 기준 블록 분리
    - "[문서 i]" 마커가 있으면 추가 분할
    - 너무 짧은 블록 제거
    - 상한 10개로 잘라 RAGAS 비용 제어
    """
    if not merged:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", merged) if p.strip()]
    refined: List[str] = []
    for p in parts:
        if "[문서" in p:
            refined += [c.strip() for c in p.split("[문서") if c.strip()]
        else:
            refined.append(p)
    refined = [c for c in refined if len(c) > 20]
    return refined[:10] or [""]  # 최소 1개 보장


# ---------- 유틸: Milvus 연결 정보 구성 (환경변수 기반) ----------
def build_milvus_data_from_env() -> Dict[str, Any]:
    """
    다른 에이전트 RAGAS 코드 스타일을 따라 환경변수에서 연결 정보 수집.
    (retrieve_agent 안의 Milvus 검색 노드는 milvus_data를 그대로 넘겨 사용함)
    """
    d = {
        "connection_status": True,
        "host": os.getenv("MILVUS_HOST", "localhost"),
        "port": os.getenv("MILVUS_PORT", "19530"),
        "embedding_model_name": os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large"),
        # 필요시 token/username/password 등 추가
    }
    return d


# ---------- 유틸: RAGAS 직전 최소 안전화 ----------
_END_PUNCT_RE = re.compile(r"(?:[\.!?…]+(?:\s*)|[\.!?…]+[”’'\")\]\}]+\s*|\s*[”’'\")\]\}]+\s*)$")
def _normalize_unicode_space(s: str) -> str:
    s = str(s or "")
    for z in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        s = s.replace(z, "")
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _min_sanitize(s: Any, hint: str = "") -> str:
    t = _normalize_unicode_space(str(s or "")).strip() or _normalize_unicode_space(str(hint or "")).strip()
    if not t:
        t = "."
    if not _END_PUNCT_RE.search(t):
        m = re.search(r"[”’'\")\]\}]+$", t)
        if m:
            closers = m.group(0)
            core = t[:m.start()].rstrip()
            t = (core + "." + closers) if core and core[-1] not in ".!?…" else (core + closers)
        else:
            t = t.rstrip() + "."
    return t


# ---------- 메인 평가 파이프라인 ----------
def run_evaluation(
    goldenset_path: str,
    out_dir: str,
    limit: Optional[int] = None,
    sleep_sec: float = 0.0,
    openai_model: str = "gpt-4o-mini",
    embedding_model: str = "intfloat/multilingual-e5-large",
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"ragas_eval_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    rows = load_jsonl(goldenset_path)
    if limit:
        rows = rows[:limit]

    # 1) 에이전트 인스턴스
    agent = RetrieveAgent()  # invoke({"retrieval_question": "...", "milvus_data": {...}}) 사용.  # :contentReference[oaicite:2]{index=2}

    # 2) 예측 생성
    predictions: List[Dict[str, Any]] = []
    for r in tqdm(rows, ncols=100, desc="Answering with retrieve_agent"):
        q = r["question"]
        gt = r["ground_truth"]
        milvus_data = build_milvus_data_from_env()  # 다른 RAGAS 코드 스타일 참조  # :contentReference[oaicite:3]{index=3}

        try:
            res = agent.invoke({"retrieval_question": q, "milvus_data": milvus_data})
            ans = res.get("retrieve_answer", "")
            merged = (res.get("retrieval") or {}).get("merged_context", "")
            pred_ctx = split_contexts_from_merged(merged)
        except Exception as e:
            ans = ""
            pred_ctx = [""]
            merged = f"(error: {e})"

        predictions.append({
            "question": q,
            "answer": ans,
            "contexts": pred_ctx,
            "ground_truth": gt,
            "_agent_ctx_merged": merged,      # 디버그
            "_gold_contexts": r["gold_contexts"],
        })

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    # 3) RAGAS 입력 Dataset 구성 (최소 안전화)
    data = {
        "question":      [_min_sanitize(p["question"]) for p in predictions],
        "answer":        [_min_sanitize(p["answer"], hint=p["question"]) for p in predictions],
        "contexts":      [[_min_sanitize(c, hint=p["question"]) for c in (p["contexts"] or [""])] for p in predictions],
        "ground_truths": [[_min_sanitize(p["ground_truth"], hint=p["question"])] for p in predictions],
        "reference":     [_min_sanitize(p["ground_truth"], hint=p["question"]) for p in predictions],
    }
    ds = Dataset.from_dict(data)

    # 4) RAGAS 평가기
    eval_llm = LangchainLLMWrapper(ChatOpenAI(model=openai_model, temperature=0.0, max_tokens=2048))
    eval_emb = HuggingFaceEmbeddings(model=embedding_model)

    metrics = [
        faithfulness,        # 답변-컨텍스트 충실성
        answer_relevancy,    # 질문-답변 관련성
        context_precision,   # 제공 컨텍스트의 정밀도
        context_recall,      # 필요한 정보가 컨텍스트에 포함되었는가
        answer_correctness,  # 정답/레퍼런스와의 일치도
    ]

    res = evaluate(ds, metrics=metrics, llm=eval_llm, embeddings=eval_emb, raise_exceptions=False)

    # 5) 저장
    # 전체 점수 요약
    with open(os.path.join(run_dir, "overall_scores.json"), "w", encoding="utf-8") as f:
        json.dump(res.scores, f, ensure_ascii=False, indent=2)

    # 샘플별 점수 + 예측/디버그
    import pandas as pd
    per_sample = res.dataset.to_pandas()
    per_sample.to_csv(os.path.join(run_dir, "per_sample_scores.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(run_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n✅ RAGAS 평가 완료")
    print(f" - 전체 점수: {os.path.join(run_dir, 'overall_scores.json')}")
    print(f" - 샘플별 점수: {os.path.join(run_dir, 'per_sample_scores.csv')}")
    print(f" - 예측/디버그: {os.path.join(run_dir, 'predictions.jsonl')}")


# ---------- CLI ----------
def main():
    input_path = "teacher/agents/retrieve/out/goldenset_20250913_141611.jsonl"
    # input_path = "teacher/agents/retrieve/out/goldenset_20250913_154620.jsonl"
    out_dir = "./ragas_out"
    limit = None
    sleep_sec = 0.0
    openai_model = "gpt-4o-mini"
    embedding_model = "intfloat/multilingual-e5-large"

    run_evaluation(
        goldenset_path=input_path,
        out_dir=out_dir,
        limit=limit,
        sleep_sec=sleep_sec,
        openai_model=openai_model,
        embedding_model=embedding_model,
    )

if __name__ == "__main__":
    main()
