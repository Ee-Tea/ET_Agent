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
from langchain_huggingface import HuggingFaceEmbeddings
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
def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"

MAX_CHARS_CTX = 1200
MAX_CTX_BLOCKS = 5
MAX_CHARS_ANS = 1500

def split_contexts_from_merged(merged: str) -> List[str]:
    if not merged:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", merged) if p.strip()]
    refined = []
    for p in parts:
        if "[문서" in p:
            refined += [c.strip() for c in p.split("[문서") if c.strip()]
        else:
            refined.append(p)
    refined = [c for c in refined if len(c) > 20]
    refined = [_clip(c, MAX_CHARS_CTX) for c in refined]
    return refined[:MAX_CTX_BLOCKS] or [""]






# ---------- 유틸: Milvus 연결 정보 구성 (환경변수 기반) ----------
def build_milvus_data_from_env() -> Dict[str, Any]:
    """
    MilvusDBManager(개별 인서터)로 생성된 'concepts' 컬렉션과 정확히 일치하도록
    검색 노드가 사용할 설정을 구성한다.
    retrieve_agent는 이 dict를 그대로 받아 Milvus에 접속/검색한다는 가정.
    """
    host   = os.getenv("MILVUS_HOST", "localhost")
    port   = os.getenv("MILVUS_PORT", "19530")

    # 인서터에서 기본값: collection_name='concepts', vector_field='embedding'
    collection = os.getenv("CONCEPT_COLL", "concepts")
    vector_field = os.getenv("MILVUS_VECTOR_FIELD", "embedding")

    # 인서터 기본 인덱스: HNSW + COSINE 권장
    index_type = os.getenv("MILVUS_INDEX", "HNSW").upper()          # HNSW | IVF_FLAT | ...
    metric     = os.getenv("MILVUS_METRIC", "COSINE").upper()       # COSINE | IP | L2
    ef         = int(os.getenv("MILVUS_EF", "128"))                 # HNSW용 파라미터
    nprobe     = int(os.getenv("NPROBE", "32"))                     # IVF용 파라미터
    top_k      = int(os.getenv("TOP_K", "5"))

    # 인서터에서 사용한 SentenceTransformer 모델과 반드시 동일해야 차원 mismatch가 안 남
    # (MilvusDBManager 기본: jhgan/ko-sroberta-multitask → 보통 768-d)
    embedding_model_name = os.getenv("EMBED_MODEL", "jhgan/ko-sroberta-multitask")
    expected_dim_env = os.getenv("EMBED_DIM")  # 선택: 차원 고정 검증
    expected_dim = int(expected_dim_env) if expected_dim_env and expected_dim_env.isdigit() else None

    # 검색 파라미터 (Milvus SDK search params)
    if index_type == "HNSW":
        search_params = {"metric_type": metric, "params": {"ef": ef}}
    else:
        search_params = {"metric_type": metric, "params": {"nprobe": nprobe}}

    # retrieve_agent가 사용할 출력 필드들 (인서터 스키마와 일치)
    output_fields = [
        "id", "subject", "source_file", "item_id",
        "item_title", "content", "chunk_index", "n_tokens",
    ]

    # (선택) subject 필터를 외부에서 주입하고 싶을 때를 대비한 자리
    subject_filter = os.getenv("SUBJECT_FILTER")  # e.g., "데이터베이스 구축"

    return {
        "connection_status": True,
        "host": host,
        "port": port,
        "collection_name": collection,
        "vector_field": vector_field,
        "output_fields": output_fields,
        "metric_type": metric,
        "index_type": index_type,
        "search_params": search_params,
        "top_k": top_k,
        "subject_filter": subject_filter,          # None이면 미사용
        "embedding_model_name": embedding_model_name,
        "expected_dim": expected_dim,              # None이면 검증 생략
        "normalize_embeddings": True,              # 인서터와 동일 설정 권장
        "consistency_level": os.getenv("MILVUS_CONSISTENCY", "Eventually"),
        # 필요 시 인증/네임스페이스 등 확장 키 추가 가능
    }


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
        "answer":        [_clip(_min_sanitize(p["answer"], hint=p["question"]), MAX_CHARS_ANS)
                        for p in predictions],
        "contexts":      [[_min_sanitize(c, hint=p["question"]) for c in (p["contexts"] or [""])]
                        for p in predictions],
        # ground_truths: 리스트 형태 유지
        "ground_truths": [[_min_sanitize(p["ground_truth"], hint=p["question"])]
                        for p in predictions],
        # reference: 단일 문자열 (대개 ground_truths[0]을 그대로 사용)
        "reference":     [_min_sanitize(p["ground_truth"], hint=p["question"])
                        for p in predictions],
    }
    ds = Dataset.from_dict(data)


    # 4) RAGAS 평가기
    eval_llm = LangchainLLMWrapper(ChatOpenAI(model=openai_model, temperature=0.0, max_tokens=2048))
    eval_emb = HuggingFaceEmbeddings(model_name=embedding_model) 

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
    input_path = "teacher/agents/retrieve/goldensets/goldenset_20250913_154620.jsonl"
    out_dir = "teacher/agents/retrieve/eval_results"
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
