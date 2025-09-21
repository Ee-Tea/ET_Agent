# teacher/agents/TestGenerator/RAGAS_runner_generator.py
# -*- coding: utf-8 -*-
import os, re, json, sys, traceback, importlib
from typing import Any, Dict, List, Tuple

# =========================================================
# 0) 프로젝트 루트(sys.path) 설정
# =========================================================
def _add_repo_root_to_syspath():
    cwd = os.path.abspath(os.getcwd())
    cand_dirs = [cwd, os.path.dirname(__file__) if "__file__" in globals() else cwd]
    for base in cand_dirs:
        cur = os.path.abspath(base)
        for _ in range(7):
            if os.path.isdir(os.path.join(cur, "teacher")):
                if cur not in sys.path:
                    sys.path.insert(0, cur)
                return cur
            nxt = os.path.dirname(cur)
            if nxt == cur: break
            cur = nxt
    return None
_repo_root = _add_repo_root_to_syspath()

# =========================================================
# 1) generator 에이전트 임포트 (패키지 경로로)
# =========================================================
try:
    gen_mod = importlib.import_module("teacher.agents.TestGenerator.generator")
except Exception as e:
    raise ImportError(
        "패키지 임포트 실패: 'teacher.agents.TestGenerator.generator'. "
        f"(repo_root 추정: {_repo_root})\n원인: {type(e).__name__}: {e}"
    )

AgentCls = (
    getattr(gen_mod, "InfoProcessingExamAgent", None)
    or getattr(gen_mod, "GeneratorAgent", None)
    or getattr(gen_mod, "ExamGeneratorAgent", None)
)
if AgentCls is None:
    raise ImportError("generator.py에서 InfoProcessingExamAgent/GeneratorAgent/ExamGeneratorAgent 클래스를 찾지 못했습니다.")

# =========================================================
# 2) Milvus 연결 정보 (레퍼런스 코드 반영)
# =========================================================
try:
    from common.milvus_helpers import get_milvus_connection_info
except Exception:
    # 레퍼런스에 맞춰 동일 시그니처의 폴백 생성
    def get_milvus_connection_info(host: str, port: str, embedding_model_name: str) -> Dict[str, Any]:
        return {
            "connection_status": False,
            "host": host, "port": port, "embedding_model_name": embedding_model_name
        }

# === 기존 _build_milvus_data 정의를 아래로 교체 ===
def _build_milvus_data(host: str, port: str, emb_model: str) -> Dict[str, Any]:
    """
    get_milvus_connection_info 호환:
    - 어떤 구현은 인자 없이 호출, 어떤 구현은 dict 한 개만 받음.
    - 반환값도 dict 혹은 (dict 또는 bool/tuple)일 수 있어 보정.
    """
    # 강제 OFF
    if os.getenv("GENERATOR_DISABLE_MILVUS", "0") == "1":
        print("[Milvus] 환경변수로 연결 비활성화(GENERATOR_DISABLE_MILVUS=1).")
        return {"connection_status": False, "host": host, "port": port, "embedding_model_name": emb_model}

    base = {"host": host, "port": port, "embedding_model_name": emb_model}
    try:
        try:
            info = get_milvus_connection_info(base)  # 1-인자 버전 우선
        except TypeError:
            info = get_milvus_connection_info()      # 무인자 버전 폴백
        # 반환 보정
        if isinstance(info, tuple):
            # (status, host, port, model) 같은 변형 대비
            status = bool(info[0]) if len(info) > 0 else False
            h = info[1] if len(info) > 1 else host
            p = info[2] if len(info) > 2 else port
            m = info[3] if len(info) > 3 else emb_model
            return {"connection_status": status, "host": h, "port": p, "embedding_model_name": m}
        if isinstance(info, dict):
            return {
                "connection_status": bool(info.get("connection_status", True)),
                "host": info.get("host", host),
                "port": info.get("port", port),
                "embedding_model_name": info.get("embedding_model_name", emb_model),
            }
        # 알 수 없는 형태면 안전하게 OFF
        print(f"[Milvus][WARN] 예상치 못한 반환형: {type(info).__name__} → 오프라인 모드.")
        return {"connection_status": False, **base}
    except Exception as e:
        print(f"[Milvus][WARN] 연결 정보 획득 실패 → 오프라인 모드: {type(e).__name__}: {e}")
        return {"connection_status": False, **base}

# =========================================================
# 3) RAGAS / 안전 전처리
# =========================================================
from datasets import Dataset

_ZWS = "\u200b\u200c\u200d\ufeff"
_END_PUNCT_RE = re.compile(r"(?:[\.!?…]+(?:\s*)|[\.!?…]+[”’'\")\]\}]+\s*|\s*[”’'\")\]\}]+\s*)$")

def _normalize_unicode_space(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = (s.replace("\xa0", " ").replace("\u00ad", "")
         .translate({ord(c): None for c in _ZWS}))
    s = s.replace("．",".").replace("。",".").replace("！","!").replace("？","?").replace("⋯","…")
    return s

def _min_sanitize(text: Any, hint: str = "") -> str:
    t = _normalize_unicode_space(str(text or "")).strip()
    if not t:
        t = _normalize_unicode_space(str(hint or "")).strip()
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

def _make_ds_safe(ds: Dataset) -> Dataset:
    data = {"question": [], "answer": [], "contexts": [], "ground_truths": [], "reference": []}
    n = len(ds)
    for i in range(n):
        q  = ds["question"][i] if "question" in ds.column_names else ""
        a  = ds["answer"][i] if "answer" in ds.column_names else ""
        r  = ds["reference"][i] if "reference" in ds.column_names else ""
        gts = ds["ground_truths"][i] if "ground_truths" in ds.column_names else [r]
        ctxs = ds["contexts"][i] if "contexts" in ds.column_names else []
        if not isinstance(gts, list):  gts  = [gts]
        if not isinstance(ctxs, list): ctxs = [ctxs]
        q_s  = _min_sanitize(q)
        a_s  = _min_sanitize(a, hint=q_s)
        gts_s = [_min_sanitize(x, hint=q_s) for x in gts] or [_min_sanitize(q_s)]
        ref_s = _min_sanitize(r if r else (gts_s[0] if gts_s else q_s), hint=q_s)
        ctxs_s = [_min_sanitize(x, hint=q_s) for x in ctxs] or [_min_sanitize(q_s)]
        data["question"].append(q_s)
        data["answer"].append(a_s)
        data["ground_truths"].append(gts_s)
        data["reference"].append(ref_s)
        data["contexts"].append(ctxs_s)
    return Dataset.from_dict(data)

def ragas_evaluate_with_row_debug(ds: Dataset, out_dir_file: str):
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from types import SimpleNamespace
    def _default_score_dict():
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None, "context_recall": None}
    ds_safe = _make_ds_safe(ds)
    try:
        return evaluate(ds_safe, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    except Exception as e:
        print(f"\n[RAGAS][ERROR] batch evaluate() failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        bad_rows, good_rows = [], []
        for i in range(len(ds_safe)):
            one = {k: [ds_safe[k][i]] for k in ds_safe.column_names}
            ds1 = Dataset.from_dict(one)
            try:
                _ = evaluate(ds1, metrics=[answer_relevancy])
                good_rows.append(i)
            except Exception as e2:
                print(f"[RAGAS][BAD-ROW] idx={i} → {type(e2).__name__}: {e2}")
                bad_rows.append(i)
                try:
                    os.makedirs(out_dir_file, exist_ok=True)
                    with open(os.path.join(out_dir_file, "ragas_bad_rows.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({"idx": i, **{k: ds_safe[k][i] for k in ds_safe.column_names}}, ensure_ascii=False) + "\n")
                except Exception as dump_err:
                    print(f"  (dump failed: {dump_err})")
        if not good_rows:
            return SimpleNamespace(scores=[_default_score_dict() for _ in range(len(ds_safe))])
        ds_ok = Dataset.from_dict({k: [ds_safe[k][i] for i in good_rows] for k in ds_safe.column_names})
        try:
            res_ok = evaluate(ds_ok, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
            ok_scores = list(res_ok.scores or [])
        except Exception as e3:
            print(f"[RAGAS] even on good rows failed → None: {e3}")
            return SimpleNamespace(scores=[_default_score_dict() for _ in range(len(ds_safe))])
        full_scores = []
        ok_iter = iter(ok_scores)
        bad_set = set(bad_rows)
        for i in range(len(ds_safe)):
            full_scores.append(_default_score_dict() if i in bad_set else (next(ok_iter, {}) or _default_score_dict()))
        return SimpleNamespace(scores=full_scores)

# =========================================================
# 4) 유틸 & 리트리버 패치
# =========================================================
SUBJECTS = ["소프트웨어설계","소프트웨어개발","데이터베이스구축","프로그래밍언어활용","정보시스템구축관리"]
_KOR_NUM = {"한":1,"두":2,"세":3,"네":4,"다섯":5,"여섯":6,"일곱":7,"여덟":8,"아홉":9,"열":10}

def _norm(s: Any) -> str:
    if s is None: return ""
    t = str(s)
    for z in ["\u200b","\u200c","\u200d","\ufeff"]:
        t = t.replace(z,"")
    t = re.sub(r"\s+"," ",t).strip()
    return t

def _parse_subject_k(prompt: str) -> Tuple[str, int]:
    p = _norm(prompt)
    m = re.search(r"([0-9]+)\s*문제", p)
    k = int(m.group(1)) if m else 0
    if k <= 0:
        m2 = re.search(r"(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*문제", p)
        if m2: k = _KOR_NUM.get(m2.group(1), 0)
    if k <= 0: k = 5
    subj = "전체 범위"
    for s in SUBJECTS:
        if s.replace(" ","") in p.replace(" ",""):
            subj = s
            break
    return subj, k

def _serialize_questions(questions: List[Dict[str, Any]]) -> str:
    lines=[]
    for i,q in enumerate(questions or [], start=1):
        qt = _norm(q.get("question",""))
        opts = [ _norm(o) for o in (q.get("options") or []) ][:4]
        lines.append(f"[문제 {i}] {qt}")
        lines.append("[보기]")
        for j,o in enumerate(opts, start=1):
            o = re.sub(r"^\s*\d+[\.\)]\s*", "", o)
            lines.append(f"{j}) {o}")
        lines.append("")
    txt = "\n".join(lines).strip()
    return txt if txt else "."

def _serialize_gt(gt_list: List[Dict[str, Any]]) -> str:
    return _serialize_questions(gt_list or [])

def _clip_contexts(ctxs: List[str], max_items=3, max_chars=6000) -> List[str]:
    out=[]
    for c in (ctxs or [])[:max_items]:
        c = str(c or "")
        out.append(c[:max_chars])
    return out or [""]

class _Doc:
    def __init__(self, text: str):
        self.page_content = text
        self.metadata = {"source": "golden_context"}

# === 기존 _patch_agent_retriever_with_contexts 를 아래로 교체 ===
def _make_doc_dict(text: str) -> Dict[str, Any]:
    return {
        "page_content": text,
        "content": text,
        "text": text,
        "metadata": {"source": "golden_context", "score": 1.0},
    }

def _patch_object_retrievers(obj: Any, docs: List[Dict[str, Any]]):
    """obj의 retrieve/search 관련 메서드를 모두 dict 리스트 반환으로 패치"""
    import types, inspect
    def _mk_fn():
        def _fake_retrieve(*args, **kwargs):
            top_k = kwargs.get("top_k") or kwargs.get("k")
            return docs[:top_k] if isinstance(top_k, int) and top_k > 0 else docs
        return _fake_retrieve

    # 메서드 후보명: milvus/retireve/search/fetch/lookup 등 폭넓게 커버
    name_patterns = ("retrieve", "retreive", "retreval", "milvus", "search", "fetch", "lookup")
    for name in dir(obj):
        if not any(p in name.lower() for p in name_patterns):
            continue
        try:
            attr = getattr(obj, name)
            if callable(attr):
                setattr(obj, name, types.MethodType(_mk_fn(), obj))
        except Exception:
            pass

    # 중첩 필드도 패치 (retriever, searcher, milvus, vectorstore 등)
    nested_fields = ("retriever", "searcher", "milvus", "milvus_client", "vectorstore", "store", "client")
    for nf in nested_fields:
        if hasattr(obj, nf):
            try:
                _patch_object_retrievers(getattr(obj, nf), docs)
            except Exception:
                pass

def _patch_agent_retriever_with_contexts(agent: Any, ctx_texts: List[str]) -> None:
    """제너레이터가 어디서 retrieval을 호출하든 dict 리스트를 돌려주도록 광역 패치"""
    docs = [_make_doc_dict(t) for t in ctx_texts]
    _patch_object_retrievers(agent, docs)

# =========================================================
# 5) 한 샘플 평가
# =========================================================
def eval_one(item: Dict[str, Any], agent, milvus_data: Dict[str, Any], out_dir_file: str) -> Dict[str, Any]:
    prompt = _norm(item.get("question",""))
    contexts_raw = [c for c in (item.get("contexts") or []) if str(c).strip()]
    contexts = _clip_contexts(contexts_raw)
    gt_list = item.get("ground_truth") or []
    subject, k = _parse_subject_k(prompt)

    # 연결 플래그 결정 (레퍼런스 반영 + 환경변수)
    use_ctx_patch = (os.getenv("FORCE_GOLDEN_CTX", "1") == "1")
    try_milvus = bool(milvus_data.get("connection_status"))
    if os.getenv("GENERATOR_DISABLE_MILVUS", "0") == "1":
        try_milvus = False

    if contexts and (use_ctx_patch or not try_milvus):
        _patch_agent_retriever_with_contexts(agent, contexts)


    # --- generator 호출 ---
    questions = []
    def _invoke(payload: Dict[str, Any]):
        res = agent.invoke(payload)  # type: ignore
        result = (res or {}).get("result") or {}
        qs = (
            result.get("result", {}).get("subjects", {}).get(subject, {}).get("questions") or
            result.get("result", {}).get("all_questions") or
            result.get("questions") or
            result.get("all_questions") or
            []
        )
        return qs

    def _payload(base: Dict[str, Any]) -> Dict[str, Any]:
        base = dict(base)
        base["milvus_data"] = dict(milvus_data, connection_status=try_milvus)
        base["save_to_file"] = False
        base.setdefault("difficulty", "중급")
        # 🔹 외부 컨텍스트 힌트: generator 구현에 따라 인식 가능한 키들 다 넣어줌(무해)
        base.setdefault("external_contexts", contexts)       # 흔한 키
        base.setdefault("provided_contexts", contexts)       # 대안 키
        base.setdefault("context_overrides", contexts)       # 또다른 대안
        return base

    if subject in SUBJECTS:
        base = {"mode": "subject_quiz", "subject_area": subject, "target_count": k}
    else:
        per = max(1, k // 5) or 1
        base = {"mode": "partial_exam", "selected_subjects": SUBJECTS, "questions_per_subject": per}

    try:
        questions = _invoke(_payload(base))
    except Exception as e_first:
        # 첫 시도 실패 시 Milvus 끄고 재시도 (레퍼런스식 폴백)
        if try_milvus:
            print(f"[generator][WARN] invoke 실패(연결 on) → 연결 off로 재시도: {type(e_first).__name__}: {e_first}")
            try_milvus = False
            questions = _invoke(_payload(base))
        else:
            raise

    if len(questions) > k:
        questions = questions[:k]

    generated = _serialize_questions(questions)
    golden = _serialize_gt(gt_list)

    ds = Dataset.from_dict({
        "question":      [_min_sanitize(prompt)],
        "contexts":      [[_min_sanitize(c) for c in contexts]],
        "answer":        [_min_sanitize(generated)],
        "ground_truths": [[_min_sanitize(golden)]],
        "reference":     [_min_sanitize(golden)],
    })

    ragas_res = ragas_evaluate_with_row_debug(ds, out_dir_file)
    sc = (ragas_res.scores or [{}])[0] if hasattr(ragas_res, "scores") else {}

    return {
        "prompt": prompt, "subject": subject, "k": k,
        "n_generated": len(questions or []),
        "scores": sc,
        "preview": generated[:300]
    }

# =========================================================
# 6) 메인 러너
# =========================================================
def _load_golden(path: str) -> List[Dict[str, Any]]:
    txt = open(path, "r", encoding="utf-8").read().strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, list): return obj
        if isinstance(obj, dict): return [obj]
    except Exception:
        pass
    items=[]
    for line in txt.splitlines():
        line=line.strip()
        if not line: continue
        try:
            items.append(json.loads(line))
        except Exception:
            pass
    if not items:
        raise ValueError("지원하지 않는 골든셋 포맷")
    return items

def run(golden_path: str,
        out_dir: str = "./teacher/agents/TestGenerator/eval_results",
        milvus_host="localhost", milvus_port="19530", emb_model="jhgan/ko-sroberta-multitask",
        limit: int | None = None):
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    items = _load_golden(golden_path)
    if isinstance(limit, int) and limit > 0:
        items = items[:limit]

    # 레퍼런스 방식의 연결 정보 구성
    milvus_data = _build_milvus_data(milvus_host, milvus_port, emb_model)

    agent = AgentCls()   # 모델 초기화가 외부 API를 쓰면, 별도 skip 플래그를 추가해도 좋음

    base = os.path.splitext(os.path.basename(golden_path))[0]
    out_dir_file = os.path.join(out_dir, base)
    os.makedirs(out_dir_file, exist_ok=True)

    rows=[]
    for i, it in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] eval…")
        try:
            rows.append(eval_one(it, agent, milvus_data, out_dir_file))
        except Exception as e:
            print(f"[ERROR] row {i}: {type(e).__name__}: {e}")
            traceback.print_exc()
            rows.append({
                "prompt": _norm(it.get("question","")),
                "subject": None, "k": None, "n_generated": 0,
                "scores": {"faithfulness": None, "answer_relevancy": None, "context_precision": None, "context_recall": None},
                "preview": ""
            })

    df = pd.DataFrame([{
        "prompt": r["prompt"], "subject": r["subject"], "k": r["k"], "n_generated": r["n_generated"],
        "faithfulness": r["scores"].get("faithfulness"),
        "answer_relevancy": r["scores"].get("answer_relevancy"),
        "context_precision": r["scores"].get("context_precision"),
        "context_recall": r["scores"].get("context_recall"),
        "generated_preview": r["preview"],
    } for r in rows])
    ragas_csv = os.path.join(out_dir_file, "ragas_scores.csv")
    df.to_csv(ragas_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 저장 완료: {ragas_csv}")

def main():
    target = os.getenv("GOLDENSET_PATH", "teacher/agents/TestGenerator/goldensets/generator_golden_20250917_152907.jsonl")
    outdir = os.getenv("OUT_DIR", "./teacher/agents/TestGenerator/eval_results")
    run(target, outdir,
        milvus_host=os.getenv("MILVUS_HOST","localhost"),
        milvus_port=os.getenv("MILVUS_PORT","19530"),
        emb_model=os.getenv("EMB_MODEL","jhgan/ko-sroberta-multitask"),
        limit=int(os.getenv("LIMIT","0")) or None)

if __name__ == "__main__":
    main()