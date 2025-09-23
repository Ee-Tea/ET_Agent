# teacher/agents/TestGenerator/RAGAS_runner_generator.py
# -*- coding: utf-8 -*-
import os, re, json, sys, traceback, importlib, math
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
# 1) 유틸(문장 인식/보정, 컨텍스트/결과 파서)
# =========================================================
_ZWS = "\u200b\u200c\u200d\ufeff"
_END_PUNCT_RE = re.compile(r"(?:[\.!?…]+(?:\s*)|[\.!?…]+[”’'\")\]\}]+\s*|\s*[”’'\")\]\}]+\s*)$")
_ENUM_ONLY_LINE = re.compile(r"^\s*(?:\(?\d{1,3}\)?[.)]?)\s*$")
_BULLET_ONLY_LINE = re.compile(r"^\s*[\-\*\•]\s*$")
_ENUM_PREFIX_LINE = re.compile(r"^\s*(?:\(?\d{1,3}\)?[.)]\s+)")
_FENCE_BLOCK = re.compile(r"```.+?```", re.DOTALL)              # ★ PATCH: 코드블록 제거
_TAG_BLOCK = re.compile(r"<[^>]+>")                             # ★ PATCH: HTML 태그 제거
_RULE_LINE = re.compile(r"^\s*([=\-\*_~]{3,})\s*$")             # ★ PATCH: 구분선 제거

def _normalize_unicode_space(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = (s.replace("\xa0"," ").replace("\u00ad","")
         .translate({ord(c): None for c in _ZWS}))
    s = (s.replace("．",".").replace("。",".").replace("！","!").replace("？","?").replace("⋯","…"))
    # ★ PATCH: 전각 괄호/따옴표 정규화
    s = s.replace("（","(").replace("）",")").replace("［","[").replace("］","]").replace("｛","{").replace("｝","}")
    s = s.replace("“","\"").replace("”","\"").replace("‘","'").replace("’","'")
    return s

def _strip_enumeration_only_lines(text: str) -> Tuple[str, int]:
    if not isinstance(text, str): return "", 0
    removed = 0; kept=[]
    for ln in text.splitlines():
        if _ENUM_ONLY_LINE.match(ln) or _BULLET_ONLY_LINE.match(ln) or _RULE_LINE.match(ln):
            removed += 1
            continue
        kept.append(ln)
    return "\n".join(kept), removed

def _strip_enumeration_prefix_lines(text: str) -> Tuple[str, int]:
    if not isinstance(text, str): return "", 0
    removed = 0; new=[]
    for ln in text.splitlines():
        if _ENUM_PREFIX_LINE.match(ln):
            ln2 = _ENUM_PREFIX_LINE.sub("", ln, count=1)
            if ln2 != ln:
                removed += 1
                ln = ln2
        new.append(ln)
    return "\n".join(new), removed

# ★ PATCH: 기호 과다/의미 없음 라인 제거
def _is_symbol_heavy_short(ln: str) -> bool:
    t = re.sub(r"\s+", "", ln)
    if not t: return True
    if len(t) > 40: return False
    letters = re.findall(r"[A-Za-z0-9가-힣]", t)
    ratio = len(letters) / max(len(t),1)
    return ratio < 0.4  # 글자 비율 40% 미만이면 잡음

# ★ PATCH: 코드블록/태그 제거 + 잡음 라인 필터
def _strip_noise_blocks(text: str) -> str:
    if not isinstance(text, str): return ""
    t = _FENCE_BLOCK.sub(" ", text)
    t = _TAG_BLOCK.sub(" ", t)
    # markdown 헤더 기호 정리
    t = re.sub(r"^\s{0,3}#+\s*", "", t, flags=re.MULTILINE)
    # 잡음 라인 제거
    kept=[]
    for ln in t.splitlines():
        raw = ln.strip()
        if not raw:
            kept.append("")
            continue
        if _is_symbol_heavy_short(raw):
            continue
        kept.append(raw)
    return "\n".join(kept)

def _sanit_flags(name: str, s: str):
    tail = (s or "")[-6:]
    tail_codes = " ".join([f"U+{ord(c):04X}" for c in tail])
    flags=[]
    if not (s or "").strip():
        flags.append("empty")
    elif not _END_PUNCT_RE.search(s or ""):
        flags.append("no_end_punct")
    print(f"[RAGAS][{name}] flags={flags} len={len(s or '')} tail={repr(tail)} tail_codes=[{tail_codes}]")

def sanitize_for_ragas(text: str, role: str = "answer", fallback_hint: str = "", max_len: int = 2000) -> str:
    raw = (text or "")
    t = _normalize_unicode_space(raw)
    t = _strip_noise_blocks(t)                                   # ★ PATCH
    t = t.strip()
    t, _ = _strip_enumeration_only_lines(t)
    t, _ = _strip_enumeration_prefix_lines(t)

    if not t.strip():
        t = _normalize_unicode_space((fallback_hint or "").strip())
    if not t: t = "."

    if not _END_PUNCT_RE.search(t):
        m = re.search(r"[”’'\")\]\}]+$", t)
        if m:
            closers = m.group(0); core = t[:m.start()].rstrip()
            t = (core + "." + closers) if core and core[-1] not in ".!?…" else (core + closers)
        else:
            t = t.rstrip() + "."

    # ★ PATCH: 연속 공백/개행 정리 + 길이 제한
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"\s{2,}", " ", t)
    if len(t) > max_len:
        t = t[:max_len].rstrip() + "…"
    _sanit_flags(f"{role}(sanitize)", t)
    return t

def _debug_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t: return []
    t = re.sub(r"\s+"," ", t)
    parts = re.split(r"(?<=[\.!?])\s+|\n+", t)
    return [p.strip() for p in parts if p and p.strip()]

# =========================================================
# 1-2) 컨텍스트 추출/클립
# =========================================================
def _clip_contexts(ctxs: List[str], max_items=3, max_chars=1000) -> List[str]:
    # ★ PATCH: 컨텍스트 아이템 길이 하드캡 1000자
    out=[]
    for c in (ctxs or []):
        s = str(c or "").strip()
        if not s: continue
        s = sanitize_for_ragas(s, role="context", fallback_hint="", max_len=max_chars)
        out.append(s)
        if len(out) >= max_items: break
    return out

def _extract_milvus_contexts_from_res(res: Dict[str, Any], subject: str | None = None) -> List[str]:
    out: List[str] = []
    def _push_doclike(item):
        if item is None: return
        txt = getattr(item, "page_content", None)
        if not txt and isinstance(item, dict):
            txt = item.get("page_content") or item.get("content") or item.get("text")
        if isinstance(txt, str) and txt.strip(): out.append(txt.strip())
    def _push_text(val):
        if isinstance(val, str):
            t = val.strip()
            if t: out.append(t)
    res = res or {}
    result = res.get("result") or {}
    roots = [res, result, res.get("state") or {}]
    if subject:
        subjects_map = (result.get("result") or {}).get("subjects") or result.get("subjects") or {}
        subj_node = subjects_map.get(subject) or {}
        if isinstance(subj_node, dict): roots.append(subj_node)
    for root in roots:
        if not isinstance(root, dict): continue
        for key in ("documents", "retrieved_docs", "retrieved_documents"):
            items = root.get(key)
            if isinstance(items, list):
                for it in items: _push_doclike(it)
        items_text = root.get("documents_text")
        if isinstance(items_text, list):
            for t in items_text: _push_text(t)
        for key in ("context", "problems_contexts_text", "concept_contexts_text"):
            val = root.get(key); _push_text(val)
    # ★ PATCH: 중복 제거 + 클리핑 + 잡음 필터
    uniq = []
    for s in dict.fromkeys(out):
        s2 = sanitize_for_ragas(s, role="context", fallback_hint="")
        if s2 and s2 != ".": uniq.append(s2)
    clipped = _clip_contexts(uniq, max_items=3, max_chars=1000)
    if clipped: print("[DEBUG][_extract_ctx] first_ctx_head=", repr(clipped[0][:120]))
    else:       print("[DEBUG][_extract_ctx] no ctx extracted")
    return clipped



# =========================================================
# 2) generator 에이전트 임포트
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
# 3) Milvus 연결 정보 (간결 버전)
# =========================================================
try:
    from common.milvus_helpers import get_milvus_connection_info  # noqa
except Exception:
    def get_milvus_connection_info(host: str, port: str, embedding_model_name: str) -> Dict[str, Any]:
        return {"connection_status": False, "host": host, "port": port, "embedding_model_name": embedding_model_name}

def _build_milvus_data(host: str, port: str, emb_model: str) -> Dict[str, Any]:
    env_host = os.getenv("MILVUS_HOST", host or "localhost")
    env_port = os.getenv("MILVUS_PORT", port or "19530")
    env_model = os.getenv("EMB_MODEL", emb_model or "jhgan/ko-sroberta-multitask")
    return {"connection_status": True, "host": env_host, "port": env_port, "embedding_model_name": env_model}

# =========================================================
# 4) RAGAS / 최소 안전 전처리 + 평가
# =========================================================
from datasets import Dataset

def _min_sanitize(text: Any, hint: str = "", max_len: int = 2000) -> str:
    t = _normalize_unicode_space(str(text or "")).strip()
    t = _strip_noise_blocks(t)                                   # ★ PATCH
    if not t: t = _normalize_unicode_space(str(hint or "")).strip()
    if not t: t = "."
    if not _END_PUNCT_RE.search(t):
        m = re.search(r"[”’'\")\]\}]+$", t)
        if m:
            closers = m.group(0); core = t[:m.start()].rstrip()
            t = (core + "." + closers) if core and core[-1] not in ".!?…" else (core + closers)
        else:
            t = t.rstrip() + "."
    if len(t) > max_len: t = t[:max_len].rstrip() + "…"
    return t

# ★ PATCH: NaN/비문자/리스트/딕셔너리 방어
def _to_str(x: Any) -> str:
    try:
        if x is None: return ""
        if isinstance(x, (list, tuple, set)): return " ".join(_to_str(e) for e in x)
        if isinstance(x, dict): return json.dumps(x, ensure_ascii=False)
        s = str(x)
        if s.lower() == "nan": return ""
        return s
    except Exception:
        return ""

def _coerce_list_of_str(x: Any) -> List[str]:
    out: List[str] = []
    if isinstance(x, list):
        for e in x:
            s = _min_sanitize(_to_str(e))
            if s and s != ".": out.append(s)
    else:
        s = _min_sanitize(_to_str(x))
        if s and s != ".": out.append(s)
    if not out:
        out = ["."]
    return out

def _make_ds_safe(ds: Dataset) -> Dataset:
    cols = {"question": [], "answer": [], "contexts": [], "ground_truths": [], "reference": []}
    n = len(ds)
    for i in range(n):
        q  = _to_str(ds["question"][i]) if "question" in ds.column_names else ""
        a  = _to_str(ds["answer"][i]) if "answer" in ds.column_names else ""
        r  = _to_str(ds["reference"][i]) if "reference" in ds.column_names else ""
        g  = ds["ground_truths"][i] if "ground_truths" in ds.column_names else [r]
        c  = ds["contexts"][i] if "contexts" in ds.column_names else []

        q_s   = _min_sanitize(q)
        a_s   = _min_sanitize(a, hint=q_s)
        gts_s = _coerce_list_of_str(g)
        ctxs_s= _coerce_list_of_str(c)

        # ★ PATCH: contexts 최소 1개 보장 + 길이 제어(_clip_contexts에서 이미 처리)
        if not any(ctxs_s): ctxs_s = [q_s]

        ref_s = _min_sanitize(r if r else (gts_s[0] if gts_s else q_s), hint=q_s)

        cols["question"].append(q_s)
        cols["answer"].append(a_s)
        cols["ground_truths"].append(gts_s)
        cols["reference"].append(ref_s)
        cols["contexts"].append(ctxs_s)
    return Dataset.from_dict(cols)

# ---------- NEW: 임베딩 빌더 ----------
def _build_embeddings():
    """
    기본: HuggingFace 임베딩(로컬/무료)
    OpenAI를 쓰려면: RAGAS_OPENAI=1, OPENAI_EMB_MODEL(기본 text-embedding-3-large)
    """
    use_openai = os.getenv("RAGAS_OPENAI", "0") == "1"
    if use_openai:
        try:
            from langchain_openai import OpenAIEmbeddings
            model = os.getenv("OPENAI_EMB_MODEL", "text-embedding-3-large")
            print(f"[RAGAS] Using OpenAIEmbeddings: model={model}")
            return OpenAIEmbeddings(model=model)
        except Exception as e:
            print(f"[RAGAS][WARN] OpenAIEmbeddings 사용 실패 → HF로 폴백: {e}")

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model = os.getenv("RAGAS_EMB", "sentence-transformers/all-MiniLM-L6-v2")
        device = os.getenv("HF_DEVICE", "cpu")
        print(f"[RAGAS] Using HuggingFaceEmbeddings: model={model}, device={device}")
        return HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": device})
    except Exception as e:
        print(f"[RAGAS][ERROR] HuggingFaceEmbeddings 생성 실패: {e}")
        # ★ PATCH: 해시 임베딩 폴백(차원 384)
        class _HashEmb:
            dim = 384
            def _v(self, s: str):
                import hashlib, struct
                h = hashlib.blake2b(s.encode("utf-8","ignore"), digest_size=64).digest()
                # 64바이트 → 16개 float → 16*24=384차원으로 반복
                floats = list(struct.unpack("16f", h[:64]))
                rep = math.ceil(self.dim/len(floats))
                vec = (floats*rep)[:self.dim]
                return vec
            def embed_query(self, s: str): return self._v(s or "")
            def embed_documents(self, arr: List[str]): return [self._v(x or "") for x in arr]
        print("[RAGAS][FALLBACK] Using hash-embeddings (dim=384)")
        return _HashEmb()

def ragas_evaluate_with_row_debug(ds: Dataset, out_dir_file: str):
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from types import SimpleNamespace

    def _default_score_dict():
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None, "context_recall": None}

    def _merge_scores(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            if isinstance(src, dict) and k in src and src[k] is not None:
                dst[k] = src[k]
        return dst

    ds_safe = _make_ds_safe(ds)  # ★ PATCH: 강화된 스키마 보정

    embeddings = _build_embeddings()

    use_full = os.getenv("RAGAS_FULL", "0") == "1"
    n = len(ds_safe)
    merged = [_default_score_dict() for _ in range(n)]

    if use_full:
        try:
            res_full = evaluate(ds_safe, metrics=[faithfulness, answer_relevancy, context_precision, context_recall], embeddings=embeddings)
            return res_full
        except Exception as e:
            print(f"\n[RAGAS][WARN] full evaluate() failed → per-metric fallback: {type(e).__name__}: {e}")

    metric_plan = [
        ("answer_relevancy", [answer_relevancy]),
        ("context_precision", [context_precision]),
        ("context_recall", [context_recall]),
        ("faithfulness", [faithfulness]),  # 마지막: 실패시 패스
    ]

    err_dump = os.path.join(out_dir_file, "ragas_errors.jsonl")

    def _eval_and_merge(tag: str, metric_list):
        try:
            r = evaluate(ds_safe, metrics=metric_list, embeddings=embeddings)
            sc_list = list(r.scores or [])
            for i in range(min(n, len(sc_list))):
                val = sc_list[i]
                if isinstance(val, dict):
                    merged[i] = _merge_scores(merged[i], val)
                else:
                    try:
                        merged[i][tag] = float(val)
                    except Exception:
                        merged[i][tag] = val
            print(f"[RAGAS] metric '{tag}' ✓")
        except Exception as e:
            print(f"[RAGAS] metric '{tag}' ✗ → {type(e).__name__}: {e}")
            try:
                os.makedirs(out_dir_file, exist_ok=True)
                with open(os.path.join(out_dir_file, "ragas_bad_rows.jsonl"), "a", encoding="utf-8") as f:
                    for i in range(n):
                        f.write(json.dumps({"idx": i, **{k: ds_safe[k][i] for k in ds_safe.column_names}}, ensure_ascii=False) + "\n")
                with open(err_dump, "a", encoding="utf-8") as ef:
                    ef.write(json.dumps({"metric": tag, "error": f"{type(e).__name__}: {str(e)}"}, ensure_ascii=False) + "\n")
            except Exception as dump_err:
                print(f"  (dump failed: {dump_err})")

    for name, one_metric in metric_plan:
        _eval_and_merge(name, one_metric)

    return SimpleNamespace(scores=merged)

# =========================================================
# 5) 기타 유틸 (기존 + 일부 보정)
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
    m = re.search(r"([0-9]+)\s*문제", p); k = int(m.group(1)) if m else 0
    if k <= 0:
        m2 = re.search(r"(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*문제", p)
        if m2: k = _KOR_NUM.get(m2.group(1), 0)
    if k <= 0: k = 5
    subj = "전체 범위"
    for s in SUBJECTS:
        if s.replace(" ","") in p.replace(" ",""): subj = s; break
    return subj, k

def _serialize_questions(questions: List[Dict[str, Any]]) -> str:
    lines=[]
    for i,q in enumerate(questions or [], start=1):
        qt = _norm(q.get("question",""))
        opts = [_norm(o) for o in (q.get("options") or [])][:4]
        # ★ PATCH: 보기에서 선행 번호/기호 제거 강화
        clean_opts = [re.sub(r"^\s*(?:\d+[\.\)]\s*|[\-\*\•]\s*)", "", o) for o in opts]
        lines.append(f"[문제 {i}] {qt}")
        lines.append("[보기]")
        for j,o in enumerate(clean_opts, start=1):
            lines.append(f"{j}) {o}")
        lines.append("")
    txt = "\n".join(lines).strip()
    return txt if txt else "."

def _serialize_gt(gt_list: List[Dict[str, Any]]) -> str:
    return _serialize_questions(gt_list or [])

# =========================================================
# 6) 한 샘플 평가 + jsonl 로깅 (대체로 기존 유지)
# =========================================================
def _coerce_gt(gt) -> List[Dict[str, Any]]|List[str]:
    if gt is None: return []
    if isinstance(gt, list): return gt
    return [gt]

# ===== [PATCH] Missing helpers & numpy import =====
import numpy as np  # 유사도 계산용

def _default_scores():
    return {"faithfulness": None, "answer_relevancy": None, "context_precision": None, "context_recall": None}

def _append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _extract_questions_from_result(res: Dict[str, Any], subject: str) -> List[Dict[str, Any]]:
    """
    에이전트 반환 구조의 다양한 케이스를 안전하게 커버.
    """
    result = (res or {}).get("result") or {}
    # 최우선: 과목별
    subj_q = (
        result.get("result", {}).get("subjects", {}).get(subject, {}).get("questions")
        if isinstance(result.get("result"), dict) else None
    )
    if isinstance(subj_q, list) and subj_q:
        return subj_q

    # 전역 all_questions
    for key in ("all_questions", "questions"):
        q = result.get(key) or (res.get(key) if isinstance(res, dict) else None)
        if isinstance(q, list) and q:
            return q

    # 최후: state 안쪽
    state = res.get("state") or {}
    for key in ("all_questions", "questions"):
        q = state.get(key)
        if isinstance(q, list) and q:
            return q
    return []

def _is_degenerate_case(answer_text: str, contexts_list: List[str]) -> bool:
    if not answer_text or answer_text.strip() in {".", ""}:
        return True
    if not contexts_list or all((not (c or "").strip()) for c in contexts_list):
        return True
    if len(answer_text.strip()) < 5:
        return True
    return False

# --- 과목/키워드 기반 1차 필터 ---
def _filter_ctx_by_subject_keyword(ctxs: List[str], subject: str, question_text: str, min_keep: int = 2) -> List[str]:
    def _normtxt(t):
        return re.sub(r"\s+", " ", (t or "")).strip()
    subj_key = (subject or "").replace(" ", "")
    q_words = [w for w in re.split(r"[^가-힣A-Za-z0-9]+", question_text or "") if len(w) >= 2]
    q_words_lc = set(w.lower() for w in q_words)

    scored = []
    for c in (ctxs or []):
        txt = _normtxt(c)
        if not txt:
            continue
        s = 0
        if subj_key and (subj_key in txt.replace(" ", "")):
            s += 2
        lc = txt.lower()
        hit = sum(1 for w in q_words_lc if w and w in lc)
        s += min(hit, 3)
        scored.append((s, txt))
    if not scored or max(s for s, _ in scored) == 0:
        return (ctxs or [])[:max(min_keep, 3)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored][:max(min_keep, 3)]

# --- 임베딩 기반 재랭킹 ---
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _rerank_ctx_by_similarity(query: str, ctxs: List[str], top_k: int = 3) -> List[str]:
    if not ctxs:
        return []
    try:
        emb = _build_embeddings()
        if emb is None:
            # 폴백: 원본 상위 N
            return ctxs[:max(1, top_k)]
        # langchain-like 인터페이스 기대
        qv = np.array(emb.embed_query(query or ""), dtype=np.float32)
        cvs = np.array(emb.embed_documents(ctxs), dtype=np.float32)
        scores = [(_cosine(qv, cvs[i]), ctxs[i]) for i in range(len(ctxs))]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scores[:max(1, top_k)]]
    except Exception as e:
        print(f"[RERANK][WARN] fallback(no-emb or calc): {e}")
        return ctxs[:max(1, top_k)]

# ... (기존 eval_one / run / main 이하 코드는 네가 쓰던 그대로 두어도 안전하게 돌아가도록 위에서 방어막을 쳐둠)

def eval_one(item: Dict[str, Any], agent, milvus_data: Dict[str, Any], out_dir_file: str) -> Dict[str, Any]:
    prompt = _norm(item.get("question",""))
    subject, k = _parse_subject_k(prompt)

    milvus_local = dict(milvus_data); milvus_local["connection_status"] = True

    def build_payload(base: Dict[str, Any]) -> Dict[str, Any]:
        base = dict(base)
        base["milvus_data"] = dict(milvus_local)
        base["save_to_file"] = False
        base.setdefault("difficulty", "중급")
        base["force_offline_retrieval"] = False
        base["use_milvus"] = True
        base.setdefault("return_state", True)
        base.setdefault("include_contexts", True)
        base.setdefault("debug", True)
        return base

    if subject in SUBJECTS:
        base = {"mode": "subject_quiz", "subject_area": subject, "target_count": k}
    else:
        per = max(1, k // 5) or 1
        base = {"mode": "partial_exam", "selected_subjects": SUBJECTS, "questions_per_subject": per}

    try:
        res = agent.invoke(build_payload(base))
    except Exception as e:
        print(f"[generator][ERROR] invoke 실패: {type(e).__name__}: {e}")
        res = {}

    questions = _extract_questions_from_result(res, subject) or []
    if len(questions) > k: questions = questions[:k]

    generated = _serialize_questions(questions)
    golden = _serialize_gt(_coerce_gt(item.get("ground_truth") or item.get("ground_truths")))

    
    milvus_ctxs = _extract_milvus_contexts_from_res(res, subject)
    print("[DEBUG][RAGAS] milvus_ctxs_len=", len(milvus_ctxs))
    if milvus_ctxs: print("[DEBUG][RAGAS] ctx_head=", repr(milvus_ctxs[0][:120]))
    print("[DEBUG][RAGAS] gen_head=", repr(generated[:120]))
    
    # 1차 과목/키워드 필터
    milvus_ctxs = _filter_ctx_by_subject_keyword(milvus_ctxs, subject, prompt, min_keep=2)
    # 2차 임베딩 재랭킹(질문+생성문항을 쿼리로)
    rerank_query = (prompt + "\n" + generated)[:800]  # 쿼리 너무 길면 자르기
    milvus_ctxs = _rerank_ctx_by_similarity(rerank_query, milvus_ctxs, top_k=3)

    rm_only_total = 0; rm_pref_total = 0
    if _is_degenerate_case(generated, milvus_ctxs):
        print("[RAGAS][SKIP] Empty/degenerate sample → default scores")
        sc = _default_scores()
        q_for_ragas = sanitize_for_ragas(prompt, role="question", fallback_hint=prompt)
        a_for_ragas = sanitize_for_ragas(generated, role="answer", fallback_hint=prompt)
        gt_for_ragas = sanitize_for_ragas(golden, role="ground_truth", fallback_hint=prompt)
        ctx_blob = sanitize_for_ragas(prompt, role="context", fallback_hint=prompt)
    else:
        cleaned_ctxs=[]
        for c in milvus_ctxs:
            c = c if isinstance(c, str) else str(c or "")
            c_norm = _normalize_unicode_space(c)
            c1, rm_only = _strip_enumeration_only_lines(c_norm)
            c2, rm_pref = _strip_enumeration_prefix_lines(c1)
            rm_only_total += rm_only; rm_pref_total += rm_pref
            cleaned_ctxs.append(c2)

        ctx_blob = "\n\n".join(
            sanitize_for_ragas(c, role="context", fallback_hint=prompt)
            for c in cleaned_ctxs if c and str(c).strip()
        ) or sanitize_for_ragas(prompt, role="context", fallback_hint=prompt)

        q_for_ragas = sanitize_for_ragas(prompt, role="question", fallback_hint=prompt)
        a_for_ragas = sanitize_for_ragas(generated, role="answer", fallback_hint=prompt)
        gt_for_ragas = sanitize_for_ragas(golden, role="ground_truth", fallback_hint=prompt)

        print("[RAGAS][sentences] q:", _debug_sentences(q_for_ragas))
        print("[RAGAS][sentences] a:", _debug_sentences(a_for_ragas))
        print("[RAGAS][sentences] ctx(n_char={}):".format(len(ctx_blob)), _debug_sentences(ctx_blob)[:3])

        from datasets import Dataset
        ds = Dataset.from_dict({
            "question":      [q_for_ragas],
            "contexts":      [cleaned_ctxs],
            "answer":        [a_for_ragas],
            "ground_truths": [[gt_for_ragas]],
            "reference":     [gt_for_ragas],
        })
        ragas_res = ragas_evaluate_with_row_debug(ds, out_dir_file)
        sc = (ragas_res.scores or [{}])[0] if hasattr(ragas_res, "scores") else _default_scores()

    dbg_path = os.path.join(out_dir_file, "agent_ragas_inputs.jsonl")
    try:
        _append_jsonl(dbg_path, {
            "question": q_for_ragas,
            "contexts": [ctx_blob],
            "answer":   a_for_ragas,
            "ground_truth": gt_for_ragas,
            "ragas_inputs": {
                "question": q_for_ragas,
                "contexts": [[ctx_blob]],
                "answer": a_for_ragas,
                "ground_truths": [gt_for_ragas],
                "reference": gt_for_ragas,
            },
            "agent_min": {
                "questions_n": len(questions or []),
                "generated_head": generated[:200],
                "milvus_ctxs_n": len(milvus_ctxs),
                "milvus_ctxs_head": [c[:150] for c in milvus_ctxs[:2]],
            },
            "clean_notes": {
                "enum_only_lines_removed": rm_only_total,
                "enum_prefix_lines_removed": rm_pref_total
            },
            "meta": {"subject": subject, "k": k},
        })
    except Exception as e:
        print(f"[WARN] jsonl dump failed: {e}")

    return {
        "prompt": prompt,
        "subject": subject,
        "k": k,
        "n_generated": len(questions or []),
        "scores": sc,
        "preview": generated[:300],
    }

# =========================================================
# 7) 메인 러너
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
    if not items: raise ValueError("지원하지 않는 골든셋 포맷")
    return items

def run(golden_path: str,
        out_dir: str = "./teacher/agents/TestGenerator/eval_results",
        milvus_host="localhost", milvus_port="19530", emb_model="jhgan/ko-sroberta-multitask",
        limit: int | None = None):
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    items = _load_golden(golden_path)
    if isinstance(limit, int) and limit > 0: items = items[:limit]

    milvus_data = _build_milvus_data(milvus_host, milvus_port, emb_model)
    agent = AgentCls()

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

    import pandas as pd
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
    print(f"ℹ️ 디버그 jsonl: {os.path.join(out_dir_file, 'agent_ragas_inputs.jsonl')}")
    print(f"ℹ️ 에러 덤프:   {os.path.join(out_dir_file, 'ragas_bad_rows.jsonl')} / ragas_errors.jsonl")

def main():
    target = os.getenv("GOLDENSET_PATH", "teacher/agents/TestGenerator/goldensets/generator_golden_ragas_style_5.jsonl")
    outdir = os.getenv("OUT_DIR", "./teacher/agents/TestGenerator/eval_results")
    run(target, outdir,
        milvus_host=os.getenv("MILVUS_HOST","localhost"),
        milvus_port=os.getenv("MILVUS_PORT","19530"),
        emb_model=os.getenv("EMB_MODEL","jhgan/ko-sroberta-multitask"),
        limit=int(os.getenv("LIMIT","0")) or None)

if __name__ == "__main__":
    main()
