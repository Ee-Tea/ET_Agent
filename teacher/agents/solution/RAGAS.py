# RAGAS_runner.py
import os, json, glob, re, logging, traceback
from typing import Any, Dict, List
from dataclasses import dataclass

# === 여러분이 올린 파일을 그대로 import ===
from common.milvus_helpers import get_milvus_connection_info
from teacher.agents.solution.solution_agent import SolutionAgent

# ragas는 지연 import (설치 필요: pip install ragas datasets)
from datasets import Dataset

# -----------------------------
# 유틸/전처리/디버그 헬퍼
# -----------------------------
from types import SimpleNamespace

# 환경 스위치: 원문 그대로 쓸지 여부(1이면 그대로 사용)
KEEP_RAW = os.getenv("RAGAS_KEEP_RAW", "0") == "1"

# 유니코드 공백/제로폭/풀와이드 마침표 등 정규화
_ZWS = "\u200b\u200c\u200d\ufeff"
def _normalize_unicode_space(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # NBSP -> space, 제로폭 제거
    s = (s.replace("\xa0", " ")
           .replace("\u00ad", "")      # soft hyphen
           .translate({ord(c): None for c in _ZWS}))
    # 풀와이드/전각 기호 → 일반 기호
    s = s.replace("．", ".").replace("。", ".").replace("！", "!").replace("？", "?").replace("⋯", "…")
    return s

_END_PUNCT_RE = re.compile(r"(?:[\.!?…]+(?:\s*)|[\.!?…]+[”’'\")\]\}]+\s*|\s*[”’'\")\]\}]+\s*)$")

def _sanit_flags(name: str, s: str):
    tail = (s or "")[-6:]
    tail_codes = " ".join([f"U+{ord(c):04X}" for c in tail])
    flags=[]
    if s is None or not str(s).strip():
        flags.append("empty")
    else:
        body = re.sub(r"(정답\s*:\s*\)?\s*)|(풀이\s*:\s*)|(과목\s*:\s*)", "", s or "", flags=re.IGNORECASE).strip()
        if not body:
            flags.append("only_labels")
        if not _END_PUNCT_RE.search(s or ""):
            flags.append("no_end_punct")
        if len((s or "").strip()) < 8:
            flags.append("too_short")
    print(f"[RAGAS][{name}] flags={flags} len={len(s or '')} tail={repr(tail)} tail_codes=[{tail_codes}]")

def _is_semantically_empty(s: str) -> bool:
    if not s or not str(s).strip():
        return True
    body = re.sub(r"(정답\s*:\s*\)?\s*)|(풀이\s*:\s*)|(과목\s*:\s*)", "", str(s), flags=re.IGNORECASE)
    body = _normalize_unicode_space(body)
    return len((body or "").strip()) == 0

def sanitize_for_ragas(text: str, role: str = "answer", fallback_hint: str = "") -> str:
    """
    - 유니코드 공백/제로폭/풀와이드 기호 표준화
    - 레이블만 있는 답변/정답지 → 질문 일부로 placeholder 채움
    - 종결부호가 없거나 닫힘기호로 끝나면 종결부호를 닫힘기호 '앞'에 삽입
    - 과도 개행 정리
    """
    if KEEP_RAW:
        # 원문 그대로(필요 최소한의 캐스팅/트리밍만)
        t = _normalize_unicode_space((text or "")).strip()
        if not t:
            t = _normalize_unicode_space((fallback_hint or "")).strip() or ""
        # contexts 등 완전 빈 문자열은 허용하지만 끝에서 깨지지 않게 최소 처리
        return t

    _sanit_flags(f"{role}(before)", text)
    t = _normalize_unicode_space((text or "").strip())

    if _is_semantically_empty(t):
        fh = _normalize_unicode_space((fallback_hint or "").strip())[:120] or "내용 없음"
        t = fh

    if not _END_PUNCT_RE.search(t):
        m = re.search(r"[”’'\")\]\}]+$" , t)
        if m:
            closers = m.group(0)
            core = t[:m.start()].rstrip()
            if core and core[-1] not in ".!?…":
                t = core + "." + closers
            else:
                t = core + closers
        else:
            t = t.rstrip() + "."

    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    _sanit_flags(f"{role}(after)", t)
    return t

def _doc_to_text_list(items) -> List[str]:
    """
    SolutionAgent가 state에 넣어두는 컨텍스트들을 문자열 리스트로 통일.
    - langchain.schema.Document 또는 dict 둘 다 허용
    - keys: page_content / content / explanation 등에서 안전하게 추출
    """
    out: List[str] = []
    for d in items or []:
        txt = getattr(d, "page_content", None)
        if not txt and isinstance(d, dict):
            txt = d.get("page_content") or d.get("content") or d.get("text")
        if not txt:
            md = getattr(d, "metadata", {}) if not isinstance(d, dict) else d.get("metadata", {})
            if isinstance(md, dict):
                txt = md.get("explanation") or md.get("content")
        txt = (txt or "")
        if str(txt).strip():
            out.append(str(txt).strip())
    deduped = list(dict.fromkeys(out))
    return deduped

def _build_question_with_options(q: str, options: List[str]) -> str:
    lines = ["[문제]", (q or "").strip(), "", "[보기]"]
    for i, o in enumerate(options or [], start=1):
        lines.append(f"{i}) {str(o)}")
    return "\n".join(lines)

def _debug_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    t = re.sub(r"\s+", " ", t)
    parts = re.split(r"(?<=[\.!?])\s+|\n+", t)
    return [p.strip() for p in parts if p and p.strip()]

def _min_sanitize(s: Any, hint: str = "") -> str:
    """
    RAGAS 진입 직전 '최소' 안전화:
    - str 캐스팅, 제로폭/전각기호 정규화
    - 완전 비거나 레이블뿐이면 hint(보통 질문)로 대체
    - 문장 종결부호 보강(., !, ?, … 중 하나로 끝나게)
    - 너무 공격적이지 않게 최소만 한다
    """
    s = _normalize_unicode_space(str(s or "")).strip()
    if not s:
        s = _normalize_unicode_space(str(hint or "")).strip()
    if not s:
        s = "."
    if not _END_PUNCT_RE.search(s):
        # 닫힘기호가 꼬리에 있으면 그 '앞'에 점을 넣음
        m = re.search(r"[”’'\")\]\}]+$", s)
        if m:
            closers = m.group(0)
            core = s[:m.start()].rstrip()
            s = (core + "." + closers) if core and core[-1] not in ".!?…" else (core + closers)
        else:
            s = s.rstrip() + "."
    return s

def _make_ds_safe(ds: Dataset) -> Dataset:
    """
    어떤 입력이 오든 RAGAS가 기대하는 스키마/형식으로 안전하게 변환.
    - question/answer/reference: 최소 한 문장 보장
    - ground_truths: 비지 않게, 모두 str, 최소 한 문장 보장
    - contexts: List[str], 최소 1개 보장, 각 항목 한 문장 보장
    """
    data = {"question": [], "answer": [], "contexts": [], "ground_truths": [], "reference": []}
    n = len(ds)
    for i in range(n):
        # 원본 값 뽑기(없으면 기본값)
        q  = ds["question"][i] if "question" in ds.column_names else ""
        a  = ds["answer"][i] if "answer" in ds.column_names else ""
        r  = ds["reference"][i] if "reference" in ds.column_names else ""
        gts = ds["ground_truths"][i] if "ground_truths" in ds.column_names else [r]
        ctxs = ds["contexts"][i] if "contexts" in ds.column_names else []

        # 리스트 강제
        if not isinstance(gts, list):  gts  = [gts]
        if not isinstance(ctxs, list): ctxs = [ctxs]

        # 최소 전처리(KEEP_RAW 여부와 무관하게 RAGAS 직전에만 적용)
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
    """
    evaluate()를 감싸서, 실패하면 문제 row를 1개씩 평가해 범인 식별 + 덤프.
    이후 '정상 행만 평가' + '원래 길이로 재조합'하여 .scores를 반환 → 인덱스 에러 방지.
    배치/단일 평가 모두 최소 안전 전처리된 ds_safe를 사용.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from types import SimpleNamespace

    def _default_score_dict() -> Dict[str, Any]:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

    # ✅ RAGAS 직전, 무조건 안전화(KEEP_RAW와 무관)
    ds_safe = _make_ds_safe(ds)

    try:
        return evaluate(ds_safe, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    except Exception as e:
        print(f"\n[RAGAS][ERROR] evaluate() failed: {type(e).__name__}: {e}")
        traceback.print_exc()

        bad_rows, good_rows = [], []
        for i in range(len(ds_safe)):
            one = {k: [ds_safe[k][i]] for k in ds_safe.column_names}
            ds1 = Dataset.from_dict(one)
            try:
                _ = evaluate(ds1, metrics=[answer_relevancy])
                good_rows.append(i)
            except Exception as e2:
                print(f"\n[RAGAS][BAD-ROW] idx={i} → {type(e2).__name__}: {e2}")
                q = ds_safe['question'][i]
                a = ds_safe['answer'][i]
                gts = ds_safe['ground_truths'][i]
                print(f"  question(head): {str(q)[:200].replace('\\n',' ')}")
                print(f"  answer(head):   {str(a)[:200].replace('\\n',' ')}")
                print(f"  ground_truths:  {gts[:1]} ... (n={len(gts) if isinstance(gts, list) else 'NA'})")
                ans_sents = _debug_sentences(a)
                print(f"  answer sentence guess → {len(ans_sents)} sentences: {ans_sents[:3]}")
                bad_rows.append(i)
                # 덤프
                try:
                    os.makedirs(out_dir_file, exist_ok=True)
                    dump_path = os.path.join(out_dir_file, "ragas_bad_rows.jsonl")
                    with open(dump_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "idx": i,
                            "question": q,
                            "answer": a,
                            "ground_truths": gts,
                            "contexts_n": len(ds_safe['contexts'][i]) if 'contexts' in ds_safe.column_names else None,
                            "answer_sent_debug": ans_sents
                        }, ensure_ascii=False) + "\n")
                    print(f"  → dumped to {dump_path}")
                except Exception as dump_err:
                    print(f"  (dump failed: {dump_err})")

        if len(good_rows) == 0:
            print("[RAGAS] 모든 행이 실패 → 모든 점수를 None으로 채워 반환")
            full_scores = [_default_score_dict() for _ in range(len(ds_safe))]
            return SimpleNamespace(scores=full_scores)

        # 정상 행만 재평가
        ds_ok = Dataset.from_dict({k: [ds_safe[k][i] for i in good_rows] for k in ds_safe.column_names})
        try:
            res_ok = evaluate(ds_ok, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
            ok_scores = list(res_ok.scores or [])
        except Exception as e3:
            print(f"[RAGAS] 정상 행 재평가도 실패 → 모든 점수 None 반환: {e3}")
            full_scores = [_default_score_dict() for _ in range(len(ds_safe))]
            return SimpleNamespace(scores=full_scores)

        # 원래 길이로 재조합
        full_scores = []
        ok_iter = iter(ok_scores)
        bad_set = set(bad_rows)
        for i in range(len(ds_safe)):
            if i in bad_set:
                full_scores.append(_default_score_dict())
            else:
                sc = next(ok_iter, None)
                merged = _default_score_dict()
                if isinstance(sc, dict):
                    # ragas 버전에 따라 키 누락 대비
                    for k in merged.keys():
                        merged[k] = sc.get(k)
                full_scores.append(merged)

        print(f"[RAGAS] 재조합 완료: total={len(full_scores)} good={len(good_rows)} bad={len(bad_rows)}")
        return SimpleNamespace(scores=full_scores)


def repro_answer_relevancy(question: str, answer: str, ground_truths: List[str]):
    from ragas import evaluate
    from ragas.metrics import answer_relevancy
    ds = Dataset.from_dict({
        "question": [question or ""],
        "answer": [answer or ""],
        "contexts": [[]],              # 비워도 됨
        "ground_truths": [ground_truths or [""]],
        "reference": [ (ground_truths[0] if ground_truths else "") ],
    })
    print("[REPRO] q(head):", (question or "")[:150])
    print("[REPRO] a(head):", (answer or "")[:150])
    print("[REPRO] gts:", (ground_truths or [])[:1])
    ans_sents = _debug_sentences(answer)
    print("[REPRO] answer sentences:", ans_sents)
    return evaluate(ds, metrics=[answer_relevancy])

# -----------------------------
# 평가 파이프라인
# -----------------------------
@dataclass
class EvalRow:
    question: str
    contexts: List[str]
    answer: str
    ground_truth: str
    ground_truths: List[str]  # ragas가 list 컬럼을 더 안정적으로 지원
    meta: Dict[str, Any]

def run_one_golden_json(
    json_path: str,
    out_dir: str,
    milvus_data: Dict[str, Any],
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # questions 배열/혹은 리스트 둘 다 지원 (여러 포맷을 위해)
    if isinstance(raw, dict) and isinstance(raw.get("questions"), list):
        items = raw["questions"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"지원하지 않는 JSON 구조: {json_path}")

    agent = SolutionAgent()
    rows: List[EvalRow] = []

    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue

        q_text   = (it.get("question") or "").strip()
        options  = it.get("options") or []
        if not isinstance(options, list):
            # 이상치 방지
            try:
                options = list(options)
            except Exception:
                options = []
        options = [str(o) for o in options]

        ans_str  = str(it.get("answer") or "").strip()
        gt_idx   = int(ans_str) if ans_str.isdigit() else None
        gt_text  = options[gt_idx - 1] if (gt_idx is not None and 1 <= gt_idx <= len(options)) else ""
        gt_exp   = (it.get("explanation") or "").strip()
        gt_sub   = (it.get("subject") or "").strip()

        user_input = "이 문제의 정답 번호와 풀이, 그리고 과목명을 알려주세요."
        state = agent.invoke(
            user_input_txt=user_input,
            user_problem=q_text,
            user_problem_options=options,
            milvus_data={
                "connection_status": True,
                "host": milvus_data.get("host", "localhost"),
                "port": milvus_data.get("port", "19530"),
                "embedding_model_name": milvus_data.get("embedding_model_name", "jhgan/ko-sroberta-multitask"),
            },
            recursion_limit=1000,
        )

        # 생성 결과
        pred_num = str(state.get("generated_answer") or "").strip()
        pred_idx = int(pred_num) if pred_num.isdigit() else None
        pred_text = options[pred_idx - 1] if (pred_idx is not None and 1 <= pred_idx <= len(options)) else ""
        pred_exp  = (state.get("generated_explanation") or "").strip()
        pred_sub  = (state.get("generated_subject") or "").strip()

        # 컨텍스트(문자열 리스트)
        ctx_texts: List[str] = []
        ctx_texts += _doc_to_text_list(state.get("problems_contexts") or [])
        ctx_texts += _doc_to_text_list(state.get("concept_contexts") or [])

        # *_text 가 있으면 블록으로 분리해서 추가
        for key in ("problems_contexts_text", "concept_contexts_text"):
            t = (state.get(key) or "").strip()
            if t:
                blocks = [b.strip() for b in re.split(r"\n\s*\n", t) if b.strip()]
                ctx_texts.extend(blocks)

        # 중복/공백 정리 + 문자열화
        ctx_texts = [str(c).strip() for c in dict.fromkeys(ctx_texts) if str(c).strip()]

        def _san(s: str, role: str, hint: str) -> str:
            return sanitize_for_ragas(s, role=role, fallback_hint=hint)

        # 컨텍스트 보정
        ctx_texts = [_san(c, "context", q_text) for c in ctx_texts if str(c).strip()]
        if not ctx_texts:
            # 완전 빈 리스트 방지
            ctx_texts = [_san(q_text or "컨텍스트 없음", "context", q_text)]

        # 질문/정답/정답지 포맷(+안전보정)
        q_full = _build_question_with_options(q_text, options)
        q_full = _san(q_full, "question", q_text)

        gt_blob = f"정답: {gt_idx}) {gt_text}".strip() if gt_idx else f"정답: {gt_text}".strip()
        # if gt_exp: gt_blob += f"\n풀이: {gt_exp}"
        if gt_sub: gt_blob += f"\n과목: {gt_sub}"
        gt_blob = _san(gt_blob, "ground_truth", q_text)

        pred_blob = f"정답: {pred_num}) {pred_text}\n풀이: {pred_exp}\n과목: {pred_sub}".strip()
        pred_blob = _san(pred_blob, "answer", q_text)

        rows.append(EvalRow(
            question=q_full,
            contexts=ctx_texts,
            answer=pred_blob,
            ground_truth=gt_blob,
            ground_truths=[gt_blob],
            meta={
                "options": options,
                "gt_answer_idx": gt_idx,
                "pred_idx": pred_idx,
                "gt_subject": gt_sub,
                "pred_subject": pred_sub,
                "validated": bool(state.get("validated", False)),
            }
        ))

    if not rows:
        return {"file": json_path, "n": 0, "ragas_csv": None, "leaderboard_csv": None}

    # RAGAS용 데이터셋 구성 (question, contexts, answer, ground_truths)
    def _ensure_list_str_list(v):
        # RAGAS는 contexts가 List[str] 여야 하므로 강제 보정
        if not isinstance(v, list):
            return [str(v or "")]
        out = []
        for x in v:
            out.append(str(x or ""))
        # 최소 1개 보장
        return out if out else [""]

    data = {
        "question":      [str(r.question or "") for r in rows],
        "contexts":      [_ensure_list_str_list(r.contexts) for r in rows],
        "answer":        [str(r.answer or "") for r in rows],
        "ground_truths": [[str(x or "") for x in (r.ground_truths or [""])] for r in rows],
        "reference":     [str(r.ground_truth or "") for r in rows],
    }
    ds = Dataset.from_dict(data)

    # 저장 경로(파일별 하위폴더) 먼저 준비
    import pandas as pd
    base = os.path.splitext(os.path.basename(json_path))[0]
    out_dir_file = os.path.join(out_dir, base)
    os.makedirs(out_dir_file, exist_ok=True)

    # 메트릭 준비 + 평가 실행 (디버그 래퍼 사용: 실패 행 건너뛰고 재조합)
    ragas_res = ragas_evaluate_with_row_debug(ds, out_dir_file)

    # 점수표 저장
    ragas_scores = pd.DataFrame(ragas_res.scores or [])
    # 길이 불일치 방지(원래 row 수로 강제 패딩)
    if len(ragas_scores) != len(rows):
        print(f"[WARN] ragas_scores len({len(ragas_scores)}) != rows({len(rows)}) → 패딩 조정")
        while len(ragas_scores) < len(rows):
            ragas_scores.loc[len(ragas_scores)] = {"faithfulness": None, "answer_relevancy": None, "context_precision": None, "context_recall": None}
        if len(ragas_scores) > len(rows):
            ragas_scores = ragas_scores.iloc[:len(rows)].reset_index(drop=True)

    ragas_csv = os.path.join(out_dir_file, "ragas_scores.csv")
    ragas_scores.to_csv(ragas_csv, index=False)

    # 리더보드(+정답률)
    md = [r.meta for r in rows]
    df_md = pd.DataFrame(md)

    def mc_acc(series):
        ok = 0; tot = 0
        for m in series:
            if not isinstance(m, dict): continue
            gt = m.get("gt_answer_idx")
            pr = m.get("pred_idx")
            if gt is None or pr is None: continue
            tot += 1; ok += int(gt == pr)
        return (ok / tot) if tot else None

    acc = mc_acc(df_md)
    # concat 전 길이 일치 확인
    if len(df_md) != len(ragas_scores):
        print(f"[WARN] meta({len(df_md)}) != scores({len(ragas_scores)}) → 인덱스 리셋 및 패딩")
        df_md = df_md.reset_index(drop=True)
        ragas_scores = ragas_scores.reset_index(drop=True)
        while len(ragas_scores) < len(df_md):
            ragas_scores.loc[len(ragas_scores)] = {"faithfulness": None, "answer_relevancy": None, "context_precision": None, "context_recall": None}
        if len(ragas_scores) > len(df_md):
            ragas_scores = ragas_scores.iloc[:len(df_md)].reset_index(drop=True)

    leaderboard = pd.concat([df_md, ragas_scores], axis=1)
    leaderboard["mc_accuracy"] = acc
    lb_csv = os.path.join(out_dir_file, "agent_eval_leaderboard.csv")
    leaderboard.to_csv(lb_csv, index=False)

    return {"file": json_path, "n": len(rows), "ragas_csv": ragas_csv, "leaderboard_csv": lb_csv}

def main():
    # 환경변수/인자값으로 설정
    GOLDEN_DIR = "./teacher/exam/test_parsed_exam_json"
    OUT_DIR    = os.getenv("EVAL_OUT_DIR", "./teacher/agents/solution/eval_results")
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    EMB_MODEL   = os.getenv("EMB_MODEL", "jhgan/ko-sroberta-multitask")

    # 버전/로그레벨(디버그에 도움)
    try:
        import ragas, datasets
        logging.getLogger("ragas").setLevel(logging.DEBUG)
        print(f"[VER] ragas={getattr(ragas, '__version__', 'unknown')}, datasets={getattr(datasets, '__version__', 'unknown')}")
        print(f"[CFG] RAGAS_KEEP_RAW={'ON' if KEEP_RAW else 'OFF'}")
    except Exception:
        pass

    milvus_data = {
        "connection_status": True,
        "host": MILVUS_HOST,
        "port": MILVUS_PORT,
        "embedding_model_name": EMB_MODEL,
    }

    # 연결 사전 점검(선택): 실패해도 계속 시도
    try:
        _ = get_milvus_connection_info(milvus_data)
    except Exception as e:
        print(f"[경고] Milvus 연결 사전 점검 중 오류: {e}")

    files = sorted(glob.glob(os.path.join(GOLDEN_DIR, "*.json")))
    if not files:
        print(f"[알림] 골든셋 JSON이 없습니다: {GOLDEN_DIR}")
        return

    summary = []
    for fp in files:
        print(f"\n=== {os.path.basename(fp)} 평가 시작 ===")
        try:
            res = run_one_golden_json(fp, OUT_DIR, milvus_data)
            summary.append(res)
            print(f"→ RAGAS: {res['ragas_csv']}, Leaderboard: {res['leaderboard_csv']}")
        except Exception as e:
            print(f"[ERROR] {os.path.basename(fp)} 평가 중단: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n=== 전체 요약 ===")
    for r in summary:
        print(f"{os.path.basename(r['file'])}: {r['n']}개 / scores={r['ragas_csv']} / lb={r['leaderboard_csv']}")

if __name__ == "__main__":
    main()
