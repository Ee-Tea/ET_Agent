# RAGAS_runner.py
import os, json, glob, re
from typing import Any, Dict, List
from dataclasses import dataclass

# === 여러분이 올린 파일을 그대로 import ===
from common.milvus_helpers import get_milvus_connection_info   # :contentReference[oaicite:3]{index=3}
from teacher.agents.solution.solution_agent import SolutionAgent

# ragas는 지연 import (설치 필요: pip install ragas datasets)
from datasets import Dataset

def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _doc_to_text_list(items) -> List[str]:
    """
    SolutionAgent가 state에 넣어두는 컨텍스트들을 문자열 리스트로 통일.
    - langchain.schema.Document 또는 dict 둘 다 허용
    - keys: page_content / content / explanation 등에서 안전하게 추출
    """
    out: List[str] = []
    for d in items or []:
        # Document처럼 접근
        txt = getattr(d, "page_content", None)
        if not txt and isinstance(d, dict):
            txt = d.get("page_content") or d.get("content") or d.get("text")
        if not txt:
            # 메타에 풀이글이 있을 수 있음
            md = getattr(d, "metadata", {}) if not isinstance(d, dict) else d.get("metadata", {})
            if isinstance(md, dict):
                txt = md.get("explanation") or md.get("content")
        txt = (txt or "").strip()
        if txt:
            out.append(txt)
    # 중복 제거
    deduped = list(dict.fromkeys(out))
    return deduped

def _build_question_with_options(q: str, options: List[str]) -> str:
    lines = ["[문제]", (q or "").strip(), "", "[보기]"]
    for i, o in enumerate(options or [], start=1):
        lines.append(f"{i}) {str(o)}")
    return "\n".join(lines)

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

    agent = SolutionAgent()  # 내부에서 LangGraph compile됨  :contentReference[oaicite:5]{index=5}
    rows: List[EvalRow] = []

    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue

        q_text   = (it.get("question") or "").strip()
        options  = it.get("options") or []
        ans_str  = (it.get("answer") or "").strip()
        gt_idx   = int(ans_str) if ans_str.isdigit() else None
        gt_text  = options[gt_idx - 1] if gt_idx and 1 <= gt_idx <= len(options) else ""
        gt_exp   = (it.get("explanation") or "").strip()
        gt_sub   = (it.get("subject") or "").strip()

        user_input = "이 문제의 정답 번호와 풀이, 그리고 과목명을 알려주세요."
        state = agent.invoke(
            user_input_txt=user_input,
            user_problem=q_text,
            user_problem_options=options,
            milvus_data={
                # SolutionAgent는 state["milvus_data"]를 통해 검색을 수행합니다.  :contentReference[oaicite:6]{index=6}
                "connection_status": True,
                "host": milvus_data.get("host", "localhost"),
                "port": milvus_data.get("port", "19530"),
                "embedding_model_name": milvus_data.get("embedding_model_name", "jhgan/ko-sroberta-multitask"),
            },
            recursion_limit=1000,
        )

        # 생성 결과
        pred_num = (state.get("generated_answer") or "").strip()
        pred_idx = int(pred_num) if pred_num.isdigit() else None
        pred_text = options[pred_idx - 1] if (pred_idx and 1 <= pred_idx <= len(options)) else ""
        pred_exp  = (state.get("generated_explanation") or "").strip()
        pred_sub  = (state.get("generated_subject") or "").strip()

        # 컨텍스트(문자열 리스트)
        # SolutionAgent는 problems_contexts / concept_contexts(+ *_text) 를 제공합니다.  :contentReference[oaicite:7]{index=7}
        ctx_texts = []
        ctx_texts += _doc_to_text_list(state.get("problems_contexts") or [])
        ctx_texts += _doc_to_text_list(state.get("concept_contexts") or [])
        if not ctx_texts:
            # *_text가 있으면 분리해 추가
            for key in ("problems_contexts_text", "concept_contexts_text"):
                t = (state.get(key) or "").strip()
                if t:
                    # 두 줄 간격 블록 분리
                    blocks = [b.strip() for b in re.split(r"\n\s*\n", t) if b.strip()]
                    ctx_texts.extend(blocks)
        # 중복/공백 정리
        ctx_texts = [c for c in dict.fromkeys([c.strip() for c in ctx_texts]) if c]

        # 질문/정답 포맷
        q_full = _build_question_with_options(q_text, options)
        gt_blob = f"정답: {gt_idx}) {gt_text}".strip()
        if gt_exp: gt_blob += f"\n풀이: {gt_exp}"
        if gt_sub: gt_blob += f"\n과목: {gt_sub}"

        pred_blob = f"정답: {pred_num}) {pred_text}\n풀이: {pred_exp}\n과목: {pred_sub}".strip()

        rows.append(EvalRow(
            question=q_full,
            contexts=ctx_texts,
            answer=pred_blob,
            ground_truth=gt_blob,
            ground_truths=[gt_blob],  # ragas가 list형을 선호
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
    data = {
        "question":      [r.question for r in rows],
        "contexts":      [r.contexts for r in rows],
        "answer":        [r.answer for r in rows],
        "ground_truths": [r.ground_truths for r in rows],  # list[str] 유지
        "reference":     [r.ground_truth for r in rows],   # ← 단일 문자열(필수)
    }
    ds = Dataset.from_dict(data)

    # 메트릭 준비 + 평가 실행  :contentReference[oaicite:8]{index=8}
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    ragas_res = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        # LLM/임베딩은 RAGAS.py의 설정을 그대로 활용해도 되고, 기본값이면 생략 가능  :contentReference[oaicite:9]{index=9}
    )

    # 저장
    import pandas as pd
    base = os.path.splitext(os.path.basename(json_path))[0]
    out_dir_file = os.path.join(out_dir, base)
    os.makedirs(out_dir_file, exist_ok=True)

    # 점수표
    ragas_scores = pd.DataFrame(ragas_res.scores)
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

    milvus_data = {
        "connection_status": True,  # SolutionAgent가 이 플래그를 확인  :contentReference[oaicite:10]{index=10}
        "host": MILVUS_HOST,
        "port": MILVUS_PORT,
        "embedding_model_name": EMB_MODEL,
    }

    # 연결 사전 점검(선택): 실패해도 계속 시도
    try:
        _ = get_milvus_connection_info(milvus_data)  # 내부에서 manager connect 시도  :contentReference[oaicite:11]{index=11}
    except Exception as e:
        print(f"[경고] Milvus 연결 사전 점검 중 오류: {e}")

    files = sorted(glob.glob(os.path.join(GOLDEN_DIR, "*.json")))
    if not files:
        print(f"[알림] 골든셋 JSON이 없습니다: {GOLDEN_DIR}")
        return

    summary = []
    for fp in files:
        print(f"\n=== {os.path.basename(fp)} 평가 시작 ===")
        res = run_one_golden_json(fp, OUT_DIR, milvus_data)
        summary.append(res)
        print(f"→ RAGAS: {res['ragas_csv']}, Leaderboard: {res['leaderboard_csv']}")

    print("\n=== 전체 요약 ===")
    for r in summary:
        print(f"{os.path.basename(r['file'])}: {r['n']}개 / scores={r['ragas_csv']} / lb={r['leaderboard_csv']}")

if __name__ == "__main__":
    main()
