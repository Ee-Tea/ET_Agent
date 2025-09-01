import os, re, json, glob
from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from teacher.agents.solution.solution_agent import SolutionAgent

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# 🔧 Milvus 연결/스키마 확인용
from pymilvus import connections, Collection, DataType
from langchain_milvus import Milvus

# --- run_eval() 마지막에 이어 붙이세요 ---
import numpy as np
import matplotlib.pyplot as plt
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQAI_API_KEY = os.getenv("GROQAI_API_KEY", "")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))


# === RAGAS 평가에 사용할 LLM/임베딩 명시 (필수) ===
# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0,
#     api_key=os.environ["OPENAI_API_KEY"],
# )
llm = ChatOpenAI(
    api_key=GROQAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model=OPENAI_LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    max_tokens=min(LLM_MAX_TOKENS, 2048),
)

emb = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cpu"}
)

# --------- Milvus 벡터스토어 연결 유틸 ---------
def connect_vectorstore(
    collection_name: str,
    host: str = None,
    port: str = None,
    model_name: str = "jhgan/ko-sroberta-multitask",
) -> Milvus:
    """
    컬렉션 스키마를 읽어 text/vector 필드명을 자동 추론한 뒤 LangChain Milvus 래퍼로 연결.
    (problems=vector/text, concepts=embedding/content 같은 이질 스키마 대응)
    """
    host = host or os.getenv("MILVUS_HOST", "localhost")
    port = port or os.getenv("MILVUS_PORT", "19530")

    # 동기 HTTP 스킴으로 연결 (Streamlit 아님 → async 루프 이슈 회피)
    if "default" not in connections.list_connections():
        connections.connect(alias="default", uri=f"http://{host}:{port}")

    # 스키마 탐색
    c = Collection(collection_name)
    vec_field = None
    text_field = None

    for f in c.schema.fields:
        if f.dtype == DataType.FLOAT_VECTOR and vec_field is None:
            vec_field = f.name

    varchar_fields = [f.name for f in c.schema.fields if f.dtype == DataType.VARCHAR]
    for cand in ("text", "page_content", "content", "question", "item_title", "title"):
        if cand in varchar_fields:
            text_field = cand
            break
    if text_field is None and varchar_fields:
        text_field = varchar_fields[0]

    if vec_field is None:
        raise RuntimeError(f"[Milvus] '{collection_name}'에 FLOAT_VECTOR 필드가 없습니다.")

    # Embeddings
    _emb = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})

    # LangChain Milvus VectorStore (검색만 할 것이므로 index/search_params는 생략)
    vs = Milvus(
        embedding_function=_emb,
        collection_name=collection_name,
        connection_args={"uri": f"http://{host}:{port}"},
        text_field=text_field,
        vector_field=vec_field,
    )
    print(f"✅ Milvus connected: {collection_name} (text_field={text_field}, vector_field={vec_field})")
    return vs

# ---------- 유틸 ----------
def _ctx_text(d) -> str:
    md = getattr(d, "metadata", {}) or {}
    # concepts는 content에, problems는 page_content에 있을 수 있음
    txt = (md.get("content") or getattr(d, "page_content", "") or "")
    return str(txt).strip()

def build_contexts(retrieved_problems, retrieved_concepts, *, min_len: int = 20, max_k: int = 10) -> List[str]:
    raw = [ _ctx_text(d) for d in (retrieved_problems or []) + (retrieved_concepts or []) ]
    # 공백 제거 → 중복 제거 → 너무 짧은 조각 제거 → 상한 clip
    cleaned = [s for s in (t.strip() for t in raw) if s]
    deduped = list(dict.fromkeys(cleaned))
    filtered = [s for s in deduped if len(s) >= min_len]
    return filtered[:max_k]

def build_question_with_options(question_text: str, options: List[str]) -> str:
    lines = ["[문제]", question_text, "", "[보기]"]
    for i, opt in enumerate(options, start=1):
        lines.append(f"{i}) {opt}")
    return "\n".join(lines)

def doc_to_ctx_text(d) -> str:
    meta = getattr(d, "metadata", {}) or {}
    opts = meta.get("options", [])
    if isinstance(opts, str):
        try:
            opts = json.loads(opts)
        except Exception:
            opts = [opts]
    opts_blob = "\n".join(f"{i+1}) {o}" for i, o in enumerate(opts)) if opts else ""
    base = str(getattr(d, "page_content", "")) or ""
    if opts_blob:
        return (base + "\n\n[보기]\n" + opts_blob)[:2000]
    return base[:2000]

def parse_idx_from_text(s: str) -> int | None:
    """
    모델 출력에서 '정답: 1' 또는 '(1)' 또는 '1번' 등 숫자 후보를 robust하게 추출
    """
    if not s:
        return None
    m = re.search(r"(?:정답[:\s]*|^|\s|\[|\()(\d{1,2})(?:\)|\]|\s|번|\.|,|$)", s)
    try:
        if m:
            val = int(m.group(1))
            return val
    except Exception:
        pass
    return None

from dataclasses import dataclass
@dataclass
class EvalRow:
    question: str
    contexts: List[str]
    answer: str
    ground_truth: str
    metadata: Dict[str, Any]

# ---------- 핵심: 테스트 JSON -> 실행 -> RAGAS 입력 ----------

def run_eval(
    test_json_path: str,
    out_dir: str = "./eval_out",
    vectorstore_p: Milvus = None,
    vectorstore_c: Milvus = None,
):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[로드] JSON 파일 불러오기: {test_json_path}")

    # ✅ 최상위에 questions 키가 있는 형식 대응
    with open(test_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "questions" in raw and isinstance(raw["questions"], list):
        items = raw["questions"]
        print(f"[정규화] 'questions' 키 발견 → 항목 {len(items)}개")
    elif isinstance(raw, list):
        items = raw
        print(f"[정규화] 최상위가 list → 항목 {len(items)}개")
    else:
        raise ValueError(f"지원하지 않는 JSON 구조입니다: top-level={type(raw).__name__}")

    print(f"[로드 완료] 문제 수: {len(items)}")

    agent = SolutionAgent()
    print("[에이전트 초기화 완료]")

    rows: List[EvalRow] = []
    golden_rows: List[Dict[str, Any]] = []

    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            print(f"[경고] {i}번째 항목이 dict가 아님 → 건너뜀 (type={type(it).__name__})")
            continue

        print(f"\n[{i}] 문제 처리 시작")
        q_text     = (it.get("question") or "").strip()
        options    = it.get("options") or []
        gt_idx_str = (it.get("answer") or "").strip()
        gt_idx     = int(gt_idx_str) if gt_idx_str.isdigit() else None
        gt_text    = options[gt_idx - 1] if (gt_idx and 1 <= gt_idx <= len(options)) else ""
        gt_exp     = (it.get("explanation") or "").strip()
        gt_sub     = (it.get("subject") or "").strip()

        user_input_txt = "이 문제의 정답이 되는 보기 번화와 문제 풀이, 그리고 이 문제의 과목 이름을 알려주세요."
        question_only = q_text
        options_list  = options

        print(f" - 문제: {q_text[:50]}...")
        print(f" - 보기 개수: {len(options_list)}")

        # ✅ 변경: 두 벡터스토어를 명시적으로 전달
        final_state = agent.invoke(
            user_input_txt=user_input_txt,
            user_problem=question_only,
            user_problem_options=options_list,
            vectorstore_p=vectorstore_p,
            vectorstore_c=vectorstore_c,
            recursion_limit=1000,
        )

        gen_ans = final_state.get("generated_answer", "") or ""
        pred_idx = parse_idx_from_text(gen_ans)
        gen_text = options[pred_idx - 1] if (pred_idx and 1 <= pred_idx <= len(options)) else ""
        gen_exp = final_state.get("generated_explanation", "") or ""
        gen_sub = final_state.get("generated_subject", "") or ""
        print(f" - 예측 정답: {gen_ans}")
        print(f" - 예측 풀이: {gen_exp[:50]}...")

        # ✅ 변경: 유사문제 + 개념요약 컨텍스트 모두 사용
        retrieved_problems = final_state.get("retrieved_docs", []) or []
        retrieved_concepts = final_state.get("concept_contexts", []) or []
        ctx_texts = build_contexts(retrieved_problems, retrieved_concepts)
        print(f" - Retrieval Contexts: {len(ctx_texts)}개 (유사문제+개념요약)")

        q_full = f"{user_input_txt}\n\n" + build_question_with_options(q_text, options)

        gt_blob = f"정답: {gt_idx}) {gt_text}".strip()
        if gt_exp:
            gt_blob += f"\n풀이: {gt_exp}"
        if gt_sub:
            gt_blob += f"\n과목: {gt_sub}"

        rows.append(EvalRow(
            question=q_full,
            contexts=retrieved_problems,
            answer=f"정답: {gen_ans}) {gen_text}\n풀이: {gen_exp}\n과목: {gen_sub}".strip(),
            ground_truth=gt_blob,
            metadata={
                "options": options,
                "gt_answer_idx": gt_idx,
                "gt_sub": gt_sub,
                "pred_idx": pred_idx,
                "gen_sub": gen_sub,
                "validated": final_state.get("validated", False),
            }
        ))

        # ✅ 골든셋 행 누적 (문제/보기/정답/풀이/과목: GT 기준)
        golden_rows.append({
            "id": i,
            "question": q_text,
            "options": json.dumps(options, ensure_ascii=False),
            "gt_answer_idx": gt_idx,
            "gen_answer_idx": pred_idx,
            "gt_sub": gt_sub,
            "gen_subject": gen_sub,
            "gen_explanation": gen_exp,
            "retrieved_problems": [doc.page_content[:500] for doc in retrieved_problems],
            "retrieved_concepts": [doc.page_content[:500] for doc in retrieved_concepts],
            "contexts": ctx_texts
        })

        print(f"[{i}/{len(items)}] 완료 — GT:{gt_idx} / Pred:{pred_idx}")

    # 빈 rows 방지
    if not rows:
        print("[경고] 수집된 행이 없습니다. 저장 및 RAGAS 평가를 생략합니다.")
        return

    df = pd.DataFrame([{
        "question": r.question,
        "contexts": r.contexts,
        "answer": r.answer,
        "ground_truth": r.ground_truth,
        "metadata": r.metadata,
    } for r in rows])

    os.makedirs(out_dir, exist_ok=True)
    df_path = os.path.join(out_dir, "qa_eval_rows.parquet")
    df.to_parquet(df_path, index=False)
    print(f"[저장] QA 행 저장 완료 → {df_path}")

    # ✅ 골든셋 CSV 저장 (엑셀 호환을 위해 utf-8-sig)
    golden_df = pd.DataFrame(golden_rows)
    golden_csv = os.path.join(out_dir, "golden_set.csv")
    golden_df.to_csv(golden_csv, index=False, encoding="utf-8-sig")
    print(f"[저장] Golden set 저장 완료 → {golden_csv}")

    # ---------- RAGAS 평가 ----------
    try:
        print(f"[RAGAS] using llm={type(llm).__name__}, embeddings={type(emb).__name__}")
        df_eval = df[
            df["contexts"].apply(lambda x: isinstance(x, list) and len(x) > 0) &
            df["ground_truths"].apply(lambda x: isinstance(x, list) and any(bool(t.strip()) for t in x))
        ].copy()

        if df_eval.empty:
            print("[경고] 평가 가능한 행이 없습니다. (contexts 혹은 ground_truths 비어있음)")
            # 저장만 하고 종료
            df.to_parquet(df_path, index=False)
            golden_df.to_csv(golden_csv, index=False, encoding="utf-8-sig")
            return
        ds = Dataset.from_pandas(df_eval)
        ragas_result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=emb,
        )
        ragas_scores = pd.DataFrame(ragas_result.scores)
        print("[RAGAS 평가 완료]")
        print("RAGAS mean:", ragas_result)

    except Exception as e:
        print("[오류] RAGAS 평가 실패:", repr(e))
        # 빈 DataFrame으로라도 파일 생성
        ragas_scores = pd.DataFrame(columns=["faithfulness", "answer_relevancy", "context_precision", "context_recall"])

    ragas_csv = os.path.join(out_dir, "ragas_scores.csv")
    ragas_scores.to_csv(ragas_csv, index=False)
    print(f"[저장] RAGAS 점수 저장 완료 → {ragas_csv}")

    # ---------- 객관식 정확도(정답 인덱스 일치율) ----------
    def multiple_choice_accuracy(md_series: pd.Series) -> float | None:
        ok, tot = 0, 0
        for m in md_series:
            if not isinstance(m, dict):
                continue
            gt = m.get("gt_answer_idx")
            pr = m.get("pred_idx")   # ← 여기!
            if gt is None or pr is None:
                continue
            tot += 1
            ok += int(gt == pr)
        return ok / tot if tot else None

    mc_acc = multiple_choice_accuracy(df["metadata"])
    print(f"[정확도] 객관식 정답률: {mc_acc}")

    leaderboard = pd.concat([df.drop(columns=["contexts", "question", "answer", "ground_truth"], errors="ignore"),
                             ragas_scores], axis=1)
    leaderboard["mc_accuracy"] = mc_acc
    lb_csv = os.path.join(out_dir, "agent_eval_leaderboard.csv")
    leaderboard.to_csv(lb_csv, index=False)
    print(f"[저장] Leaderboard 저장 완료 → {lb_csv}")

    def make_plots_for_run(out_dir: str):
        ragas_csv = os.path.join(out_dir, "ragas_scores.csv")
        if not os.path.exists(ragas_csv):
            print(f"[시각화] {ragas_csv} 없음 → 스킵")
            return

        import pandas as pd
        df = pd.read_csv(ragas_csv)
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        for m in metrics:
            if m not in df.columns:
                df[m] = np.nan

        # (A) 평균 막대 그래프
        means = df[metrics].mean().fillna(0.0)
        plt.figure()
        plt.bar(np.arange(len(metrics)), means.values, width=0.6)
        plt.xticks(np.arange(len(metrics)), metrics, rotation=20, ha="right")
        plt.ylim(0, 1.0)
        plt.ylabel("Mean score")
        plt.title("RAGAS mean scores")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ragas_means_bar.png"), dpi=150)
        plt.close()

        # (B) 문항별 라인 그래프
        plt.figure()
        x = np.arange(1, len(df) + 1)
        for m in metrics:
            plt.plot(x, df[m].values, marker="o", linestyle="-", label=m)
        plt.ylim(0, 1.0)
        plt.xlabel("Question index")
        plt.ylabel("Score")
        plt.title("Per-question RAGAS scores")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_question_scores.png"), dpi=150)
        plt.close()

        # (C) 히트맵 (문항 × 지표)
        heat = df[metrics].to_numpy(dtype=float)
        plt.figure()
        im = plt.imshow(heat, aspect="auto", vmin=0.0, vmax=1.0)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.yticks(np.arange(len(df)), [str(i) for i in x])
        plt.xticks(np.arange(len(metrics)), metrics, rotation=20, ha="right")
        plt.title("RAGAS scores heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ragas_heatmap.png"), dpi=150)
        plt.close()

        print(f"[시각화] 저장 완료 → {out_dir}")

    # run_eval()의 마지막 줄 근처에 추가 호출
    make_plots_for_run(out_dir)


# ---------- 이미 생성된 리더보드가 있으면 건너뛰기 ----------
def already_done(out_dir: str) -> bool:
    lb_csv = os.path.join(out_dir, "agent_eval_leaderboard.csv")
    return os.path.exists(lb_csv)

if __name__ == "__main__":
    # 🔌 평가 시작 전 한 번만 두 컬렉션 연결
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

    vectorstore_p = connect_vectorstore("problems", MILVUS_HOST, MILVUS_PORT)
    vectorstore_c = connect_vectorstore("concepts", MILVUS_HOST, MILVUS_PORT)

    test_json_dir = "./teacher/exam/test_parsed_exam_json"
    json_files = glob.glob(os.path.join(test_json_dir, "*.json"))
    print(f"'{test_json_dir}' 폴더 내 {len(json_files)}개 JSON 파일 발견")
    out_dir_base = "./teacher/agents/solution/eval_results"

    skipped, done = 0, 0
    for json_file in json_files:
        print(f"\n=== '{json_file}' 처리 준비 ===")
        file_name = os.path.basename(json_file).replace(".json", "")
        out_dir = os.path.join(out_dir_base, file_name)

        if already_done(out_dir):
            print(f" - 이미 Leaderboard 존재 → 건너뜀 (out_dir: {out_dir})")
            skipped += 1
            continue

        print(f" - 실행 시작 (out_dir: {out_dir})")
        run_eval(
            test_json_path=json_file,
            out_dir=out_dir,
            vectorstore_p=vectorstore_p,
            vectorstore_c=vectorstore_c,
        )
        print(f"=== '{json_file}' 처리 완료 ===")
        done += 1

    print(f"\n요약: 실행 {done}개 / 건너뜀 {skipped}개")
