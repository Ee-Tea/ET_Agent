# ragas_repair.py
import os
import json
import argparse
from typing import List, Dict, Any
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def read_df_from_outdir(out_dir: str) -> pd.DataFrame:
    """qa_eval_rows.(parquet|csv) 읽기 (parquet 실패 시 csv 폴백)"""
    pq = os.path.join(out_dir, "qa_eval_rows.parquet")
    csv = os.path.join(out_dir, "qa_eval_rows.csv")
    if os.path.exists(pq):
        try:
            return pd.read_parquet(pq)
        except Exception as e:
            print(f"[WARN] parquet 읽기 실패({e}) → CSV 시도")
    if os.path.exists(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError(f"[ERR] 원본 QA 데이터가 없습니다: {pq} / {csv}")


def ensure_metadata_dict(df: pd.DataFrame) -> pd.DataFrame:
    """CSV로 저장된 경우 metadata가 str일 수 있어 dict로 복원"""
    if "metadata" not in df.columns:
        raise KeyError("[ERR] DataFrame에 'metadata' 컬럼이 없습니다.")
    def to_dict(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    return json.loads(s.replace("'", '"'))
                except Exception:
                    # single quote json fallback
                    try:
                        return json.loads(s)
                    except Exception:
                        return {}
        return {}
    df = df.copy()
    df["metadata"] = df["metadata"].apply(to_dict)
    return df


def multiple_choice_accuracy(md_series: pd.Series) -> float | None:
    ok, tot = 0, 0
    for m in md_series:
        if not isinstance(m, dict):
            continue
        gt = m.get("gt_answer_idx")
        pr = m.get("pred_answer_idx")
        if gt is None or pr is None:
            continue
        tot += 1
        ok += int(gt == pr)
    return ok / tot if tot else None


def has_valid_ragas(out_dir: str) -> bool:
    """ragas_scores.csv가 존재하고, 지표 중 하나라도 NaN이 아닌 값이 있는지 확인"""
    path = os.path.join(out_dir, "ragas_scores.csv")
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty:
        return False
    cols = [c for c in ["faithfulness","answer_relevancy","context_precision","context_recall"] if c in df.columns]
    if not cols:
        return False
    return df[cols].notna().sum().sum() > 0


def build_llm(model_name: str | None, api_key: str | None):
    """사전 테스트로 접근 가능한 ChatOpenAI 인스턴스 생성. 실패 시 None 반환."""
    if not api_key:
        print("[LLM] OPENAI_API_KEY 미설정 → LLM 지표 스킵")
        return None
    candidates = [m for m in [model_name, os.getenv("RAGAS_EVAL_MODEL"), "gpt-4o-mini", "gpt-4o"] if m]
    last_err = None
    for m in candidates:
        try:
            llm = ChatOpenAI(model=m, temperature=0, api_key=api_key)
            _ = llm.invoke("ping")  # 권한/엔드포인트 사전 확인
            print(f"[LLM] 사용 모델 확정: {m}")
            return llm
        except Exception as e:
            print(f"[LLM] 모델 '{m}' 사용 불가 → {e}")
            last_err = e
    print("[LLM] 사용 가능한 모델 없음 → LLM 지표 스킵")
    if last_err:
        print(f"[LLM] 마지막 오류: {repr(last_err)}")
    return None


def build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"}
    )


def recompute_ragas_and_leaderboard(out_dir: str, llm, emb) -> None:
    print(f"\n[REPAIR] {out_dir} 재계산 시작")
    df = read_df_from_outdir(out_dir)
    df = ensure_metadata_dict(df)

    ds = Dataset.from_pandas(df)

    # 어떤 지표를 돌릴지 결정 (LLM 유무에 따라)
    if llm is None:
        metrics_to_run = [context_precision, context_recall]
        print("[RAGAS] LLM 없음 → 임베딩 기반 지표만 계산(context_precision, context_recall)")
    else:
        metrics_to_run = [faithfulness, answer_relevancy, context_precision, context_recall]
        print(f"[RAGAS] using llm={type(llm).__name__}, embeddings={type(emb).__name__}")

    # 평가
    try:
        ragas_result = evaluate(
            ds,
            metrics=metrics_to_run,
            llm=llm if llm is not None else None,
            embeddings=emb,
        )
        ragas_scores = pd.DataFrame(ragas_result.scores)
    except Exception as e:
        print("[RAGAS] 평가 실패:", repr(e))
        ragas_scores = pd.DataFrame(columns=["faithfulness","answer_relevancy","context_precision","context_recall"])

    # 누락 컬럼 보정 및 정렬
    for col in ["faithfulness","answer_relevancy","context_precision","context_recall"]:
        if col not in ragas_scores.columns:
            ragas_scores[col] = pd.NA
    ragas_scores = ragas_scores[["faithfulness","answer_relevancy","context_precision","context_recall"]]

    # 저장
    ragas_csv = os.path.join(out_dir, "ragas_scores.csv")
    ragas_scores.to_csv(ragas_csv, index=False)
    print(f"[REPAIR] ragas_scores.csv 저장 → {ragas_csv}")

    # 리더보드 재작성
    mc_acc = multiple_choice_accuracy(df["metadata"])
    leaderboard = pd.concat(
        [df.drop(columns=["contexts","question","answer","ground_truth"], errors="ignore"), ragas_scores],
        axis=1
    )
    leaderboard["mc_accuracy"] = mc_acc
    lb_csv = os.path.join(out_dir, "agent_eval_leaderboard.csv")
    leaderboard.to_csv(lb_csv, index=False)
    print(f"[REPAIR] agent_eval_leaderboard.csv 저장 → {lb_csv}")


def find_eval_dirs(root: str) -> List[str]:
    """루트 아래에서 qa_eval_rows 파일이 있는 디렉터리 목록 수집"""
    out_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "qa_eval_rows.parquet" in filenames or "qa_eval_rows.csv" in filenames:
            out_dirs.append(dirpath)
    return sorted(out_dirs)


def main():
    parser = argparse.ArgumentParser(description="RAGAS 점수/리더보드 재계산 도구")
    parser.add_argument("--out_dir", nargs="+", help="대상 결과 디렉터리(여러 개 가능)")
    parser.add_argument("--root", help="루트 폴더를 주면 하위에서 자동 검색")
    parser.add_argument("--force", action="store_true", help="ragas가 있어도 강제로 재계산")
    parser.add_argument("--model", help="LLM 모델명(미지정시 RAGAS_EVAL_MODEL→gpt-4.1-mini 순서로 시도)")
    parser.add_argument("--emb_model", default="jhgan/ko-sroberta-multitask", help="HuggingFace 임베딩 모델명")
    args = parser.parse_args()

    # 대상 디렉터리 수집
    targets: List[str] = []
    if args.out_dir:
        targets.extend(args.out_dir)
    if args.root:
        targets.extend(find_eval_dirs(args.root))
    if not targets:
        print("[ERR] 대상 디렉터리가 지정되지 않았습니다. --out_dir 또는 --root 사용")
        return

    # 리소스 준비
    openai_key = os.environ.get("OPENAI_API_KEY")
    llm = build_llm(args.model, openai_key)
    emb = build_embeddings(args.emb_model)

    # 처리
    done, skipped = 0, 0
    for out_dir in targets:
        print(f"\n=== 대상: {out_dir} ===")
        if (not args.force) and has_valid_ragas(out_dir):
            print(" - 유효한 ragas_scores.csv 존재 → 건너뜀 ( --force 로 강제 가능 )")
            skipped += 1
            continue
        recompute_ragas_and_leaderboard(out_dir, llm, emb)
        done += 1

    print(f"\n요약: 재계산 {done}개 / 스킵 {skipped}개")


if __name__ == "__main__":
    main()
