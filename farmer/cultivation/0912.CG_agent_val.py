import os
import sys
import types
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# RAGAS
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# 챗봇 실행 함수
from CG_agent_valv import run_agent

# -------------------------
# 환경 변수 세팅
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY 환경 변수가 없습니다. .env 확인하세요.")
    sys.exit()

# 평가용 LLM과 임베딩
ragas_llm = ChatOpenAI(model_name="gpt-4o-mini")
ragas_embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# -------------------------
# answer_relevancy 안전한 오버라이드
# -------------------------
# 목적: LLM에게 질문 생성시키는 기존 흐름 대신,
#       (골든셋) 질문(user_input) ↔ (모델) 답변(response) 임베딩 유사도로 계산.
async def gold_question_ascore(self, row, callbacks):
    """
    self: metric instance (answer_relevancy)
    row: dict with keys including 'user_input' (question) and 'response' (model answer)
    """
    # 안전하게 값 가져오기
    question = row.get("user_input") or row.get("question") or ""
    answer = row.get("response") or row.get("answer") or ""

    # 빈 경우 평가 불가
    if not question or not answer:
        # NaN으로 두면 나중에 분석 시 '측정 불가'로 식별 가능
        return float("nan")

    # embeddings가 설정되어 있어야 함 (evaluate 호출 시 ragas가 설정)
    emb = getattr(self, "embeddings", None)
    if emb is None:
        # 안전장치: evaluate가 embeddings를 전달하지 않았을 때 NaN 반환
        return float("nan")

    # 전처리: 문자열로 강제
    q_text = str(question).strip()
    a_text = str(answer).strip()

    try:
        # 임베딩 가져오기 — embed_query 사용 (단일 문장)
        q_vec = np.asarray(emb.embed_query(q_text)).reshape(1, -1)
        a_vec = np.asarray(emb.embed_query(a_text)).reshape(1, -1)

        q_norm = np.linalg.norm(q_vec)
        a_norm = np.linalg.norm(a_vec)
        if q_norm == 0 or a_norm == 0:
            return float("nan")

        cos = float(np.dot(q_vec, a_vec.T).reshape(-1)[0] / (q_norm * a_norm))
        # cos 범위 -1..1 -> 정규화 0..1
        cos_norm = (cos + 1.0) / 2.0

        # 디버깅 로그(필요하면 주석 해제)
        # print(f"[DEBUG answer_relevancy] q='{q_text[:60]}...' a='{a_text[:60]}...' cos={cos:.4f} norm={cos_norm:.4f}")

        return float(cos_norm)
    except Exception as e:
        # 디버깅 출력(필요 시 활성화)
        print("⚠️ answer_relevancy 계산 중 오류:", e)
        return float("nan")

# 오버라이드 적용
answer_relevancy._ascore = types.MethodType(gold_question_ascore, answer_relevancy)

# -------------------------
# 평가 실행 함수
# -------------------------
def run_ragas_with_csv(csv_path: str, save_path: str = "ragas_scores.csv"):
    golden_df = pd.read_csv(csv_path)

    # gt_contexts: 단일 텍스트 → 리스트로 변환
    golden_df["gt_contexts"] = golden_df["gt_contexts"].apply(lambda x: [x] if pd.notna(x) else [])

    results = []

    for i, row in golden_df.iterrows():
        question = row["question"]
        gold_answer = row["ground_truth"]
        gold_contexts = row["gt_contexts"]
        source_file = row.get("source_file", "")

        print(f"\n🔍 질문 {i+1}: {question}")

        try:
            chatbot_result = run_agent(question)
        except Exception as e:
            print(f"❌ 챗봇 실행 오류: {e}")
            continue

        answer = chatbot_result.get("answer")
        db_sources = chatbot_result.get("db_sources", [])
        web_sources = chatbot_result.get("web_sources", [])

        is_sufficient = chatbot_result.get("is_sufficient", "no")
        web_search_used = "no" if is_sufficient == "yes" else "yes"

        contexts = [src.get("content") for src in db_sources] + [src.get("content") for src in web_sources]

        if not answer or not contexts:
            print("⚠️ 답변 또는 컨텍스트 부족 → 평가 스킵")
            continue

        data = {
            "question": [question],
            "answer": [answer],
            "user_input": [question],   # 골든셋 질문 그대로 사용
            "response": [answer],
            "contexts": [contexts],
            "ground_truths": [gold_answer],
            "ground_truth_contexts": [" ".join(gold_contexts)],
            "reference": [" ".join(gold_contexts)],
        }
        dataset = Dataset.from_dict(data)

        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )
            df = result.to_pandas()

            # 원본 정보 추가
            
            df["source_file"] = source_file
            df["gold_answer"] = gold_answer
            df["web_search_used"] = web_search_used

            print(df[["faithfulness", "answer_relevancy", "context_recall", "context_precision", "web_search_used"]])

            results.append(df)

        except Exception as e:
            print(f"❌ 평가 오류: {e}")
            continue

    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 최종 평가 결과 {save_path} 저장 완료!")
        return final_df
    else:
        print("❌ 평가 결과가 없습니다.")
        return pd.DataFrame()

if __name__ == "__main__":
    print("🌱 골든셋 기반 RAGAS 평가 시작")
    run_ragas_with_csv("test_golden_set_llm_v2.csv", save_path="score(ragas)_v2.csv")
