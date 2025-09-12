import os
import sys
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
import types

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
# answer_relevancy 수정
# -------------------------
async def gold_question_ascore(self, row, callbacks):
    # 골든셋 질문을 그대로 사용, noncommittal=0
    responses = [type('Response', (), {'question': row["user_input"], 'noncommittal': 0})()]
    return self._calculate_score(responses, row)

# _ascore 덮어쓰기
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
        source_pdf = row["source_pdf"]

        print(f"\n🔍 질문 {i+1}: {question}")

        try:
            chatbot_result = run_agent(question)
        except Exception as e:
            print(f"❌ 챗봇 실행 오류: {e}")
            continue

        answer = chatbot_result.get("answer")
        db_sources = chatbot_result.get("db_sources", [])
        web_sources = chatbot_result.get("web_sources", [])
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
            df["question"] = question
            df["source_pdf"] = source_pdf
            df["gold_answer"] = gold_answer
            df["model_answer"] = answer

            print(
                df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]]
            )

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
    run_ragas_with_csv("golden_set_20250911_094918.csv", save_path="ragas_scores1.csv")
