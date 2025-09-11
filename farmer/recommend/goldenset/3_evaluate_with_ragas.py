# 3_evaluate_with_ragas.py (v1.9: JSON에 PASS/FAIL 추가 및 평균 계산)

import os
import json
import logging
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import torch
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import ast

# ==================== 설정: 입력 파일 이름 ====================
INPUT_CSV_FILENAME = "2_rag_answers_20250911_110105.csv" # <-- 여기를 수정하세요
# ==========================================================

# ==================== 설정 (공통) ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("evaluate_ragas")
load_dotenv(find_dotenv())

# 환경 변수
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY가 .env에 필요합니다.")

# 전역 객체
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"임베딩 모델을 위한 장치로 '{device}'를 사용합니다.")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={"device": device})
llm_eval = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)

# 빈 값을 안전하게 처리하는 함수
def safe_literal_eval(val):
    """NaN이나 잘못된 형식의 문자열을 안전하게 처리하여 리스트로 변환합니다."""
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

# ==================== 메인 실행 로직 ====================
def main():
    logger.info(f"'{INPUT_CSV_FILENAME}' 파일에서 평가 데이터를 불러옵니다.")
    try:
        df = pd.read_csv(INPUT_CSV_FILENAME)
    except FileNotFoundError:
        logger.error(f"'{INPUT_CSV_FILENAME}' 파일을 찾을 수 없습니다.")
        logger.error("스크립트 상단의 파일 이름이 정확한지, 파일이 스크립트와 같은 폴더에 있는지 확인하세요.")
        return

    logger.info("'contexts' 컬럼을 리스트 형식으로 변환합니다...")
    df['contexts'] = df['contexts'].apply(safe_literal_eval)
    if 'gt_contexts' in df.columns:
        df['gt_contexts'] = df['gt_contexts'].apply(safe_literal_eval)

    # RAGAS Dataset 형식으로 변환
    dataset_dict = {
        'question': df['question'].tolist(),
        'answer': df['answer'].tolist(),
        'contexts': df['contexts'].tolist(),
        'ground_truth': df['ground_truth'].tolist()
    }
    dataset = Dataset.from_dict(dataset_dict)

    logger.info("RAGAS 평가를 시작합니다...")
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm_eval, embeddings=embedding_model)
    logger.info("RAGAS 평가가 완료되었습니다.")

    # RAGAS 결과를 원본 데이터프레임에 병합
    results_df = result.to_pandas()
    df['faithfulness'] = results_df['faithfulness']
    df['answer_relevancy'] = results_df['answer_relevancy']
    df['context_recall'] = results_df['context_recall']
    df['context_precision'] = results_df['context_precision']

    # ✨ 평균 점수 계산
    overall_avg = results_df.mean(numeric_only=True).to_dict()
    
    # ✨ JSON에 추가될 최종 데이터 구조 생성
    json_output = {
        "overall_average": {
            "faithfulness": overall_avg.get('faithfulness', 0.0),
            "answer_relevancy": overall_avg.get('answer_relevancy', 0.0),
            "context_recall": overall_avg.get('context_recall', 0.0),
            "context_precision": overall_avg.get('context_precision', 0.0),
        },
        "questions": []
    }
    
    # PASS/FAIL 판단 기준 설정
    PASS_THRESHOLD = 0.7

    # 각 질문에 대한 개별 결과를 리스트에 추가
    for _, row in df.iterrows():
        is_pass = (row['answer_relevancy'] >= PASS_THRESHOLD) and (row['context_precision'] >= PASS_THRESHOLD)
        
        # ✨ 개별 질문의 RAGAS 지표 평균 계산
        individual_avg = (
            row['faithfulness'] + 
            row['answer_relevancy'] + 
            row['context_recall'] + 
            row['context_precision']
        ) / 4
        
        item = {
            "question": row['question'],
            "answer": row['answer'],
            "ragas_metrics": {
                "faithfulness": row['faithfulness'],
                "answer_relevancy": row['answer_relevancy'],
                "context_recall": row['context_recall'],
                "context_precision": row['context_precision'],
                "average": individual_avg  # ✨ 개별 평균 점수 추가
            },
            "pass_fail": "PASS" if is_pass else "FAIL"
        }
        json_output['questions'].append(item)

    # 이 부분이 for 루프 밖으로 이동했습니다.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"ragas_evaluation_{timestamp}.json"
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)
        
    logger.info(f"RAGAS 평가 결과가 '{json_filename}' 파일로 저장되었습니다.")
    
    # RAGAS 평가 요약 출력 부분
    print("\n" + "=" * 30 + "\nRAGAS 평가 요약\n" + "=" * 30)
    print("전체 질문에 대한 평균 값\n" + "=" * 30)
    
    print(f"- faithfulness: {json_output['overall_average']['faithfulness']:.4f}")
    print(f"- answer_relevancy: {json_output['overall_average']['answer_relevancy']:.4f}")
    print(f"- context_recall: {json_output['overall_average']['context_recall']:.4f}")
    print(f"- context_precision: {json_output['overall_average']['context_precision']:.4f}")
    print("=" * 30)

if __name__ == "__main__":
    main()