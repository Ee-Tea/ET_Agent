# validation.py (새 파일)

import os
import sys
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

# RAGAS 라이브러리
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    ContextUtilization,
    LLMContextPrecisionWithoutReference,
)
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# 챗봇 파일에서 핵심 함수를 import합니다.
from CG_agent_valv import create_retriever, run_agent

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    sys.exit()

# RAGAS 평가를 위한 LLM 및 임베딩 모델 정의
ragas_llm = ChatOpenAI(model_name="gpt-4o-mini")
ragas_embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

def run_ragas_evaluation(question: str, answer: str, contexts: List[str]):
    """
    주어진 질문, 답변, 맥락으로 RAGAS 평가를 실행하고 결과를 출력합니다.
    """
    print("\n--- 🤖 RAGAS 자동 평가 시작 ---")
    
    data = {
        'question': [question],
        'answer': [answer],
        'contexts': [contexts]
    }
    dataset = Dataset.from_dict(data)

    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
        ContextUtilization(),
        LLMContextPrecisionWithoutReference()
    ]

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_evaluate,
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        print("✅ RAGAS 평가 완료.")
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        result_df = result.to_pandas()
        
        print("\n--- 📊 RAGAS 평가 점수 ---")
        if 'faithfulness' in result_df.columns:
            print(f"충실성 (faithfulness): {result_df['faithfulness'].iloc[0]:.4f}")
        if 'answer_relevancy' in result_df.columns:
            print(f"답변 관련성 (answer_relevancy): {result_df['answer_relevancy'].iloc[0]:.4f}")
        if 'context_utilization' in result_df.columns:
            print(f"맥락 활용도 (context_utilization): {result_df['context_utilization'].iloc[0]:.4f}")
        if 'llm_context_precision_without_reference' in result_df.columns:
            print(f"맥락 정확성 (context_precision): {result_df['llm_context_precision_without_reference'].iloc[0]:.4f}")
        
        print("\n--- 전체 평가 데이터프레임 ---")
        print(result_df)
        
        return result_df

    except Exception as e:
        print(f"❌ RAGAS 평가 중 오류가 발생했습니다: {e}")
        return pd.DataFrame({'faithfulness': [0.0], 'answer_relevancy': [0.0], 'context_utilization': [0.0], 'context_precision': [0.0]})

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    print("🌱 챗봇 검증 시스템 시작...")
    
    # 챗봇 에이전트 준비
    print("챗봇 시스템을 준비하는 중입니다...")
    try:
        retriever = create_retriever()
    except Exception as e:
        print(f"오류: 챗봇 시스템을 초기화할 수 없습니다. {e}")
        sys.exit()

    print("챗봇 시스템 준비 완료!\n")
    
    while True:
        prompt = input("검증할 질문을 입력하세요 (종료하려면 'exit' 또는 'quit' 입력): ")
        if prompt.lower() in ["exit", "quit"]:
            print("검증 시스템을 종료합니다.")
            break
        
        print("\n🔍 챗봇 답변을 생성하는 중...")
        
        # CG_agent_valv.py의 함수를 호출하여 답변 및 컨텍스트를 가져옴
        try:
            chatbot_result = run_agent(prompt, retriever)
        except Exception as e:
            print(f"챗봇 답변 생성 중 오류가 발생했습니다: {e}")
            continue

        answer = chatbot_result.get('answer')
        db_sources = chatbot_result.get('db_sources', [])
        web_sources = chatbot_result.get('web_sources', [])
        
        # 답변 생성에 사용된 모든 컨텍스트를 하나의 리스트로 통합
        contexts = [src.get('content') for src in db_sources] + \
                   [src.get('content') for src in web_sources]
        
        if answer and contexts:
            print("✅ 답변 생성 완료. RAGAS 평가를 시작합니다.")
            run_ragas_evaluation(
                question=prompt,
                answer=answer,
                contexts=contexts
            )
        else:
            print("❗ 답변 또는 컨텍스트가 부족하여 RAGAS 평가를 수행할 수 없습니다.")
            if answer:
                print("답변:", answer)
            
        print("\n-------------------------------------------\n")