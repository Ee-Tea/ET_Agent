# API 키 설정
import os
import pandas as pd
import sys
import asyncio
from dotenv import load_dotenv
from tqdm import tqdm
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# OpenAI와 Hugging Face 모델을 임포트합니다.
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# Windows 환경에서 asyncio 정책을 설정합니다.
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
# OpenAI API 키를 환경 변수에서 불러옵니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("오류: 환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

# 메인 비동기 함수
async def main():
    """RAGAS 테스트셋 생성 과정을 관리하는 메인 함수입니다."""
    
    # 1. PDF 파일 로드하기 (진행률 포함)
    print("문서를 로드 중입니다...")
    directory_path = './data/cropinfo/cropinfo2'
    if not os.path.exists(directory_path):
        print(f"오류: 디렉터리 '{directory_path}'를 찾을 수 없습니다.")
        return
        
    pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]

    documents = []
    for file_path in tqdm(pdf_files, desc="PDF 파일 로딩"):
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"\n오류 발생: {file_path} 파일을 처리하는 중 오류가 발생했습니다 - {e}")
            
    if not documents:
        print("\n로드된 문서가 없습니다. 작업을 종료합니다.")
        return

    print(f"\n총 {len(documents)}개의 문서를 로드했습니다.")
    print("---")

    # 2. RAGAS 테스트셋 생성기 설정
    generator_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    critic_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

    # Hugging Face 임베딩 모델 이름을 'multitask'로 수정했습니다.
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    # 3. 전체 문서에서 골든 데이터셋 생성
    TARGET_QUESTIONS = 10
    print(f"\n총 {TARGET_QUESTIONS}개의 질문을 생성 중입니다...")
    try:
        # 이 함수 자체는 비동기가 아니므로 await를 사용하지 않습니다.
        # 하지만, 이 함수가 내부적으로 생성하는 비동기 작업들을
        # asyncio.run()이 관리하도록 main 함수에 포함시킵니다.
        testset = generator.generate_with_langchain_docs(
            documents,
            test_size=TARGET_QUESTIONS,
            distributions={
                simple: 0.5,
                reasoning: 0.25,
                multi_context: 0.25
            }
        )

        # 4. 생성된 데이터셋을 Pandas DataFrame으로 확인 및 저장
        df = testset.to_pandas()
        print("---")
        print("생성된 골든 데이터셋 (일부):")
        print(df.head())

        df.to_csv("golden_dataset.csv", index=False, encoding="utf-8-sig")
        print(f"\n총 {len(df)}개의 질문이 golden_dataset.csv 파일로 저장되었습니다.")

    except Exception as e:
        print(f"데이터셋 생성 중 오류가 발생했습니다: {e}")

# 스크립트 실행
if __name__ == "__main__":
    asyncio.run(main())
