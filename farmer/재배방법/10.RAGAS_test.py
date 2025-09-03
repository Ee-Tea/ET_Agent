# API 키 설정
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# Groq와 Hugging Face 모델을 임포트합니다.
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
# Groq API 키를 환경 변수에서 불러옵니다.
# OPENAI_API_KEY 변수명 대신 GROQ_API_KEY를 사용합니다.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY2 = os.getenv("GROQ_API_KEY2")


# 1. PDF 파일 로드하기 (진행률 포함)
print("문서를 로드 중입니다...")
directory_path = './data/cropinfo'
pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]

documents = []
for file_path in tqdm(pdf_files, desc="PDF 파일 로딩"):
    try:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    except Exception as e:
        print(f"\n오류 발생: {file_path} 파일을 처리하는 중 오류가 발생했습니다 - {e}")

print(f"\n총 {len(documents)}개의 문서를 로드했습니다.")
print("---")

# 2. RAGAS 테스트셋 생성기 설정 (Groq LLM & Hugging Face 임베딩 사용)
# generator_llm과 critic_llm에 Groq의 초고속 모델을 사용합니다.
generator_llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.7, groq_api_key=GROQ_API_KEY)
critic_llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.7, groq_api_key=GROQ_API_KEY2)

# Hugging Face 임베딩 모델을 사용합니다.
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# 3. 전체 문서에서 골든 데이터셋 생성
TARGET_QUESTIONS = 30

print(f"\n총 {TARGET_QUESTIONS}개의 질문을 생성 중입니다... (이 단계에서는 진행률이 표시되지 않습니다)")
try:
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