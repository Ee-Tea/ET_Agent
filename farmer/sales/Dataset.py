# API 키 설정
import os
import pandas as pd
import sys
import asyncio
from dotenv import load_dotenv
from tqdm import tqdm
from ragas.testset import TestsetGenerator

# OpenAI와 Hugging Face 모델을 임포트합니다.
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.schema import Document

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
    
    # 1. CSV 파일 로드하기 (커스텀 방식)
    print("CSV 파일을 로드 중입니다...")
    csv_file_path = './farmer/sales/data/20240812.csv'
    if not os.path.exists(csv_file_path):
        print(f"오류: CSV 파일 '{csv_file_path}'를 찾을 수 없습니다.")
        return

    documents = []
    try:
        # pandas로 CSV 파일을 직접 읽기
        df = pd.read_csv(csv_file_path, encoding='euc-kr')
        print(f"CSV 파일에서 {len(df)}개의 행을 읽었습니다.")
        print(f"컬럼: {list(df.columns)}")
        
        # 각 행마다 개별 문서 생성 (각 행마다 질문 하나씩)
        df['품목'] = df['품목'].fillna("정보 없음")
        
        for index, row in df.iterrows():
            # 각 행을 개별 문서로 생성 (100토큰 이상의 긴 문서)
            content = f"""
{row['판매장 이름']}은 {row['주소']}에 위치한 농산물 판매장입니다. 
이 판매장의 주요 판매 품목은 {row['품목']}이며, 지역 농산물 유통의 중요한 역할을 담당하고 있습니다.

판매장 정보:
- 상호명: {row['판매장 이름']}
- 위치: {row['주소']}
- 취급 품목: {row['품목']}
- 업종: 농산물 유통 및 판매

이 판매장은 지역 농업 생산자와 소비자를 연결하는 중요한 유통 채널로서, 신선한 농산물을 제공하고 있습니다. 
고객들은 이곳에서 신선하고 품질 좋은 농산물을 구매할 수 있으며, 지역 경제 활성화에도 기여하고 있습니다.

판매장 운영 정보:
- 운영 방식: 직접 판매 및 유통
- 주요 고객: 지역 주민 및 상업 구매자
- 특색: 신선한 농산물 중심의 판매
- 지역 기여도: 농업 생산자 지원 및 지역 경제 활성화
            """.strip()
            
            # Document 객체 생성
            doc = Document(
                page_content=content,
                metadata={
                    "source": csv_file_path,
                    "row_index": index,
                    "판매장_이름": row['판매장 이름'],
                    "주소": row['주소'],
                    "품목": row['품목']
                }
            )
            documents.append(doc)
            
        print(f"총 {len(documents)}개의 문서를 생성했습니다.")
        
    except Exception as e:
        print(f"\n오류 발생: CSV 파일을 처리하는 중 오류가 발생했습니다 - {e}")
        return
            
    if not documents:
        print("\n로드된 문서가 없습니다. 작업을 종료합니다.")
        return

    print(f"\n총 {len(documents)}개의 문서를 로드했습니다.")
    print("---")

    # 2. RAGAS 테스트셋 생성기 설정 (공식 문서 방식)
    # LangChain 모델들 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    # TestsetGenerator 초기화 (공식 문서의 from_langchain 방식)
    generator = TestsetGenerator.from_langchain(        
        llm=llm,
        embedding_model=embedding_model
    )

    # 3. 전체 문서에서 골든 데이터셋 생성 (공식 문서 방식)
    TARGET_QUESTIONS = 3  # 3개만 생성
    print(f"\n총 {TARGET_QUESTIONS}개의 질문을 생성 중입니다...")
    try:
        # 공식 문서에 따른 테스트셋 생성
        testset = generator.generate_with_langchain_docs(
            documents=documents, 
            testset_size=TARGET_QUESTIONS,
            with_debugging_logs=True  # 디버깅 로그 활성화 (오류 파악용)
        )

        # 4. 생성된 데이터셋을 Pandas DataFrame으로 확인 및 저장
        df = testset.to_pandas()
        print("---")
        print("생성된 골든 데이터셋 (일부):")
        print(df.head())

        df.to_csv("golden_dataset.csv", index=False, encoding="euc-kr")
        print(f"\n총 {len(df)}개의 질문이 golden_dataset.csv 파일로 저장되었습니다.")

    except Exception as e:
        print(f"데이터셋 생성 중 오류가 발생했습니다: {e}")

# 스크립트 실행
if __name__ == "__main__":
    asyncio.run(main())