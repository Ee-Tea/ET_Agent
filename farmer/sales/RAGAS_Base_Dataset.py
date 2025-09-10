# API 키 설정
import os
import pandas as pd
import sys
import asyncio
from dotenv import load_dotenv
from tqdm import tqdm
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper # LangchainEmbeddingsWrapper 다시 사용
from langchain_community.embeddings import HuggingFaceEmbeddings # langchain_community 임베딩 모델 임포트
from ragas.testset.persona import Persona

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
        print(f"컬럼: {df.columns.tolist()}")

        # 모든 컬럼을 문자열로 강제 변환 (Pydantic 오류 방지)
        df = df.astype(str)
        
        # 성능 최적화: 샘플링으로 문서 수 대폭 줄이기
        df['품목'] = df['품목'].fillna("정보 없음")
        
        # 242개 중 3개만 샘플링 (성능 향상)
        sample_size = min(3, len(df))
        sampled_df = df.sample(n=sample_size, random_state=42)
        print(f"성능 최적화: {len(df)}개 중 {sample_size}개 샘플링")
        
        for index, row in sampled_df.iterrows():
            # 완전히 안전한 형태로 한국어 텍스트 생성
            store_name = str(row['판매장 이름']).replace('(', '').replace(')', '').replace(',', ' ').strip()
            address = str(row['주소']).replace('(', '').replace(')', '').replace(',', ' ').strip()
            products = str(row['품목']).replace('(', '').replace(')', '').replace(',', ' ').strip()

            # 빈 문자열 처리 (NaN 값 방지)
            store_name = store_name if store_name and store_name.lower() != 'nan' else "알 수 없는 판매장"
            address = address if address and address.lower() != 'nan' else "알 수 없는 주소"
            products = products if products and products.lower() != 'nan' else "알 수 없는 품목"

            print(f"DEBUG: store_name={type(store_name)} {store_name}")
            print(f"DEBUG: address={type(address)} {address}")
            print(f"DEBUG: products={type(products)} {products}")

            # content = f"판매장명 {store_name} 위치 {address} 판매품목 {products} 업종 농산물유통. {store_name}은 {address}에 위치한 농산물 판매장으로 {products}를 판매합니다."
            # content = store_name # page_content를 판매장 이름으로 극단적으로 단순화

            # 농가 관점에서 농작물 판매처 정보를 찾는 내용으로 재구성
            content = (
                f"농산물 판매처 정보: {store_name}. "
                f"판매처 유형: 농산물유통업체. "
                f"판매처 위치: {address}. "
                f"수취하는 농산물 품목: {products}. "
                f"이 판매처는 농가에서 생산한 농산물을 직접 수취하여 판매하는 유통업체입니다."
                f"농가에서는 이 판매처를 통해 자신이 재배한 농작물을 안정적으로 판매할 수 있으며,"
                f"직거래를 통해 중간 유통업체를 거치지 않고 더 높은 수익을 얻을 수 있습니다. "
                f"판매처에서는 신선하고 품질 좋은 농산물을 선호하며,"
                f"농가와의 지속적인 거래 관계를 중시합니다. "
                f"농가에서는 이 판매처에 연락하여 농산물 공급 조건, 가격, 수취 시기 등을 문의할 수 있습니다."
                f"농산물 판매를 원하는 농가에게는 좋은 판로 확보 기회가 될 수 있습니다."
            )

            documents.append(Document(page_content=content, metadata={"store_name": store_name, "address": address, "products": products}))
            
        print(f"총 {len(documents)}개의 문서를 생성했습니다.")
        
    except Exception as e:
        print(f"\n오류 발생: CSV 파일을 처리하는 중 오류가 발생했습니다 - {e}")
        return
            
    if not documents:
        print("\n로드된 문서가 없습니다. 작업을 종료합니다.")
        return

    print(f"\n총 {len(documents)}개의 문서를 로드했습니다.")
    print("---")

    # 2. 농가 페르소나 정의
    FARMER_PERSONA = Persona(
        name = "Beginner Farming",
        role_description = "농사를 시작한지 얼마되지 않은 농가 주인으로, 자신이 키운 농작물을 팔 수 있는 판매처를 찾고 있습니다. 영어를 못하고 한국어만을 사용합니다."
    )
    personas = [FARMER_PERSONA]
    
    # 3. RAGAS 테스트셋 생성기 설정 (농가 페르소나 적용)
    # LLM을 LangchainLLMWrapper로 감싸기
    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.8
        )
    )
    
    # 임베딩 모델을 LangchainEmbeddingsWrapper로 감싸기
    generator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", model_kwargs={'device': 'cpu'})
    )

    # TestsetGenerator 초기화 (공식 문서 방식)
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas
    )

    # 3. RAGAS를 통한 골든 데이터셋 생성
    TARGET_QUESTIONS = 3
    print(f"\n총 {TARGET_QUESTIONS}개의 질문을 RAGAS로 생성 중입니다...")
    try:
        
        # 공식 문서에 따른 테스트셋 생성
        testset = generator.generate_with_langchain_docs(
            documents=documents, 
            testset_size=TARGET_QUESTIONS
        )

        # 4. 생성된 데이터셋을 Pandas DataFrame으로 확인 및 저장
        df = testset.to_pandas()
        
        # 요청한 개수만큼만 사용 (3개로 제한)
        if len(df) > TARGET_QUESTIONS:
            df = df.head(TARGET_QUESTIONS)
            print(f"요청한 {TARGET_QUESTIONS}개로 제한했습니다.")
        
        print("---")
        print("생성된 골든 데이터셋:")
        print(df.head())

        df.to_csv("golden_dataset.csv", index=False, encoding="utf-8-sig")
        print(f"\n총 {len(df)}개의 질문이 golden_dataset.csv 파일로 저장되었습니다.")

    except Exception as e:
        print(f"데이터셋 생성 중 오류가 발생했습니다: {e}")

# 스크립트 실행
if __name__ == "__main__":
    asyncio.run(main())