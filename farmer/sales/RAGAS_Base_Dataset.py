# API 키 설정
import os
import pandas as pd
import sys
import asyncio
import requests
import random
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

# OpenAI와 Hugging Face 모델을 임포트합니다.
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.schema import Document

def import_ragas_modules():
    """RAGAS 관련 모듈들을 지연 로딩으로 import"""
    try:
        from ragas.testset import TestsetGenerator
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.testset.persona import Persona
        from ragas.testset.transforms.extractors.llm_based import NERExtractor
        from ragas.testset.transforms.splitters import HeadlineSplitter
        from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
        
        return {
            'TestsetGenerator': TestsetGenerator,
            'LangchainLLMWrapper': LangchainLLMWrapper,
            'LangchainEmbeddingsWrapper': LangchainEmbeddingsWrapper,
            'Persona': Persona,
            'NERExtractor': NERExtractor,
            'HeadlineSplitter': HeadlineSplitter,
            'SingleHopSpecificQuerySynthesizer': SingleHopSpecificQuerySynthesizer
        }
    except ImportError as e:
        print(f"❌ RAGAS 모듈 import 실패: {e}")
        raise

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# Windows 환경에서 asyncio 정책을 설정합니다.
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
# API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("오류: 환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

# KAMIS API 키 설정
api_key = os.getenv("KAMIS_API_KEY")
api_id = os.getenv("KAMIS_ID")
if not api_key or not api_id:
    print("오류: KAMIS API 키가 설정되지 않았습니다.")
    sys.exit(1)

# RAGAS 설정
TARGET_QUESTIONS = 50  # RAGAS에서 생성할 질문 개수

def fetch_api_data():
    """KAMIS API에서 농산물 가격 정보를 가져오는 함수입니다."""
    url = "http://www.kamis.or.kr/service/price/xml.do?action=dailySalesList"
    params = {
        "p_cert_key": api_key,
        "p_cert_id": api_id,
        "p_returntype": "json"
    }
    response = requests.get(url, params=params)
    docs = []
    
    if response.status_code == 200:
        data = response.json()
        items = []
        price = data.get("price", {})
        if isinstance(price, dict):
            items = price.get("item", [])
        elif isinstance(price, list):
            items = price
        if isinstance(items, dict):
            items = [items]

        def safe_val(val):
            if isinstance(val, list):
                return val[0] if val else ""
            return val if val is not None else ""

        for item in items:
                
            category = safe_val(item.get('category_name', ''))
            if category not in ['수산물', '축산물'] and safe_val(item.get('product_cls_name', '')) != '소매':
                direction_raw = safe_val(item.get('direction', ''))
                value_raw = safe_val(item.get('value', ''))
                dpr1 = safe_val(item.get('dpr1', ''))
                dpr2 = safe_val(item.get('dpr2', ''))
                day3 = safe_val(item.get('day3', ''))
                dpr3 = safe_val(item.get('dpr3', ''))
                day4 = safe_val(item.get('day4', ''))
                dpr4 = safe_val(item.get('dpr4', ''))

                try:
                    dpr1_val = int(str(dpr1).replace(',', '').replace(' ', '') or '0')
                    dpr2_val = int(str(dpr2).replace(',', '').replace(' ', '') or '0')
                    diff = abs(dpr1_val - dpr2_val)
                except (ValueError, TypeError):
                    diff = 0
                
                change_str = "와 변동 없는"
                if str(direction_raw) == "0":
                    change_str = f"보다 {value_raw}%({diff}원) 감소한"
                elif str(direction_raw) == "1":
                    change_str = f"보다 {value_raw}%({diff}원) 증가한"
                
                doc = (
                    f"{safe_val(item.get('item_name', ''))} ({safe_val(item.get('unit', ''))})의 가격은 어제"
                    f"{change_str} {dpr1}원 입니다."
                )
                if dpr3 and str(dpr3).strip() != "" and str(dpr3).strip() != "원":
                    doc += f" {day3}에는 {dpr3}원,"
                if dpr4 and str(dpr4).strip() != "" and str(dpr4).strip() != "원":
                    doc += f" {day4}에는 {dpr4}원 이었습니다."
                docs.append(doc)
    else:
        print("API 호출 실패:", response.status_code)

    return docs

# 메인 비동기 함수
async def main():
    """RAGAS 테스트셋 생성 과정을 관리하는 메인 함수입니다."""
    
    # 사용자 입력 받기
    print("데이터 소스를 선택하세요:")
    print("1. CSV 문서만 사용 (판매처 정보)")
    print("2. API 문서만 사용 (가격 정보)")
    
    while True:
        try:
            choice = input("선택 (1 또는 2): ").strip()
            if choice in ['1', '2']:
                break
            else:
                print("1 또는 2를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            return
    
    documents = []
    dataset_filename = ""
    
    if choice == '1':
        # CSV 문서만 사용
        print("\n=== CSV 문서만 사용하여 데이터셋 생성 ===")
        print("CSV 파일에서 판매장 정보를 가져오는 중입니다...")
        csv_file_path = './farmer/sales/data/20240812.csv'
        if os.path.exists(csv_file_path):
            try:
                df = pd.read_csv(csv_file_path, encoding='euc-kr')
                df = df.astype(str)
                
                print(f"CSV에서 전체 {len(df)}개 데이터 사용")
                
                # 데이터를 랜덤으로 섞기
                df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
                print(f"📊 CSV 데이터를 랜덤으로 섞었습니다.")
                
                for index, row in df_shuffled.iterrows():
                    store_name = str(row['판매장 이름']).replace('(', '').replace(')', '').replace(',', ' ').strip()
                    address = str(row['주소']).replace('(', '').replace(')', '').replace(',', ' ').strip()
                    
                    store_name = store_name if store_name and store_name.lower() != 'nan' else "알 수 없는 판매장"
                    address = address if address and address.lower() != 'nan' else "알 수 없는 주소"

                    content = f"판매장 이름: {store_name}. 주소: {address}."
                    documents.append(Document(
                        page_content=content, 
                        metadata={"source": "csv", "store_name": store_name, "address": address}
                    ))
                    
            except Exception as e:
                print(f"CSV 파일 처리 중 오류: {e}")
        else:
            print("CSV 파일을 찾을 수 없습니다.")
            return
        
        # 타임스탬프가 포함된 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        dataset_filename = f"./farmer/sales/data/sales_golden_dataset_{timestamp}.csv"
        
    elif choice == '2':
        # API 문서만 사용
        print("\n=== API 문서만 사용하여 데이터셋 생성 ===")
        print("KAMIS API에서 농산물 가격 정보를 가져오는 중입니다...")
        try:
            api_docs = fetch_api_data()
            print(f"API에서 {len(api_docs)}개의 농산물 가격 정보를 가져왔습니다.")
            
            # API 데이터를 랜덤으로 섞기
            random.shuffle(api_docs)
            print(f"📊 API 데이터를 랜덤으로 섞었습니다.")
            
            for i, doc_text in enumerate(api_docs):
                content = doc_text
                documents.append(Document(
                    page_content=content, 
                    metadata={"source": "kamis_api", "index": i+1}
                ))
                
        except Exception as e:
            print(f"API 데이터 처리 중 오류: {e}")
            return
        
        # 타임스탬프가 포함된 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_filename = f"./farmer/sales/data/price_golden_dataset_{timestamp}.csv"
    
    print(f"총 {len(documents)}개의 문서를 생성했습니다.")
            
    if not documents:
        print("\n로드된 문서가 없습니다. 작업을 종료합니다.")
        return

    print(f"\n총 {len(documents)}개의 문서를 로드했습니다.")
    print("---")

    # RAGAS 모듈들 import
    print("📦 RAGAS 모듈들 로딩 중...")
    ragas_modules = import_ragas_modules()
    print("✅ RAGAS 모듈들 로딩 완료")
    
    # 2. Initialize required models (공식 문서 구조)
    print("Initialize required models...")
    generator_llm = ragas_modules['LangchainLLMWrapper'](ChatOpenAI(model="gpt-4o-mini", temperature=0.7))
    generator_embeddings = ragas_modules['LangchainEmbeddingsWrapper'](
        HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", model_kwargs={'device': 'cpu'})
    )

    # 3. Setup Persona and transforms (공식 문서 구조)
    print("Setup Persona and transforms...")
    
    if choice == '1':
        # CSV (판매처) 전용 페르소나
        personas = [
            ragas_modules['Persona'](
                name="Sales Channel Farmer",
                role_description="I am a new farmer looking for places to sell the crops I've grown. I can't speak English and only use Korean."
            )
        ]
    else:
        # API (가격) 전용 페르소나
        personas = [
            ragas_modules['Persona'](
                name="Price Research Farmer",
                role_description="I am a new farmer and would like to know the current market price and price trends for the crops I've grown. I can't speak English and only use Korean."
            )
        ]

    transforms = [ragas_modules['HeadlineSplitter'](), ragas_modules['NERExtractor']()]

    # 4. Initialize test generator (공식 문서 구조)
    print("Initialize test generator...")
    generator = ragas_modules['TestsetGenerator'](
        llm=generator_llm, 
        embedding_model=generator_embeddings, 
        persona_list=personas
    )

    # 5. Load and Adapt Queries (공식 문서 구조)
    print("Load and Adapt Queries...")
    distribution = [
        (ragas_modules['SingleHopSpecificQuerySynthesizer'](llm=generator_llm), 1.0),
    ]

    for query, _ in distribution:
        prompts = await query.adapt_prompts("korean", llm=generator_llm)
        query.set_prompts(**prompts)

    # 6. Generate (공식 문서 구조)
    print(f"\nGenerate testset with {TARGET_QUESTIONS} questions...")
    try:
        
        # 공식 문서에 따른 테스트셋 생성
        dataset = generator.generate_with_langchain_docs(
            documents[:],
            testset_size=TARGET_QUESTIONS,
            transforms=transforms,
            query_distribution=distribution,
        )

        # 7. 결과 확인 및 저장 (공식 문서 구조)
        eval_dataset = dataset.to_evaluation_dataset()
        
        print("---")
        print("생성된 농가 관점 골든 데이터셋:")
        print("=" * 50)
        
        # 생성된 질문들을 자세히 출력
        for i, sample in enumerate(eval_dataset[:TARGET_QUESTIONS]):
            print(f"\n질문 {i + 1}:")
            print(f"  Query: {sample.user_input}")
            print(f"  Reference: {sample.reference}")
            print("-" * 30)

        # DataFrame으로 변환하여 저장
        df = dataset.to_pandas()
        if len(df) > TARGET_QUESTIONS:
            df = df.head(TARGET_QUESTIONS)
            print(f"요청한 {TARGET_QUESTIONS}개로 제한했습니다.")

        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(dataset_filename), exist_ok=True)
        df.to_csv(dataset_filename, index=False, encoding="utf-8-sig")
        print(f"\n총 {len(df)}개의 농가 관점 질문이 {dataset_filename} 파일로 저장되었습니다.")
        if choice == '1':
            print("이 데이터셋은 농가가 농산물 판매처를 찾을 때 사용할 수 있는 질문-답변 쌍입니다.")
        else:
            print("이 데이터셋은 농가가 농산물 가격 정보를 찾을 때 사용할 수 있는 질문-답변 쌍입니다.")

    except Exception as e:
        print(f"데이터셋 생성 중 오류가 발생했습니다: {e}")

# 스크립트 실행
if __name__ == "__main__":
    asyncio.run(main())