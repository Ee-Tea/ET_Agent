"""
RAGAS 평가용 데이터 생성 스크립트
CSV 데이터와 API 데이터를 무작위로 섞어서 사용자 질문과 예상 답변을 생성합니다.
"""

import pandas as pd
import requests
import json
import random
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("KAMIS_API_KEY")
api_id = os.getenv("KAMIS_ID")
openai_api_key = os.getenv("OPENAI_API_KEY")

# LLM 설정
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.8, api_key=openai_api_key)

# ========================================
# 전역 설정 변수들 (필요에 따라 수정 가능)
# ========================================

# 각 반복에서 사용할 데이터 개수
API_ITEMS_PER_ITERATION = 1                     # 각 반복에서 API에서 가져올 가격 정보 개수
CSV_STORES_PER_ITERATION = 1                    # 각 반복에서 CSV에서 가져올 매장 정보 개수

# 반복 횟수 (최종 생성될 질문-답변 쌍 수)
TOTAL_ITERATIONS = 50                           # 총 반복 횟수

# 파일 경로
CSV_FILE_PATH = "data/20240812.csv"             # CSV 파일 경로
OUTPUT_FILE_PATH = "ragas_evaluation_data.json" # 출력 파일 경로

def extract_keywords(query: str) -> List[str]:
    """쿼리에서 키워드를 추출합니다."""
    if not query:
        return []
    
    # 간단한 키워드 추출 (공백으로 분리)
    keywords = [word.strip() for word in query.split() if len(word.strip()) > 1]
    return keywords

def fetch_api_data(query=None):
    """API에서 가격 데이터를 가져옵니다."""
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

        # 쿼리 기반 필터링
        keywords = extract_keywords(query)
        filtered_items = []
        for item in items:
            item_name_full = safe_val(item.get('item_name', ''))
            item_name_parts = item_name_full.split('/')
            item_names = [part.strip() for part in item_name_parts]
            # 개선된 점수 계산 로직
            exact_matches = set()  # 완전 일치한 키워드들
            partial_matches = set()  # 부분 일치한 키워드들
            
            for keyword in keywords:
                for name in item_names:
                    if keyword == name:
                        exact_matches.add(keyword)
                    elif keyword in name:
                        partial_matches.add(keyword)
            
            # 완전 일치한 키워드는 부분 일치에서 제외
            partial_matches = partial_matches - exact_matches
            
            score = 0
            if keywords:
                # 추가 보너스: 여러 키워드 매칭시 추가 점수
                score = (len(exact_matches) * 10) + (len(partial_matches) * 5)
                total_matches = len(exact_matches) + len(partial_matches)
                if total_matches > 1:
                    score += total_matches * 2  # 다중 매칭 보너스
            
            match_count = len(exact_matches)
            partial_count = len(partial_matches)
            included_keywords = list(exact_matches | partial_matches)
            
            filtered_items.append((score, item))

        filtered_items.sort(key=lambda x: x[0], reverse=True)
        filtered_items = [item for _, item in filtered_items]
        
        processed_count = 0
        for item in filtered_items:
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
                    doc += f"{day3}에는 {dpr3}원, "
                if dpr4 and str(dpr4).strip() != "" and str(dpr4).strip() != "원":
                    doc += f"{day4}에는 {dpr4}원 이었습니다."
                docs.append(doc)
    else:
        print("API 호출 실패:", response.status_code)

    if docs and any(any(k in doc for k in extract_keywords(query)) for doc in docs):
        return docs
    else:
        return ["해당 작물에 대한 정보는 현재 없습니다."]

def load_csv_data(file_path: str) -> pd.DataFrame:
    """CSV 파일을 로드합니다."""
    try:
        df = pd.read_csv(file_path, encoding="euc-kr")
        df['품목'] = df['품목'].fillna("정보 없음")
        return df
    except Exception as e:
        print(f"CSV 파일 로드 실패: {e}")
        return pd.DataFrame()

def create_single_combined_info(csv_data: pd.DataFrame, api_data: List[str]) -> str:
    """API에서 무작위 1개, CSV에서 무작위 1개를 선택하여 하나의 combined 정보를 생성합니다."""
    
    combined_info = "=== 농산물 가격 및 직매장 정보 ===\n\n"
    
    # API 데이터에서 무작위로 1개 선택
    if api_data:
        combined_info += "## 농산물 가격 정보\n"
        selected_api = random.choice(api_data)
        combined_info += f"- {selected_api}\n\n"
    
    # CSV 데이터에서 무작위로 1개 선택
    if not csv_data.empty:
        combined_info += "## 직매장 정보\n"
        selected_store = csv_data.sample(1).iloc[0]
        store_name = selected_store['판매장 이름']
        address = selected_store['주소']
        items = selected_store['품목']
        combined_info += f"- {store_name} ({address}) - 주요 품목: {items}\n"
    
    return combined_info

def generate_single_question_with_llm(combined_data: str) -> Dict[str, Any]:
    """LLM을 사용하여 단일 질문-답변 쌍을 생성합니다."""
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """당신은 농산물 가격 및 직매장 정보를 바탕으로 RAGAS 평가용 질문-답변 쌍을 생성하는 전문가입니다.

주어진 정보를 바탕으로 하나의 질문과 답변을 생성해주세요:

1. 가격 관련 질문 예시: "감자 가격이 어떻게 되나요?", "사과 시세를 알려주세요" 등
2. 판매처 질문 예시: "대전에 위치한 농산물 직매장 위치는?", "강원도에서 농작물을 판매하고 싶은데 어디서 팔아야 할까요?" 등
3. 가격 + 판매처 질문 예시: "대전에 위치한 농산물 직매장 위치와 감자 가격을 알고 싶어요." 등

답변은 주어진 정보를 바탕으로 정확하고 상세하게 작성해주세요.
가격 정보가 있으면 가격 변동률과 과거 가격도 포함하고,
매장 정보가 있으면 매장 위치와 주요 품목을 포함해주세요.

JSON 형식으로 응답해주세요:
{
  "user_input": "질문 내용",
  "ground_truth": "상세한 답변 내용"
}"""),
        ("user", "다음 정보를 바탕으로 하나의 질문-답변 쌍을 생성해주세요:\n\n{data}")
    ])
    
    chain = prompt_template | llm
    
    try:
        response = chain.invoke({
            "data": combined_data
        })
        
        # JSON 파싱
        content = response.content
        # JSON 부분만 추출 (```json ... ``` 형태일 수 있음)
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        
        sample = json.loads(content)
        return sample
        
    except Exception as e:
        print(f"LLM 호출 실패: {e}")
        return {"user_input": "오류 발생", "ground_truth": "데이터 생성 중 오류가 발생했습니다."}

def generate_questions_and_answers(csv_data: pd.DataFrame, api_data: List[str]) -> List[Dict[str, Any]]:
    """반복문을 사용하여 질문과 답변을 생성합니다."""
    
    samples = []
    
    print(f"총 {TOTAL_ITERATIONS}개의 질문-답변 쌍을 생성합니다...")
    
    for i in range(TOTAL_ITERATIONS):
        print(f"진행률: {i+1}/{TOTAL_ITERATIONS}")
        
        # 각 반복에서 무작위로 데이터 선택하여 combined 정보 생성
        combined_data = create_single_combined_info(csv_data, api_data)
        
        # LLM으로 단일 질문-답변 쌍 생성
        sample = generate_single_question_with_llm(combined_data)
        samples.append(sample)
        
        # 진행 상황 출력
        if (i + 1) % 10 == 0:
            print(f"  - {i+1}개 완료")
    
    return samples

def main():
    """메인 함수"""
    print("RAGAS 평가용 데이터 생성 시작...")
    print(f"설정: 각 반복당 API {API_ITEMS_PER_ITERATION}개, CSV {CSV_STORES_PER_ITERATION}개")
    print(f"총 반복 횟수: {TOTAL_ITERATIONS}회")
    
    # CSV 데이터 로드
    csv_data = load_csv_data(CSV_FILE_PATH)
    print(f"CSV 데이터 로드 완료: {len(csv_data)}개 행")
    
    # API 데이터 가져오기
    print("API 데이터 가져오는 중...")
    api_data = fetch_api_data()
    print(f"API 데이터 로드 완료: {len(api_data)}개 항목")
    
    # 질문과 답변 생성 (반복문 방식)
    print("질문과 답변 생성 중...")
    samples = generate_questions_and_answers(csv_data, api_data)
    
    # JSON 파일로 저장
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"\nRAGAS 평가용 데이터 생성 완료: {OUTPUT_FILE_PATH}")
    print(f"총 {len(samples)}개의 샘플이 생성되었습니다.")
    
    # 샘플 출력
    print("\n생성된 샘플 예시:")
    for i, sample in enumerate(samples[:3]):
        print(f"\n--- 샘플 {i+1} ---")
        print(f"질문: {sample['user_input']}")
        print(f"답변: {sample['ground_truth']}")

if __name__ == "__main__":
    main()