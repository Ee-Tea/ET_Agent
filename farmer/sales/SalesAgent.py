# 주의 무시
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 임베딩 모델
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

# Milvus 연결 설정
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, list_collections, utility
import requests
from dotenv import load_dotenv
import os
import pandas as pd
from konlpy.tag import Okt
import re
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, END
from typing import Dict, Any

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("KAMIS_API_KEY")
api_id = os.getenv("KAMIS_ID")
groq_api_key = os.getenv(f"OPENAI_KEY1")
milvus_host = os.getenv("MILVUS_HOST", "localhost")
milvus_port = os.getenv("MILVUS_PORT", "19530")
collection_name = "market_price_docs"

# CSV 파일 임베딩 및 Milvus에 저장
def embed_and_store_csv(csv_path="data/info_20240812.csv"):
    df = pd.read_csv(csv_path, encoding="euc-kr")
    df['품목'] = df['품목'].fillna("정보 없음")
    docs = []
    for _, row in df.iterrows():
        doc = f"{row['판매장 이름']} ({row['주소']} / 주요 품목: {row['품목']})"
        docs.append(doc)
    if docs:
        embeddings = embedder.encode(docs)
        collection.insert([embeddings.tolist(), docs], fields=["embedding", "text"])

# 컬렉션 있는지 검사
def check_collection():
    global collection
    connections.connect("default", host=milvus_host, port=milvus_port)

    if collection_name in utility.list_collections():
        collection = Collection(collection_name)  # 이미 있으면 기존 컬렉션 사용
        print(f"컬렉션 '{collection_name}'이 이미 존재합니다. 기존 컬렉션을 사용합니다.")
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        ]
        schema = CollectionSchema(fields, "시장 가격 문서 컬렉션")
        collection = Collection(collection_name, schema)
        print(f"컬렉션 '{collection_name}'을 새로 생성했습니다.")
        embed_and_store_csv()

    # 컬렉션에 인덱스 생성 (임베딩 필드에 대해)
    if not collection.has_index():
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
    
    return collection

# api 요청
def fetch_api_data(query=None):
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
            match_count = sum([q == name for q in keywords for name in item_names])
            partial_count = sum([q in name for q in keywords for name in item_names])
            included_keywords = [q for q in keywords if any(q in name for name in item_names)]
            score = 0
            if keywords:
                if match_count > 0:
                    score = 3 + len(included_keywords)  # 완전 일치 + 키워드 개수
                elif partial_count > 0:
                    score = 2 + len(included_keywords)  # 부분 일치 + 키워드 개수
                else:
                    score = len(included_keywords)      # 키워드 일부만 포함
            else:
                score = 0
            filtered_items.append((score, item))

        filtered_items.sort(key=lambda x: x[0], reverse=True)
        filtered_items = [item for _, item in filtered_items]

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
                
                change_str = "변동 없는"
                if str(direction_raw) == "0":
                    change_str = f"{value_raw}%({diff}원) 감소한"
                elif str(direction_raw) == "1":
                    change_str = f"{value_raw}%({diff}원) 증가한"
                
                doc = (
                    f"{safe_val(item.get('item_name', ''))} ({safe_val(item.get('unit', ''))})의 가격은 어제보다 "
                    f"{change_str} {dpr1}원 입니다."
                )
                if dpr3 and str(dpr3).strip() != "" and str(dpr3).strip() != "원":
                    doc += f"{day3}에는 {dpr3}원, "
                if dpr4 and str(dpr4).strip() != "" and str(dpr4).strip() != "원":
                    doc += f"{day4}에는 {dpr4}원 이었습니다."
                docs.append(doc)
    else:
        print("API 호출 실패:", response.status_code)
        print(response.text)
    if docs and any(any(k in doc for k in extract_keywords(query)) for doc in docs):
        return docs
    else:
        return ["해당 작물에 대한 정보는 현재 없습니다."]

# Milvus에서 문서 검색
def search_market_docs(query, top_k=3):
    # 전체 쿼리로 한 번만 검색
    all_results = []

    query_nouns = extract_keywords(query)

    # 미리 정의된 지역명 리스트와 명사 키워드를 비교하여 지역명만 추출
    predefined_locations = ['함평','서산','대전', '춘천','광주', '경산', '강동구', '태안', '성주', '창원', '용인', '울주', '순천', '경주', '양평', '울산광역', '영암', '김제', '고창', '전주', '하동', '제천', '홍성', '화성', '의왕', '담양', '진주', '사천', '남양주', '여수', '유성구', '정읍', '홍천', '남원', '동구', '달서구', '남해', '영동', '서구', '계룡', '고성', '고양', '평택', '남구', '울진', '나주', '전라북도', '익산', '부여', '청도', '합천', '포항', '봉화', '문경', '김해', '함양', '북구', '철원', '화순', '상주', '경북도', '안산', '청양', '충주', '김천', '영광', '성남', '전라남도', '달성', '인제', '천안', '제주', '원주', '가평', '완주', '제천시', '성주군', '고성군', '진천', '거창', '청주', '김포', '화성시', '완도', '함안', '옥천', '김해시', '해남', '무안', '예산', '금산', '강서구', '상당구', '송파구', '공도읍', '곡성', '울릉군', '서귀포', '정선', '평창', '양주', '포천', '진안', '세종']
    locations = [kw for kw in query_nouns if kw in predefined_locations or any(suffix in kw for suffix in ['시', '군', '구', '도'])]

    # 1. 지역 키워드 임베딩 검색
    if locations:
        region_query = " ".join(locations)
        region_vec = embedder.encode([region_query])[0]
        region_results = collection.search(
            data=[region_vec],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 20}},
            limit=200,
            output_fields=["text"],
        )
        if region_results and region_results[0]:
            all_results.extend([hit.entity.get("text") for hit in region_results[0]])

    # 2. 전체 쿼리 임베딩 검색
    query_vec = embedder.encode([query])[0]
    query_results = collection.search(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 20}},
        limit=200,
        output_fields=["text"],
    )
    if query_results and query_results[0]:
        all_results.extend([hit.entity.get("text") for hit in query_results[0]])

    # 중복 제거
    all_results = list(dict.fromkeys(all_results))

    found_results = []
    for result_text in all_results:
        if any(loc in result_text for loc in locations):
            found_results.append(result_text)

    if not found_results:
        return ["해당 지역에 위치한 판매점 정보가 없습니다."]
    else:
        def overlap_score(result_text):
            item_part = result_text.split('(')[0].strip() if '주요 품목:' in result_text else result_text
            item_names = [x.strip() for x in re.split(r'[/,]', item_part)]
            query_strip = query.strip()
            query_nouns_set = set(extract_keywords(query_strip))
            current_score = 0
            if query_strip in item_names:
                current_score += 10000
            for name in item_names:
                name_nouns_set = set(extract_keywords(name))
                if query_nouns_set.issubset(name_nouns_set):
                    current_score += 1000
                current_score += len(query_nouns_set.intersection(name_nouns_set)) * 100
            for name in item_names:
                if any(qn in name for qn in query_nouns_set):
                    if not any(qn in extract_keywords(name) for qn in query_nouns_set):
                        current_score += 1
            return current_score
        
        found_results.sort(key=overlap_score, reverse=True)
        return found_results[:top_k]

# 키워드 추출
def extract_keywords(query):
    okt = Okt()
    return okt.nouns(query)


# 하이브리드 검색
def hybrid_search(query, top_k=3):
    kamis_results = fetch_api_data(query)  # 쿼리 기반 필터링된 결과 반환
    sales_info_results = search_market_docs(query, top_k=top_k)
    print("실시간시세 : ", kamis_results[:1],)
    print("판매처 : ", sales_info_results)
    return {
        "실시간시세": kamis_results[:1],
        "판매처": sales_info_results
    }

# Groq LLM
class GroqLLM:
    def __init__(self, model="openai/gpt-oss-20b", api_key=None):
        self.model = model
        self.api_key = groq_api_key
        self.client = None

    def invoke(self, prompt: str, context: str = None, system_instruction: str = None):
        messages = []
        
        # 시스템 메시지로 컨텍스트와 지시사항 전달
        if context or system_instruction:
            system_content = ""
            if context:
                system_content += f"[참고 정보]\n{context}\n\n"
            if system_instruction:
                system_content += system_instruction
            
            messages.append({
                "role": "system", 
                "content": system_content
            })
        
        # 사용자 질문
        messages.append({
            "role": "user", 
            "content": prompt
        })
        
        try:
            self.client = Groq(api_key=self.api_key)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                max_completion_tokens=512,
                top_p=0.8,
                reasoning_effort="low",
                stream=True,
                stop=None
            )
            result = "".join(chunk.choices[0].delta.content or "" for chunk in completion)
            return result.strip()
        except Exception as e:
            print(f"API key 실패: {e}")
            return f"LLM 호출 실패"

# 프롬프트 생성
def make_system_instruction():
    return """
        [지시]
        - [참고 정보]의 가격과 단위를 정확히 사용
        - 없는 정보는 없다고 안내
        - 순서: 품목/등락율 → 가격정보(없으면 생략) → 판매처 → 출처 

        [예시]
        감자(20kg)의 가격은 어제보다 2.8%(1,060원) 증가한 39,660원입니다.
        1개월전에는 33,260원, 1년전에는 31,576원이었습니다.

        해당 지역의 판매처는 충남 태안군 태안 로컬푸드 판매장(충남 태안군 남면 안면대로 1641 / 주요 품목: 채소, 과일, 서류) 등이 있습니다.

        시세 정보 출처: https://www.kamis.or.kr/customer/main/main.do
        판매처 정보 출처: https://www.data.go.kr/data/15025997/fileData.do
    """

# LLM 호출
def ask_llm_groq(prompt, context="", system_instruction=None, model="openai/gpt-oss-20b"):
    if system_instruction is None:
        system_instruction = make_system_instruction()
    
    llm = GroqLLM(model=model, api_key=groq_api_key)
    return llm.invoke(prompt, context, system_instruction)

# === Agent Node 기반 워크플로우 ===
# 상태 스키마 정의
class GraphState(dict):
    query: str = ""
    context: Dict[str, Any] = {}
    context_str_for_judge: str = ""
    pred_answer: str = ""
    is_recommend_ok: bool = False
    exit: bool = False
    retry_count: int = 0
    final_answer: str = ""

# LangGraph 노드 함수
def node_input_graph(state: GraphState) -> GraphState:
    # 오케스트레이터에서 state["query"]가 이미 전달된 경우, 추가 입력 없이 바로 사용
    if state.get("query"):
        state["retry_count"] = 0  # 새로운 입력 시 재분석 카운트 초기화
        return state
    query = input("작물 및 지역 정보를 입력하세요 (종료하려면 'exit'): ")
    if query.strip().lower() == "exit":
        state["exit"] = True
    else:
        state["query"] = query
        state["retry_count"] = 0
    return state

def node_collect_info_graph(state: GraphState) -> GraphState:
    query = state["query"]
    results = hybrid_search(query, top_k=3)
    state["context"] = results
    return state

def node_llm_summarize_graph(state: GraphState) -> GraphState:
    context = state["context"]
    query = state["query"]
    
    context_str = (
        f"실시간 시세 정보: {context.get('실시간시세', [])}\n"
        f"판매처 정보: {context.get('판매처', [])}"
    )

    # 이전 검증 실패 정보가 있으면 컨텍스트에 추가
    if state.get("validation_details") and state.get("retry_count", 0) > 0:
        validation_feedback = f"\n[이전 검증 실패 정보]\n" + "\n".join([f"• {issue}" for issue in state.get("validation_details", {}).get("issues", [])])
        context_str += validation_feedback + "\n\n위의 문제점들을 해결하여 다시 답변을 생성해주세요."

    # 토큰 절약된 방식으로 LLM 호출
    pred_answer = ask_llm_groq(
        prompt=query,
        context=context_str,
        system_instruction=make_system_instruction()
    )
    
    state["pred_answer"] = pred_answer
    state["context_str_for_judge"] = context_str
    return state

def node_judge_recommendation_graph(state: GraphState) -> GraphState:
    pred_answer = state["pred_answer"]
    original_context = state["context"]
    
    # 1. 가격 정보 정확성 검증
    def price_validation():
        try:
            # 원본 컨텍스트에서 가격 값 추출 (중복 제거하지 않음)
            context_prices = []
            
            for doc in original_context.get('실시간시세', []):
                # 가격 정보 추출 (콤마가 포함된 숫자 + '원' 패턴)
                price_matches = re.findall(r'(\d{1,3}(?:,\d{3})*)원', doc)
                context_prices.extend(price_matches)
                
                # 콤마가 없는 숫자 + '원' 패턴 (4자리 이상만)
                simple_price_matches = re.findall(r'(\d{4,})원', doc)
                context_prices.extend(simple_price_matches)
            
            # 중복 제거하지 않고 순서대로 유지
            print(f"원본 컨텍스트에서 추출된 가격 (순서 유지): {context_prices}")
            
            # LLM 답변에서 가격 정보 추출 (동일한 패턴 적용)
            answer_prices = []
            answer_price_matches = re.findall(r'(\d{1,3}(?:,\d{3})*)원', pred_answer)
            answer_prices.extend(answer_price_matches)
            
            simple_answer_matches = re.findall(r'(\d{4,})원', pred_answer)
            answer_prices.extend(simple_answer_matches)
            
            # 중복 제거하지 않고 순서대로 유지
            print(f"LLM 답변에서 추출된 가격 (순서 유지): {answer_prices}")
            
            # 가격 매칭 검증 (1:1 매칭, 순서 고려, 중복 제한)
            if not context_prices:
                print("원본 컨텍스트에 가격 정보가 없습니다.")
                return False, ['원본 컨텍스트에 가격 정보 없음'], 0, 2
            
            # 원본 가격의 출현 횟수 계산
            context_price_count = {}
            for price in context_prices:
                context_price_count[price] = context_price_count.get(price, 0) + 1
            
            print(f"원본 가격 출현 횟수: {context_price_count}")
            
            # 1:1 매칭 검증 (순서대로, 정확한 매칭만, 중복 제한)
            matched_prices = []
            missing_prices = []
            hallucination_prices = []
            used_answer_indices = set()  # 이미 사용된 답변 인덱스
            matched_price_count = {}  # 매칭된 가격의 횟수 추적
            
            # 원본 가격을 순서대로 확인
            for i, context_price in enumerate(context_prices):
                matched = False
                
                # 답변에서 정확히 일치하는 가격 찾기
                for j, answer_price in enumerate(answer_prices):
                    if j in used_answer_indices:
                        continue
                    
                    # 이미 해당 가격을 최대 허용 횟수만큼 매칭했다면 건너뛰기
                    if context_price in matched_price_count:
                        current_count = matched_price_count[context_price]
                        max_allowed = context_price_count[context_price]
                        print(f"  중복 체크: {context_price} (현재 {current_count}/{max_allowed})")
                        if current_count >= max_allowed:
                            print(f"  중복 제한으로 건너뛰기: {context_price}")
                            continue
                    
                    # 정확한 매칭만 허용
                    if context_price == answer_price:
                        matched_prices.append(context_price)
                        used_answer_indices.add(j)
                        matched_price_count[context_price] = matched_price_count.get(context_price, 0) + 1
                        matched = True
                        print(f"정확한 매칭: {context_price} ← {answer_price} (매칭 횟수: {matched_price_count[context_price]})")
                        break
                
                if not matched:
                    missing_prices.append(context_price)
            
            # 할루시네이션 가격이 있는지 확인 (LLM 답변에 원본에 없는 가격이 있는지)
            for j, answer_price in enumerate(answer_prices):
                if j not in used_answer_indices:
                    # 원본에 없는 가격인지 확인 (정확한 매칭만)
                    is_original = False
                    for context_price in context_prices:
                        if answer_price == context_price:
                            is_original = True
                            break
                    
                    if not is_original:
                        hallucination_prices.append(f"원본에 없는 가격: {answer_price}")
            
            # 중복 매칭 문제 확인 (LLM 답변에서 원본보다 많이 나오는 가격)
            answer_price_count = {}
            for price in answer_prices:
                answer_price_count[price] = answer_price_count.get(price, 0) + 1
            
            print(f"LLM 답변 가격 출현 횟수: {answer_price_count}")
            
            for price, answer_count in answer_price_count.items():
                context_count = context_price_count.get(price, 0)
                if answer_count > context_count:
                    hallucination_prices.append(f"가격 중복 할루시네이션: {price} (원본 {context_count}회, 답변 {answer_count}회)")
            
            # 가격 매칭 점수 계산 (100% 매칭되어야만 점수 부여)
            price_match_score = len(matched_prices) / len(context_prices)
            is_perfect_match = price_match_score == 1.0
            
            print(f"가격 매칭 점수: {len(matched_prices)}/{len(context_prices)} = {price_match_score:.2f}")
            print(f"완벽한 매칭: {'✅' if is_perfect_match else '❌'}")
            print(f"매칭된 가격 횟수: {matched_price_count}")
            
            # 검증 로직
            issues = []
            
            # 1. 가격 정보 매칭 - 100% 매칭되어야만 통과
            if is_perfect_match and not hallucination_prices:  # 할루시네이션이 없어야 함
                print("✅ 가격 정보 완벽 매칭 - 통과")
                price_valid = True
            else:
                print("❌ 가격 정보 매칭 실패 - 불통과")
                price_valid = False
                
                if hallucination_prices:
                    issues.append(f'할루시네이션 감지로 인한 가격 매칭 실패')
                else:
                    issues.append(f'가격 정보 불완전 매칭: {len(matched_prices)}/{len(context_prices)}')
                
                if missing_prices:
                    issues.append(f'누락된 가격: {missing_prices}')
                if hallucination_prices:
                    issues.append(f'할루시네이션 가격: {hallucination_prices}')
            
            # 상세한 검증 정보 출력
            print(f"\n=== 상세 검증 결과 ===")
            if matched_prices:
                print(f"✅ 매칭된 가격: {matched_prices}")
            if missing_prices:
                print(f"❌ 누락된 가격: {missing_prices}")
            if hallucination_prices:
                print(f" 할루시네이션 가격: {hallucination_prices}")
            
            return price_valid, issues
            
        except Exception as e:
            print(f"가격 검증 중 오류: {e}")
            return False, [f'검증 오류: {e}'], 0, 2
    
    # 2. 구조적 완성도 검증 (임베딩 유사도 사용)
    def structural_validation():
        try:
            # 원본 컨텍스트와 LLM 답변의 임베딩 유사도 계산
            context_embedding = embedder.encode([state["context_str_for_judge"]])[0].reshape(1, -1)
            answer_embedding = embedder.encode([pred_answer])[0].reshape(1, -1)
            
            similarity_score = cosine_similarity(context_embedding, answer_embedding)[0][0]
            
            # 임계값 설정 (기존과 동일)
            ACCURACY_THRESHOLD = 0.8
            is_valid = similarity_score > ACCURACY_THRESHOLD
            
            print(f"구조적 완성도 (임베딩 유사도): {similarity_score:.4f}, 임계값: {ACCURACY_THRESHOLD}")
            print(f"구조적 완성도 검증: {'✅' if is_valid else '❌'}")
            
            return is_valid
            
        except Exception as e:
            print(f"구조적 검증 중 오류: {e}")
            return False
    
    # 3. 판매점 정보 할루시네이션 검증
    def vendor_hallucination_validation():
        try:
            # 판매점 정보 부족을 나타내는 키워드
            no_vendor_keywords = [
                '판매점 정보가 없습니다',
                '판매점이 없습니다',
                '판매처 정보가 없습니다',
                '판매처가 없습니다',
                '해당 지역에 위치한 판매점 정보가 없습니다',
                '판매점을 찾을 수 없습니다',
                '판매 정보가 없습니다'
            ]
            
            # 원본 컨텍스트에 판매점 정보가 있는지 확인
            context_has_vendors = False
            if '판매처' in original_context:
                vendor_info = original_context['판매처']
                if vendor_info and len(vendor_info) > 0:
                    # 실제 판매점 정보가 있는지 확인 (빈 문자열이나 "정보 없음"이 아닌 경우)
                    for vendor in vendor_info:
                        if vendor and vendor != "해당 지역에 위치한 판매점 정보가 없습니다." and len(vendor.strip()) > 0:
                            context_has_vendors = True
                            break
            
            # LLM 답변에 판매점 정보 부족 키워드가 있는지 확인
            answer_has_no_vendor = any(keyword in pred_answer for keyword in no_vendor_keywords)
            
            # 할루시네이션 판단
            hallucination_detected = False
            hallucination_issues = []
            
            if context_has_vendors and answer_has_no_vendor:
                # 원본에 판매점 정보가 있는데 LLM이 "없습니다"라고 답변
                hallucination_detected = True
                hallucination_issues.append("판매점 정보 할루시네이션: 원본에 판매점 정보가 있음에도 '없습니다'라고 표시")
                print("❌ 판매점 정보 할루시네이션 감지: 원본에 판매점 정보가 있음에도 '없습니다'라고 표시")
            
            elif not context_has_vendors and not answer_has_no_vendor:
                # 원본에 판매점 정보가 없는데 LLM이 "있습니다"라고 답변
                hallucination_detected = True
                hallucination_issues.append("판매점 정보 할루시네이션: 원본에 판매점 정보가 없음에도 '있습니다'라고 표시")
                print("❌ 판매점 정보 할루시네이션 감지: 원본에 판매점 정보가 없음에도 '있습니다'라고 표시")
            
            else:
                print("✅ 판매점 정보 할루시네이션 검증 통과")
            
            return not hallucination_detected, hallucination_issues
            
        except Exception as e:
            print(f"판매점 정보 할루시네이션 검증 중 오류: {e}")
            return False, [f'검증 오류: {e}']
    
    # 종합 검증
    price_valid, price_issues = price_validation()
    structural_valid = structural_validation()
    vendor_valid, vendor_hallucination_issues = vendor_hallucination_validation()
    
    validation_results = {
        'price_accuracy': price_valid,
        'structural': structural_valid,
        'vendor_hallucination': vendor_valid,
        'issues': price_issues + vendor_hallucination_issues
    }
    
    # 검증 결과 출력
    print(f"검증 결과:")
    print(f"  - 가격 정확성: {'✅' if price_valid else '❌'}")
    print(f"  - 구조적 완성도: {'✅' if structural_valid else '❌'}")
    print(f"  - 판매점 정보 할루시네이션: {'✅' if vendor_valid else '❌'}")
    
    if validation_results['issues']:
        print(f"  - 발견된 문제점:")
        for issue in validation_results['issues']:
            print(f"    • {issue}")
    
    # 모든 검증을 통과해야 함
    overall_validation = price_valid and structural_valid and vendor_valid
    
    state["is_recommend_ok"] = overall_validation
    state["validation_details"] = validation_results
    state["needs_web_search"] = not price_valid or not vendor_valid  # 가격 검증 또는 판매점 검증 실패 시 웹 검색 필요
    
    return state

def node_reanalyze_graph(state: GraphState) -> GraphState:
    state["retry_count"] += 1
    query = state["query"]

    # 이전 검증 실패 정보 출력
    if state.get("validation_details"):
        print(f"재분석 {state['retry_count']}회차: 이전 검증에서 발견된 문제점:")
        for issue in state.get("validation_details", {}).get("issues", []):
            print(f"  - {issue}")
        print("위 문제점들을 해결하여 재분석합니다.")
    
    results = hybrid_search(query, top_k=3)
    state["context"] = results
    return state

def node_output_graph(state: GraphState) -> GraphState:
    if state["retry_count"] >= 2 and not state["is_recommend_ok"]:
        state["final_answer"] = "해당 작물과 지역에 대한 시세 또는 판매처 정보가 없습니다. 혹시 다른 작물이나 지역을 찾아드릴까요?"
    else:
        state["final_answer"] = f"{state['pred_answer']}"
    return state

# LangGraph 워크플로우 정의
graph = StateGraph(GraphState)

graph.add_node("input", node_input_graph)
graph.add_node("collect_info", node_collect_info_graph)
graph.add_node("llm_summarize", node_llm_summarize_graph)
graph.add_node("judge_recommendation", node_judge_recommendation_graph)
graph.add_node("reanalyze", node_reanalyze_graph)
graph.add_node("output", node_output_graph)

graph.add_edge("input", "collect_info")
graph.add_edge("collect_info", "llm_summarize")
graph.add_edge("llm_summarize", "judge_recommendation")

# 조건부 분기 로직
def judge_branch(state: GraphState) -> str:
    if state.get("exit"):
        return END
    
    if state.get("is_recommend_ok"):
        return "output"
    
    # 2회 재분석 후에도 적절하지 않으면 종료
    if state["retry_count"] >= 2:
        return "output"
    else:
        return "reanalyze"

graph.add_conditional_edges(
    "judge_recommendation", 
    judge_branch,
    {
        "output": "output",
        "reanalyze": "reanalyze"
    }
)
graph.add_edge("reanalyze", "llm_summarize")
graph.add_edge("output", END)

graph.set_entry_point("input")

# 실행 함수
def run(state):
    """
    판매처 에이전트의 워크플로우를 실행합니다.
    오케스트레이터에서 전달받은 상태를 바탕으로 LangGraph를 실행합니다.
    """
    check_collection()
    collection.load()
    app = graph.compile()
    
    # LangGraph가 TypedDict를 기반으로 작동하기 때문에, 일반 Dict를 TypedDict로 변환
    if not isinstance(state, GraphState):
        state = GraphState(**state)
        
    result_state = app.invoke(state)
    return result_state

if __name__ == "__main__":
    # 판매처 에이전트 단독 실행용 코드
    check_collection()
    collection.load()
    print("=== 판매처 에이전트 단독 실행 모드 ===")
    
    # LangGraph를 컴파일하고 단독으로 실행
    app = graph.compile()
    result_state = app.invoke({"query": "동두천에서 배추를 팔고 싶어"})
    
    print("\n" + "=" * 50)
    if result_state.get('final_answer'):
        print(f"\n[최종 답변]")
        print(result_state['final_answer'])
