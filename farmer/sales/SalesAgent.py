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
from typing import Dict, Any, List, Optional
from langchain_community.tools.tavily_search import TavilySearchResults

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("KAMIS_API_KEY")
api_id = os.getenv("KAMIS_ID")
groq_api_key = os.getenv(f"OPENAI_KEY1")
milvus_host = os.getenv("MILVUS_HOST", "localhost")
milvus_port = os.getenv("MILVUS_PORT", "19530")
collection_name = "market_price_docs"


# CSV 파일 임베딩 및 Milvus에 저장
def embed_and_store_csv(csv_path="sales/info_20240812.csv"):
    df = pd.read_csv(csv_path, encoding="euc-kr")
    df['품목'] = df['품목'].fillna("정보 없음")
    docs = []
    for _, row in df.iterrows():
        doc = f"{row['판매장 이름']} ({row['주소']} / 주요 품목: {row['품목']})"
        docs.append(doc)
    if docs:
        embeddings = embedder.encode(docs)
        collection.insert([embeddings.tolist(), docs], fields=["embedding", "text"])

def check_collection():
    global collection
    connections.connect("default", host=milvus_host, port=milvus_port)

    if collection_name in utility.list_collections():
        collection = Collection(collection_name)
        collection.load()
        
        # 실제 쿼리로 데이터 존재 여부 확인
        try:
            print(f"🔍 컬렉션 '{collection_name}' 데이터 존재 여부 확인 중...")
            
            # 실제 쿼리로 데이터 확인
            sample_results = collection.query(
                expr="id >= 0",
                output_fields=["id", "text"],
                limit=1
            )
            
            has_data = len(sample_results) > 0
            print(f"쿼리 결과: {len(sample_results)}개")
            print(f"🔍 데이터 존재 여부: {'있음' if has_data else '없음'}")
            
            if has_data:
                print(f"✅ 데이터가 존재합니다. 기존 컬렉션을 보존합니다.")
                return collection
            else:
                print(f"⚠️ 데이터가 없습니다. 삭제 후 재생성합니다.")
                utility.drop_collection(collection_name)
                print(f"✅ 컬렉션 '{collection_name}' 삭제 완료")
                
                # 새 컬렉션 생성
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
                ]
                schema = CollectionSchema(fields, "시장 가격 문서 컬렉션")
                collection = Collection(collection_name, schema)
                print(f"🔄 컬렉션 '{collection_name}' 재생성 완료")
                
                # 데이터 삽입
                embed_and_store_csv()
                print(f"✅ 데이터 삽입 완료")
                
        except Exception as e:
            print(f"❌ 쿼리 확인 중 오류: {e}")
            print(f"⚠️ 오류 발생으로 인해 기존 컬렉션을 보존합니다.")
            return collection
    else:
        # 컬렉션이 없는 경우 새로 생성
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
        print(f"✅ 인덱스 생성 완료")
    
    return collection

# 질문 분류 함수
def classify_question_simple(query: str) -> str:
    """핵심 의도 키워드만으로 질문을 분류합니다."""
    query_lower = query.lower()
    
    # 핵심 의도 키워드 (가장 중요한 것들만)
    selling_intent = ['팔고 싶어', '팔 수 있', '거래', '판매', '매매', '팔래','팔고싶어','팔수 있','팔수있','팔 수있', '팔까', '팔면', '파는게', '파는 것', '파는것']
    price_intent = ['가격', '시세', '얼마', '값', '원']
    location_intent = ['파는 곳', '판매점', '직매장', '시장', '어디', '파는곳','판매처']
    
    # "농작물"이 포함된 경우 특별 처리
    if "농작물" in query_lower:
        if any(keyword in query_lower for keyword in selling_intent):
            return "판매처" # "농작물을 팔고 싶어" → 판매처
        else:
            return "정보 부족"  # "농작물"만
    
    # 일반적인 분류 로직
    if any(keyword in query_lower for keyword in selling_intent):
        return "시세+판매처"  # 구체적인 작물명 + 팔고 싶다는 의도
    
    if any(keyword in query_lower for keyword in price_intent):
        if any(keyword in query_lower for keyword in location_intent):
            return "시세+판매처"  # 가격 + 위치 모두 요구
        else:
            return "시세"  # 가격만 요구
    
    if any(keyword in query_lower for keyword in location_intent):
        return "판매처"  # 위치만 요구
    
    return "시세+판매처"

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

# Milvus에서 문서 검색
def search_market_docs(query, top_k=3):
    # 전역 변수 collection을 사용하지 않고 로컬에서 처리
    try:
        # Milvus 연결
        connections.connect("default", host=milvus_host, port=milvus_port)
        
        # 컬렉션 로드
        if collection_name in utility.list_collections():
            local_collection = Collection(collection_name)
            local_collection.load()
            print(f"✅ Milvus 컬렉션 '{collection_name}' 로드 완료")
        else:
            print(f"❌ Milvus 컬렉션 '{collection_name}'을 찾을 수 없습니다.")
            connections.disconnect("default")
            return ["판매점 정보를 찾을 수 없습니다."]
        
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
            region_results = local_collection.search(
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
        query_results = local_collection.search(
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
            connections.disconnect("default")
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
            connections.disconnect("default")
            return found_results[:top_k]
            
    except Exception as e:
        print(f"❌ Milvus 검색 오류: {e}")
        try:
            connections.disconnect("default")
        except:
            pass
        return ["판매점 정보를 가져오는 중 오류가 발생했습니다."]

def supplement_missing_info_with_web_search(query: str, missing_info_type: str, existing_context: dict) -> dict:
    """웹 검색으로 부족한 정보를 보완합니다."""
    print(f"🔍 웹 검색으로 {missing_info_type} 정보 보완 중...")
    
    supplemented_context = existing_context.copy()
    
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not tavily_api_key:
            print("⚠️ Tavily API 키가 설정되지 않았습니다.")
            return supplemented_context
        
        tavily_tool = TavilySearchResults(max_results=5, api_key=tavily_api_key)
        
        search_queries = []
        if missing_info_type == "판매처":
            vendor_types = ["농산물 공판장", "로컬푸드 직매장", "농협 농산물산지유통센터", "농산물 도매시장"]
            for v_type in vendor_types:
                search_queries.append(f"{query} {v_type}")
        else:
            search_queries.append(f"{query} {missing_info_type}")

        # 여러 검색어로 검색 실행 및 결과 취합
        all_search_results = []
        seen_urls = set()
        for s_query in search_queries:
            results = tavily_tool.invoke({"query": s_query})
            for result in results:
                url = result.get('url')
                if url not in seen_urls:
                    all_search_results.append(result)
                    seen_urls.add(url)
        
        web_info = []
        if all_search_results:
            for result in all_search_results:
                summary = result.get('content', '')[:150]
                web_info.append(f"웹 검색: {result.get('title', '')} - {summary}... (출처: {result.get('url')})")

        if web_info:
            if missing_info_type == "판매처":
                supplemented_context['판매처'] = web_info
            elif missing_info_type == "시세":
                supplemented_context['실시간시세'] = web_info
            
            supplemented_context['used_web_search'] = True
            print(f"✅ 웹 검색 완료: {missing_info_type} 정보 보완")
        else:
            print(f"⚠️ 웹 검색으로도 {missing_info_type} 정보를 찾을 수 없었습니다.")
            
    except Exception as e:
        print(f"❌ 웹 검색 중 오류 발생: {e}")
    
    return supplemented_context

# 키워드 추출
def extract_keywords(query):
    okt = Okt()
    return okt.nouns(query)

def execute_milvus_search(query: str) -> list[str]:
    """Milvus 검색을 실행하는 공통 함수"""
    try:
        connections.connect("default", host=milvus_host, port=milvus_port)
        collection = Collection(collection_name)
        collection.load()
        results = search_market_docs(query, top_k=3)
        connections.disconnect("default")
        return results
    except Exception as e:
        print(f"❌ Milvus 연결 오류: {e}")
        return ["판매점 정보를 가져오는 중 오류가 발생했습니다."]

***REMOVED*** LLM
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
                max_completion_tokens=1024,
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
def make_system_instruction(classification="시세+판매처"):
    """질문 분류에 따라 적절한 시스템 지시사항을 생성합니다."""

    templates = {
        "시세": {
            "order": "품목/등락율 → 가격정보(없으면 생략) → 출처",
            "exclude": "판매처 정보는 포함하지 마세요 (시세 질문이므로)",
            "example": "감자(20kg)의 가격은 어제보다 2.8%(1,060원) 증가한 39,660원입니다.\n1개월전에는 33,260원, 1년전에는 31,576원이었습니다.\n\n시세 정보 출처: https://www.kamis.or.kr/customer/main/main.do"
        },
        "판매처": {
            "order": "판매처 정보 → 출처",
            "exclude": "시세 정보는 포함하지 마세요 (판매처 질문이므로)",
            "example": "해당 지역의 판매처는 충남 태안군 태안 로컬푸드 판매장(충남 태안군 남면 안면대로 1641 / 주요 품목: 채소, 과일, 서류) 등이 있습니다.\n\n판매처 정보 출처: https://www.data.go.kr/data/15025997/fileData.do"
        },
        "시세+판매처": {
            "order": "품목/등락율 → 가격정보(없으면 생략) → 판매처 → 출처",
            "exclude": "",
            "example": "감자(20kg)의 가격은 어제보다 2.8%(1,060원) 증가한 39,660원입니다.\n1개월전에는 33,260원, 1년전에는 31,576원이었습니다.\n\n해당 지역의 판매처는 충남 태안군 태안 로컬푸드 판매장(충남 태안군 남면 안면대로 1641 / 주요 품목: 채소, 과일, 서류) 등이 있습니다.\n\n시세 정보 출처: https://www.kamis.or.kr/customer/main/main.do\n판매처 정보 출처: https://www.data.go.kr/data/15025997/fileData.do"
        }
    }
    
    template = templates.get(classification, templates["시세+판매처"])
    
    return f"""
    [지시]
    - [참고 정보]의 가격과 단위를 정확히 사용
    - 없는 정보는 없다고 안내
    - 순서: {template['order']}
    {f"- {template['exclude']}" if template['exclude'] else ""}

    [예시]
    {template['example']}
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
    question_classification: str = ""
    context: Dict[str, Any] = {}
    pred_answer: str = ""
    is_recommend_ok: bool = False
    exit: bool = False
    retry_count: int = 0
    final_answer: str = ""
    needs_web_search: bool = False
    missing_info_types: List[str] = []
    used_web_search: bool = False
    validation_details: Optional[dict] = None

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

# 질문 분류 함수
def node_classify_question(state: GraphState) -> GraphState:
    """질문을 분류하여 적절한 도구를 결정합니다."""
    query = state["query"]
    print(f"🔍 질문 분류 중...")
    
    classification = classify_question_simple(query)
    state["question_classification"] = classification
    
    print(f"✅ 질문 분류 완료: {classification}")
    return state

def node_collect_info_graph(state: GraphState) -> GraphState:
    """질문 분류에 따라 적절한 도구를 선택하여 정보를 수집합니다."""
    query = state["query"]
    classification = state.get("question_classification", "시세+판매처")
    
    print(f"🛠️ 정보 수집 중...")
    
    # 분류에 따라 직접 도구 실행
    results = {
        "실시간시세": [],
        "판매처": [],
        "웹검색": []
    }
    
    if classification == "시세":
        results["실시간시세"] = fetch_api_data(query)[:1]
        results["판매처"] = ["해당 지역에 위치한 판매점 정보가 없습니다."]
    elif classification == "판매처":
        results["실시간시세"] = ["해당 작물에 대한 정보는 현재 없습니다."]
        results["판매처"] = execute_milvus_search(query)
    elif classification == "시세+판매처":
        results["실시간시세"] = fetch_api_data(query)[:1]
        results["판매처"] = execute_milvus_search(query)

    state["context"] = results
    
    # 사용된 도구 정보 기록
    tools_used = []
    if results.get("실시간시세"):
        tools_used.append("시세 API")
    if results.get("판매처"):
        tools_used.append("판매처 정보")
    if results.get("웹검색"):
        tools_used.append("웹 검색")
    
    print(f"✅ 정보 수집 완료.")
    
    return state

def node_llm_summarize_graph(state: GraphState) -> GraphState:
    """LLM을 사용하여 최종 답변을 생성합니다."""
    print("💻 최종 답변 생성 중...")
    classification = state["question_classification"]
    
    # 컨텍스트를 출처에 따라 명확히 구분하여 구성
    context = state["context"]
    context_str = ""

    # 1. API/DB 정보 분리 (웹 검색 결과가 아닌 것)
    db_prices = [p for p in context.get("실시간시세", []) if not str(p).startswith('웹 검색:')]
    db_vendors = [v for v in context.get("판매처", []) if not str(v).startswith('웹 검색:')]

    if db_prices:
        context_str += "[실시간시세 정보 (API)]\n" + "\n".join(map(str, db_prices)) + "\n\n"
    if db_vendors:
        context_str += "[판매처 정보 (DB)]\n" + "\n".join(map(str, db_vendors)) + "\n\n"

    # 2. 웹 검색 정보 분리
    web_prices = [p for p in context.get("실시간시세", []) if str(p).startswith('웹 검색:')]
    web_vendors = [v for v in context.get("판매처", []) if str(v).startswith('웹 검색:')]

    if web_prices or web_vendors:
        context_str += "[웹 검색 정보]\n"
        if web_prices:
            context_str += "\n".join(map(str, web_prices)) + "\n"
        if web_vendors:
            context_str += "\n".join(map(str, web_vendors)) + "\n"
        context_str += "\n"

    # 시스템 지침 구성
    system_instruction = make_system_instruction(classification)
    
    source_instruction = """
    [출처 표기 지침]
    - 답변에 정보를 포함할 때, 반드시 아래 규칙에 따라 출처를 명시해야 합니다.
    - `[실시간시세 정보 (API)]`에서 가져온 정보의 출처는 'https://www.kamis.or.kr/customer/main/main.do'을 명시하세요.
    - `[판매처 정보 (DB)]`에서 가져온 정보의 출처는 'https://www.data.go.kr/data/15025997/fileData.do'을 명시하세요.
    - `[웹 검색 정보]`에서 가져온 정보의 출처는 각 항목 끝에 '(출처: URL)' 형식으로 제공된 URL을 사용해야 합니다.
    - 만약 "정보가 없습니다"라는 내용만 있다면, 해당 정보와 출처는 답변에 포함하지 마세요.
    - 각 정보의 출처를 절대 섞거나 잘못 표기해서는 안 됩니다.
    """
    system_instruction += source_instruction

    if state.get("used_web_search"):
        web_search_instruction = f"""
        [추가 지시]
        - 사용자의 원래 질문인 '{state['query']}'와 가장 관련성 높은 정보를 중심으로 답변을 요약해주세요.
        """
        # 판매처 정보가 부족하여 웹 검색을 했을 경우, LLM에게 사용자의 의도를 명확히 전달
        if "판매처" in state.get("missing_info_types", []):
            web_search_instruction += "- 사용자는 농작물을 '판매'할 수 있는 장소(공판장, 도매시장, 로컬푸드 등)를 찾고 있으니, 검색 결과를 바탕으로 해당 장소들을 추천해주세요.\n- 판매처 정보는 사용자의 질문에 포함된 지역과 다르다면 해당 정보는 무시\n"
        
        web_search_instruction += "- 만약 웹 검색 결과에도 유용한 정보가 없다면, 정보가 없다고 솔직하게 답변해주세요."
        system_instruction += web_search_instruction

    # 검증 피드백 추가
    if state.get("validation_details") and state.get("retry_count", 0) > 0:
        issues = state["validation_details"].get("issues", [])
        context_str += f"\n[이전 검증 실패 정보]\n" + "\n".join([f"• {issue}" for issue in issues])
        context_str += "\n\n위의 문제점들을 해결하여 다시 답변을 생성해주세요."

    # LLM 호출
    pred_answer = ask_llm_groq(
        prompt=state["query"],
        context=context_str,
        system_instruction=system_instruction
    )
    
    state.update({
        "pred_answer": pred_answer
    })
    return state

def validate_prices(original_context, pred_answer):
    """가격 검증 (기존 로직 유지)"""
    # 핵심 검증만 수행
    context_prices = []
    answer_prices = []

    # 원본 컨텍스트에서 가격 값 추출 (콤마가 포함된 숫자 + '원' 패턴)
    for doc in original_context.get('실시간시세', []):
        # "해당 작물에 대한 정보는 현재 없습니다" 체크
        if "해당 작물에 대한 정보는 현재 없습니다" in doc:
            return False, ["시세 정보가 없습니다"]
            
        price_matches = re.findall(r'(\d{1,3}(?:,\d{3})*)원', doc)
        context_prices.extend(price_matches)
        
        # 콤마가 없는 숫자 + '원' 패턴 (4자리 이상만)
        simple_price_matches = re.findall(r'(\d{4,})원', doc)
        context_prices.extend(simple_price_matches)
    
    # 중복 제거하지 않고 순서대로 유지

    # LLM 답변에서 가격 정보 추출 (동일한 패턴 적용)
    answer_price_matches = re.findall(r'(\d{1,3}(?:,\d{3})*)원', pred_answer)
    answer_prices.extend(answer_price_matches)
    
    simple_answer_matches = re.findall(r'(\d{4,})원', pred_answer)
    answer_prices.extend(simple_answer_matches)

    # 중복 제거하지 않고 순서대로 유지

    # 가격 매칭 검증 (1:1 매칭, 순서 고려, 중복 제한)
    
    # 원본 가격의 출현 횟수 계산
    context_price_count = {}
    for price in context_prices:
        context_price_count[price] = context_price_count.get(price, 0) + 1
    
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
                if current_count >= max_allowed:
                    continue
            
            # 정확한 매칭만 허용
            if context_price == answer_price:
                matched_prices.append(context_price)
                used_answer_indices.add(j)
                matched_price_count[context_price] = matched_price_count.get(context_price, 0) + 1
                matched = True
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
    
    for price, answer_count in answer_price_count.items():
        context_count = context_price_count.get(price, 0)
        if answer_count > context_count:
            hallucination_prices.append(f"가격 중복 할루시네이션: {price} (원본 {context_count}회, 답변 {answer_count}회)")
    
    # 가격 매칭 점수 계산 (100% 매칭되어야만 점수 부여)
    price_match_score = len(matched_prices) / len(context_prices)
    is_perfect_match = price_match_score == 1.0
    
    # 검증 로직
    issues = []
    
    # 1. 가격 정보 매칭 - 100% 매칭되어야만 통과
    if is_perfect_match and not hallucination_prices:  # 할루시네이션이 없어야 함
        price_valid = True
    else:
        price_valid = False
        
        # 할루시네이션에 대한 피드백
        if hallucination_prices:
            issues.append(f"제공된 정보에 없는 내용(예: '{hallucination_prices[0]}')을 생성했습니다. 제공된 참고 정보만 사용해주세요.")

        # 누락된 가격에 대한 피드백
        if missing_prices:
            context_docs = original_context.get('실시간시세', [])
            missing_info_feedback = []
            
            # 누락된 가격이 포함된 원본 문서를 찾음
            for doc in context_docs:
                doc_prices = re.findall(r'(\d{1,3}(?:,\d{3})*)원', doc) + re.findall(r'(\d{4,})원', doc)
                if any(price in doc_prices for price in missing_prices):
                    missing_info_feedback.append(doc)

            # 구체적인 피드백 메시지 생성
            if missing_info_feedback:
                unique_feedback_docs = list(dict.fromkeys(missing_info_feedback))
                issues.append('다음 중요 정보를 답변에서 누락했습니다. 반드시 포함시켜 다시 답변해주세요: ' + " | ".join(unique_feedback_docs))
            # 원본 문서를 찾지 못한 경우에 대한 예외 처리
            elif missing_prices:
                issues.append(f'가격 정보를 일부({len(missing_prices)}개) 누락했습니다.')
        
        if missing_prices:
            issues.append(f'누락된 가격: {missing_prices}')
        if hallucination_prices:
            issues.append(f'할루시네이션 가격: {hallucination_prices}')
    
    # 상세한 검증 정보 출력
    return price_valid, issues

def validate_vendors(original_context, pred_answer):
    """판매점 검증 (기존 로직 유지)"""
    # 핵심 검증만 수행
    context_has_vendors = False
    answer_has_no_vendor = False

    # 원본 컨텍스트에 판매점 정보가 있는지 확인
    if '판매처' in original_context:
        vendor_info = original_context['판매처']
        if vendor_info and len(vendor_info) > 0:
            # 실제 판매점 정보가 있는지 확인 (빈 문자열이나 "정보 없음"이 아닌 경우)
            for vendor in vendor_info:
                if vendor and vendor != "해당 지역에 위치한 판매점 정보가 없습니다." and len(vendor.strip()) > 0:
                    context_has_vendors = True
                    break

    # 판매점 정보가 없으면 검증 실패로 처리 (웹 검색 필요)
    if not context_has_vendors:
        return False, ["판매처 정보가 없습니다"]

    # LLM 답변에 판매점 정보 부족 키워드가 있는지 확인
    no_vendor_keywords = [
        '판매점 정보가 없습니다',
        '판매점이 없습니다',
        '판매처 정보가 없습니다',
        '판매처가 없습니다',
        '해당 지역에 위치한 판매점 정보가 없습니다',
        '판매점을 찾을 수 없습니다',
        '판매 정보가 없습니다'
    ]
    answer_has_no_vendor = any(keyword in pred_answer for keyword in no_vendor_keywords)

    # 할루시네이션 판단
    hallucination_detected = False
    hallucination_issues = []

    if context_has_vendors and answer_has_no_vendor:
        # 원본에 판매점 정보가 있는데 LLM이 "없습니다"라고 답변
        hallucination_detected = True
        hallucination_issues.append("판매점 정보 할루시네이션: 원본에 판매점 정보가 있음에도 '없습니다'라고 표시")
    
    elif not context_has_vendors and not answer_has_no_vendor:
        # 원본에 판매점 정보가 없는데 LLM이 "있습니다"라고 답변
        hallucination_detected = True
        hallucination_issues.append("판매점 정보 할루시네이션: 원본에 판매점 정보가 없음에도 '있습니다'라고 표시")
    
    return not hallucination_detected, hallucination_issues

def node_judge_recommendation_graph(state: GraphState) -> GraphState:
    """응답 품질 검증 및 재생성 여부 결정"""
    pred_answer = state["pred_answer"]
    original_context = state["context"]
    question_classification = state.get("question_classification", "시세+판매처")
    retry_count = state.get("retry_count", 0)
    
    print(f"🔍 응답 품질 검증 중... (질문 분류: {question_classification}, 재시도: {retry_count}회)")
    
    # 검증 실행
    validations = {}
    
    # 가격 검증 (시세 관련 질문일 때만)
    if question_classification in ["시세", "시세+판매처"]:
        validations['price'] = validate_prices(original_context, pred_answer)
    
    # 판매점 검증 (판매처 관련 질문일 때만)
    if question_classification in ["판매처", "시세+판매처"]:
        validations['vendor'] = validate_vendors(original_context, pred_answer)
    
    # 전체 검증 결과 - 튜플의 첫 번째 값(is_valid)만 추출하여 평가
    all_valid = all(is_valid for is_valid, _ in validations.values()) if validations else True
    all_issues = [issue for _, issues in validations.values() for issue in issues]
    
    print(f"✅ 검증 완료: {'통과' if all_valid else '실패'}")
    
    # 웹 검색 필요성 판단 - 1회 재분석 후에만 고려
    needs_web_search = False
    missing_info_types = []
    
    # 1회 재분석 후에도 검증이 실패한 경우에만 웹 검색 고려
    if retry_count >= 1 and not all_valid:
        # 검증 실패 시 어떤 정보가 부족한지 분석
        for validation_name, (is_valid, issues) in validations.items():
            if not is_valid:
                if validation_name == 'price':
                    missing_info_types.append("시세")
                elif validation_name == 'vendor':
                    missing_info_types.append("판매처")
        
        # 웹 검색이 필요한 경우 상태 업데이트
        if missing_info_types and not state.get("used_web_search"):
            needs_web_search = True
    
    # 상태 업데이트
    state.update({
        "is_recommend_ok": all_valid,
        "validation_details": {"validations": validations, "issues": all_issues},
        "needs_web_search": needs_web_search,
        "missing_info_types": missing_info_types
    })

    return state

def node_reanalyze_graph(state: GraphState) -> GraphState:
    state["retry_count"] += 1
    print(f"🔄 재분석 중... (시도: {state['retry_count']}회)")
    # node_collect_info_graph와 동일한 로직 사용
    return node_collect_info_graph(state)

def node_web_search_supplement(state: GraphState) -> GraphState:
    """웹 검색으로 부족한 정보를 보완합니다."""
    if not state.get("needs_web_search"):
        return state
    
    query = state["query"]
    original_context = state["context"]
    missing_info_types = state.get("missing_info_types", [])
    
    print(f"🔍 웹 검색으로 정보 보강 중...")
    
    # 부족한 정보 타입별로 웹 검색 수행
    supplemented_context = original_context.copy()
    
    for info_type in missing_info_types:
        supplemented_context = supplement_missing_info_with_web_search(
            query, info_type, supplemented_context
        )
    
    # 보완된 컨텍스트로 상태 업데이트
    state["context"] = supplemented_context
    state["used_web_search"] = True
    
    return state

def node_output_graph(state: GraphState) -> GraphState:
    # 웹 검색을 통해 답변을 보완한 경우, 검증 결과와 상관없이 보완된 답변을 최종 답변으로 사용
    if state.get("used_web_search"):
        state["final_answer"] = f"{state['pred_answer']}"
        return state

    if state["retry_count"] >= 1 and not state["is_recommend_ok"]:
        # 웹 검색 없이 1회 재분석 후 종료
        state["final_answer"] = "해당 작물과 지역에 대한 시세 또는 판매처 정보가 없습니다. 혹시 다른 작물이나 지역을 찾아드릴까요?"
    else:
        state["final_answer"] = f"{state['pred_answer']}"
    return state

# LangGraph 워크플로우 정의
graph = StateGraph(GraphState)

graph.add_node("input", node_input_graph)
graph.add_node("classify_question", node_classify_question)
graph.add_node("collect_info", node_collect_info_graph)
graph.add_node("llm_summarize", node_llm_summarize_graph)
graph.add_node("judge_recommendation", node_judge_recommendation_graph)
graph.add_node("web_search_supplement", node_web_search_supplement)  # 웹 검색 노드 추가
graph.add_node("reanalyze", node_reanalyze_graph)
graph.add_node("output", node_output_graph)

graph.add_edge("input", "classify_question")
graph.add_edge("classify_question", "collect_info")
graph.add_edge("collect_info", "llm_summarize")

# 조건부 분기 로직
def summarize_branch(state: GraphState) -> str:
    """LLM 요약 후 분기: 웹 검색을 했다면 검증 없이 바로 출력, 아니면 검증으로 이동"""
    if state.get("used_web_search"):
        return "output"
    else:
        return "judge_recommendation"

def judge_branch(state: GraphState) -> str:
    """검증 결과에 따라 분기"""
    if state.get("exit"):
        return END
    
    if state.get("is_recommend_ok"):
        return "output"
    
    # 재분석 횟수가 1회 미만인 경우
    if state["retry_count"] < 1:
        return "reanalyze"
    
    # 재분석 횟수가 1회 이상인 경우
    # 웹 검색이 필요하고 아직 수행되지 않았다면 웹 검색 실행
    if state.get("needs_web_search") and not state.get("used_web_search"):
        return "web_search_supplement"
    
    # 재시도가 모두 소진되었거나 웹 검색을 이미 수행한 경우 종료
    return "output"

graph.add_conditional_edges(
    "llm_summarize",
    summarize_branch,
    {
        "output": "output",
        "judge_recommendation": "judge_recommendation",
    },
)

graph.add_conditional_edges(
    "judge_recommendation", 
    judge_branch,
    {
        "output": "output",
        "web_search_supplement": "web_search_supplement",  # 웹 검색 분기 추가
        "reanalyze": "reanalyze"
    }
)
graph.add_edge("web_search_supplement", "llm_summarize")  # 웹 검색 후 LLM으로
graph.add_edge("reanalyze", "llm_summarize")
graph.add_edge("output", END)

graph.set_entry_point("input")

# 실행 함수
def run(state):
    """
    판매처 에이전트의 워크플로우를 실행합니다.
    오케스트레이터에서 전달받은 상태를 바탕으로 LangGraph를 실행합니다.
    """
    print("\n\n===== Sales Agent 실행 시작 =====")
    # 컬렉션 초기화 추가
    try:
        check_collection()
    except Exception as e:
        print(f"❌ Milvus 컬렉션 초기화 실패: {e}")
        # 컬렉션 초기화 실패 시에도 계속 진행
    
    app = graph.compile()
    
    # LangGraph가 TypedDict를 기반으로 작동하기 때문에, 일반 Dict를 TypedDict로 변환
    if not isinstance(state, GraphState):
        state = GraphState(**state)
        
    result_state = app.invoke(state)
    return result_state

if __name__ == "__main__":
    # 판매처 에이전트 단독 실행용 코드
    print("=== 판매처 에이전트 단독 실행 모드 ===")
    
    # LangGraph를 컴파일하고 단독으로 실행
    app = graph.compile()

    # 판매처 에이전트 단독 실행, 그래프 시각화
    try:
        graph_image_path = "sales_agent_workflow.png"
        with open(graph_image_path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")
    result_state = app.invoke({"query": "덕정에서 사과를 팔고 싶어"})
    
    print("\n" + "=" * 50)
    if result_state.get('final_answer'):
        print(f"\n[최종 답변]")
        print(result_state['final_answer'])