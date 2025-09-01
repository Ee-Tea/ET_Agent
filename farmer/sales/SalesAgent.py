# 주의 무시
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 설정
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, list_collections, utility
import requests
from dotenv import load_dotenv
import os
import pandas as pd
from konlpy.tag import Okt
import re
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, Optional, TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime
from sentence_transformers import SentenceTransformer
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 임베딩 모델
embedder = SentenceTransformer("BAAI/bge-m3")

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("KAMIS_API_KEY")
api_id = os.getenv("KAMIS_ID")
openai_api_key = os.getenv("OPENAI_KEY1")
milvus_host = os.getenv("MILVUS_HOST", "localhost")
milvus_port = os.getenv("MILVUS_PORT", "19530")
collection_name = "market_price_docs"

# LLM 및 프롬프트 설정
llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct",
               temperature=0.7,
               api_key=openai_api_key)
# gpt-4o-mini
# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.8, api_key=os.getenv("OPENAI_API_KEY=REDACTED GraphState(TypedDict):
    query: str
    question_classification: str
    context: Dict[str, Any]
    pred_answer: str
    is_recommend_ok: bool
    exit: bool
    retry_count: int
    final_answer: str
    needs_web_search: bool
    missing_info_types: List[str]
    used_web_search: bool
    validation_details: Optional[dict]
    ragas_scores: Optional[Dict[str, float]]
    ragas_passed: bool

# 키워드 추출
def extract_keywords(query):
    okt = Okt()
    return okt.nouns(query)

# ===============================
# 사용자 질문 받기
# ===============================
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

# ===============================
# 질문 분류
# ===============================
def node_classify_question(state: GraphState) -> GraphState:
    """질문을 분류하여 적절한 도구를 결정합니다."""
    query = state["query"]
    print(f"🔍 질문 분류 중...")
    
    classification = classify_question_simple(query)
    state["question_classification"] = classification
    
    print(f"✅ 질문 분류 완료: {classification}")
    return state

# 키워드로 질문 분류 함수
def classify_question_simple(query: str) -> str:
    """LLM을 사용하여 질문을 분류합니다."""
    
    # 분류 프롬프트
    classification_prompt = ChatPromptTemplate.from_template("""
당신은 농작물 시세 및 판매처 관련 질문을 분류하는 전문가입니다.
사용자의 질문을 분석하여 다음 4가지 카테고리 중 하나로 분류해주세요:

1. "시세" - 가격, 시세, 얼마, 값, 원 등 가격 정보만 요구하는 경우
2. "판매처" - 파는 곳, 판매점, 직매장, 시장, 어디, 판매처 등 판매 장소만 요구하는 경우  
3. "시세+판매처" - 가격과 판매처 정보를 모두 요구하는 경우
4. "정보 부족" - 구체적인 작물명이 없는데 시세를 요구한 경우 혹은 구체적인 지역명이 없는데 판매처를 요구한 경우

**분류 기준 예시:**
- "시세"와 "판매처" 키워드가 함께 있으면 → "시세+판매처"
- "농작물"과 "시세" 키워드가 함께 있을 때는 작물명의 유무에 따라 → "시세" or "정보 부족"
- "농작물"과 "판매처" 키워드가 함께 있을 때는 지역명의 유무에 따라 → "판매처" or "정보 부족"
- 구체적인 작물명과 "팔고 싶어", "판매" 등이 있으면 → "시세+판매처"
- 가격 관련 키워드만 있으면 → "시세"
- 판매처 관련 키워드만 있으면 → "판매처"
- 애매한 경우 → "시세+판매처" (기본값)

질문: {query}

분류 결과 (반드시 위 4가지 중 하나만 출력):
""")
    
    try:
        # LLM 체인 생성
        chain = classification_prompt | llm | StrOutputParser()
        
        # 분류 실행
        result = chain.invoke({"query": query})
        
        # 결과 정리
        classification = result.strip()
        
        # 유효한 분류인지 확인
        valid_classifications = ["시세", "판매처", "시세+판매처", "정보 부족"]
        if classification not in valid_classifications:
            print(f"⚠️ LLM 분류 결과가 유효하지 않음: {classification}, 기본값 사용")
            return "시세+판매처"
        
        return classification
        
    except Exception as e:
        print(f"⚠️ LLM 분류 중 오류 발생: {e}, 기본값 사용")
        return "시세+판매처"

# ===============================
# 정보 수집
# ===============================
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

# ===============================
# 정보 수집 - api요청 함수
# ===============================
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

# ===============================
# 정보 수집 - milvus 검색 함수
# ===============================
def execute_milvus_search(query: str) -> list[str]:
    """Milvus 검색을 실행하는 공통 함수"""
    try:
        connections.connect("default", host=milvus_host, port=milvus_port)

        if not utility.has_collection(collection_name):
            print(f"❌ Milvus 컬렉션 '{collection_name}'을 찾을 수 없습니다.")
            connections.disconnect("default")
            return ["판매점 정보를 찾을 수 없습니다."]

        collection = Collection(collection_name)
        collection.load()
        print(f"✅ Milvus 컬렉션 '{collection_name}' 로드 완료")

        results = search_market_docs(query, collection, top_k=3)
        connections.disconnect("default")
        return results
    except Exception as e:
        print(f"❌ Milvus 연결 오류: {e}")
        return ["판매점 정보를 가져오는 중 오류가 발생했습니다."]

# CSV 파일 임베딩 및 Milvus에 저장 함수
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

# Milvus에서 문서 검색 함수
def search_market_docs(query, local_collection, top_k=3):
    # 전역 변수 collection을 사용하지 않고 로컬에서 처리
    try:
        # 전체 쿼리로 한 번만 검색
        all_results = []

        query_nouns = extract_keywords(query)

        # 미리 정의된 지역명 리스트와 명사 키워드를 비교하여 지역명만 추출
        predefined_locations = ['인천','함평','서산','대전', '춘천','광주', '경산', '강동구', '태안', '성주', '창원', '용인', '울주', '순천', '경주', '양평', '울산', '영암', '김제', '고창', '전주', '하동', '제천', '홍성', '화성', '의왕', '담양', '진주', '사천', '남양주', '여수', '유성구', '정읍', '홍천', '남원', '동구', '달서구', '남해', '영동', '서구', '계룡', '고성', '고양', '평택', '남구', '울진', '나주', '전라북도', '익산', '부여', '청도', '합천', '포항', '봉화', '문경', '김해', '함양', '북구', '철원', '화순', '상주', '경북도', '안산', '청양', '충주', '김천', '영광', '성남', '전라남도', '달성', '인제', '천안', '제주', '원주', '가평', '완주', '제천시', '성주군', '고성군', '진천', '거창', '청주', '김포', '화성시', '완도', '함안', '옥천', '김해시', '해남', '무안', '예산', '금산', '강서구', '상당구', '송파구', '공도읍', '곡성', '울릉군', '서귀포', '정선', '평창', '양주', '포천', '진안', '세종']
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
            
    except Exception as e:
        print(f"❌ Milvus 검색 오류: {e}")
        try:
            connections.disconnect("default")
        except:
            pass
        return ["판매점 정보를 가져오는 중 오류가 발생했습니다."]

# ===============================
# 답변 생성
# ===============================
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

    if state.get("used_web_search"):
        web_search_instruction = f"""
        [웹 검색으로 인한 추가 지시]
        - 사용자의 원래 질문인 '{state['query']}'와 가장 관련성 높은 정보를 중심으로 답변을 요약해주세요.
        """
        # 판매처 정보가 부족하여 웹 검색을 했을 경우, LLM에게 사용자의 의도를 명확히 전달
        if "판매처" in state.get("missing_info_types", []):
            web_search_instruction += "- 사용자는 **사용자가 농작물을 '판매'할 수 있는 장소(공판장, 도매시장, 로컬푸드 등)**를 찾고 있으니, 검색 결과를 바탕으로 해당 장소들을 추천해주세요.\n- 판매처 정보는 사용자의 질문에 포함된 지역과 다르다면 해당 정보는 무시\n"
        
        web_search_instruction += "- 만약 웹 검색 결과에도 유용한 정보가 없다면, 정보가 없다고 솔직하게 답변해주세요."
        system_instruction += web_search_instruction

    # 검증 피드백 추가
    if state.get("validation_details") and state.get("retry_count", 0) > 0:
        issues = state["validation_details"].get("issues", [])
        context_str += f"\n[이전 검증 실패 정보]\n" + "\n".join([f"• {issue}" for issue in issues])
        context_str += "\n\n위의 문제점들을 해결하여 다시 답변을 생성해주세요."

    # LLM 호출
    pred_answer = ask_llm_openai(
        prompt=state["query"],
        context=context_str,
        system_instruction=system_instruction
    )
    
    state.update({
        "pred_answer": pred_answer
    })
    return state

# 질문 분류에 따른 프롬프트 생성 함수
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
    [출처 규칙] 
    - `[실시간시세 정보 (API)]`에서 가져온 정보의 출처는 'https://www.kamis.or.kr/customer/main/main.do'을 명시
    - `[판매처 정보 (DB)]`에서 가져온 정보의 출처는 'https://www.data.go.kr/data/15025997/fileData.do'을 명시
    - `[웹 검색 정보]`에서 가져온 정보의 출처는 각 항목 끝에 '(출처: URL)' 형식으로 제공된 URL을 사용
    - **출처는 반드시 명시**해야하지만 정보가 없다면 출처도 제외
    {f"- {template['exclude']}" if template['exclude'] else ""}

    [예시]
    {template['example']}
    """

# LLM 호출 함수
def ask_llm_openai(prompt, context="", system_instruction=None, model="gpt-4o-mini"):
    if system_instruction is None:
        system_instruction = make_system_instruction()
    
    # LangChain ChatGroq를 사용하여 LLM 호출
    try:
        # 시스템 메시지와 사용자 메시지 구성
        messages = []
        
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
        
        messages.append({
            "role": "user", 
            "content": prompt
        })
        
        # ChatGroq를 사용하여 응답 생성
        response = llm.invoke(messages)
        return response.content.strip()
        
    except Exception as e:
        print(f"LLM 호출 실패: {e}")
        return f"LLM 호출 실패"

# ===============================
# 응답 품질 검증
# ===============================
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

# ===============================
# 응답 품질 검증 - 시세 검증 함수
# ===============================
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

    # LLM 답변에서 가격 정보 추출 (동일한 패턴 적용)
    answer_price_matches = re.findall(r'(\d{1,3}(?:,\d{3})*)원', pred_answer)
    answer_prices.extend(answer_price_matches)
    
    simple_answer_matches = re.findall(r'(\d{4,})원', pred_answer)
    answer_prices.extend(simple_answer_matches)
    
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

# ===============================
# 응답 품질 검증 - 판매처 검증 함수
# ===============================
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

# ===============================
# 검증 불충족으로 재분석
# ===============================
def node_reanalyze_graph(state: GraphState) -> GraphState:
    state["retry_count"] += 1
    print(f"🔄 재분석 중... (시도: {state['retry_count']}회)")
    # node_collect_info_graph와 동일한 로직 사용
    return node_collect_info_graph(state)

# ===============================
# 재분석으로 부족한 정보를 웹 검색으로 보완
# ===============================
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

# 웹 검색 함수
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
            vendor_types = ["농산물 공판장", "로컬푸드 직매장", "농산물 도매시장"]
            for v_type in vendor_types:
                search_queries.append(f"{query} {v_type}")
        elif missing_info_type == "시세":
            # 날짜와 구체적인 키워드를 추가하여 검색 정확도 향상
            today_str = datetime.now().strftime("%Y년")
            price_types = [
                "도매 가격", 
                "농산물유통정보 시세",
                "가락시장 시세"
            ]
            for p_type in price_types:
                search_queries.append(f"{today_str} {query} {p_type}")

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
            key = None
            not_found_msg = ""
            if missing_info_type == "판매처":
                key = '판매처'
                not_found_msg = "해당 지역에 위치한 판매점 정보가 없습니다."
            elif missing_info_type == "시세":
                key = '실시간시세'
                not_found_msg = "해당 작물에 대한 정보는 현재 없습니다."

            if key:
                current_info = supplemented_context.get(key, [])
                # "정보 없음" 메시지를 필터링하여 기존 유효한 정보만 남김
                filtered_info = [info for info in current_info if not_found_msg not in info]
                # 웹 검색 결과를 추가
                supplemented_context[key] = filtered_info + web_info
            
            supplemented_context['used_web_search'] = True
            print(f"✅ 웹 검색 완료: {missing_info_type} 정보 보완")
        else:
            print(f"⚠️ 웹 검색으로도 {missing_info_type} 정보를 찾을 수 없었습니다.")
            
    except Exception as e:
        print(f"❌ 웹 검색 중 오류 발생: {e}")
    
    return supplemented_context

# ===============================
# RAGAS 검증
# ===============================
def node_ragas_validation(state: GraphState) -> GraphState:
    """RAGAS를 사용하여 답변 품질을 평가합니다."""
    print("📊 RAGAS 검증 중...")
    
    # RAGAS 임계값 설정
    CONTEXT_PRECISION_THRESHOLD = 0.7
    FAITHFULNESS_THRESHOLD = 0.7
    ANSWER_RELEVANCY_THRESHOLD = 0.5
    
    try:
        # SalesRAGAS 모듈에서 필요한 함수들 import
        from SalesRAGAS import SalesRAGASEvaluator
        
        # 평가기 초기화
        evaluator = SalesRAGASEvaluator()
        
        # 현재 상태에서 평가 데이터 준비
        question = state["query"]
        answer = state["pred_answer"]
        context = state["context"]
        
        # 컨텍스트를 문자열로 변환 (SalesRAGAS의 _format_context 메서드 활용)
        context_str = evaluator._format_context(context)
        
        # 개별 RAGAS 평가 실행 (비동기)
        async def run_ragas_evaluation():
            return await evaluator._evaluate_single_ragas_simple(
                question, answer, context_str
            )
        
        # 비동기 실행
        ragas_scores = asyncio.run(run_ragas_evaluation())
        
        if ragas_scores:
            state["ragas_scores"] = ragas_scores
            print(f"✅ RAGAS 검증 완료:")
            
            # 임계값 확인
            context_precision_score = ragas_scores.get('context_precision', 0.0)
            faithfulness_score = ragas_scores.get('faithfulness', 0.0)
            answer_relevancy_score = ragas_scores.get('answer_relevancy', 0.0)
            
            # 임계값 미달 시 실패 처리 (모든 메트릭이 임계값을 넘어야 통과)
            if (context_precision_score < CONTEXT_PRECISION_THRESHOLD or 
                faithfulness_score < FAITHFULNESS_THRESHOLD or 
                answer_relevancy_score < ANSWER_RELEVANCY_THRESHOLD):
                state["ragas_passed"] = False
                print("❌ RAGAS 임계값 미달로 실패")
            else:
                state["ragas_passed"] = True
                print("✅ RAGAS 임계값 통과")
        else:
            state["ragas_scores"] = {}
            state["ragas_passed"] = False
            print("⚠️ RAGAS 검증 실패")
        
    except Exception as e:
        print(f"❌ RAGAS 검증 중 오류 발생: {e}")
        state["ragas_scores"] = {}
        state["ragas_passed"] = False
    
    return state

# ===============================
# 최종 답변
# ===============================
def node_output_graph(state: GraphState) -> GraphState:
    # RAGAS 검증 결과 확인
    ragas_passed = state.get("ragas_passed", False)
    
    # RAGAS 임계값 미달 시 실패 메시지 출력
    if not ragas_passed:
        state["final_answer"] = "죄송합니다. 해당 작물과 지역에 대한 시세 또는 판매처 정보를 찾을 수 없습니다. 혹시 다른 작물이나 지역을 찾아드릴까요?"
    else:
        # RAGAS 통과 시 정상 답변 출력
        state["final_answer"] = f"{state['pred_answer']}"
    
    # RAGAS 점수가 있으면 출력
    if state.get("ragas_scores"):
        print(f"\n📊 RAGAS 평가 결과:")
        for metric, score in state["ragas_scores"].items():
            print(f"  - {metric}: {score:.3f}")
    
    return state

# ===============================
# LangGraph 워크플로우 정의
# ===============================
graph = StateGraph(GraphState)

graph.add_node("input", node_input_graph)
graph.add_node("classify_question", node_classify_question)
graph.add_node("collect_info", node_collect_info_graph)
graph.add_node("llm_summarize", node_llm_summarize_graph)
graph.add_node("judge_recommendation", node_judge_recommendation_graph)
graph.add_node("web_search_supplement", node_web_search_supplement)  # 웹 검색 노드 추가
graph.add_node("reanalyze", node_reanalyze_graph)
graph.add_node("ragas_validation", node_ragas_validation)  # RAGAS 검증 노드 추가
graph.add_node("output", node_output_graph)

graph.add_edge("input", "classify_question")
graph.add_edge("classify_question", "collect_info")
graph.add_edge("collect_info", "llm_summarize")

# 회귀자 여부
def summarize_branch(state: GraphState) -> str:
    if state.get("used_web_search"):
        return "ragas_validation"  # 웹 검색 후에도 RAGAS 검증으로
    else:
        return "judge_recommendation"

# 검증 결과에 따라 분기
def judge_branch(state: GraphState) -> str:
    if state.get("exit"):
        return END
    
    if state.get("is_recommend_ok"):
        return "ragas_validation"  # RAGAS 검증으로 이동
    
    # 재분석 횟수가 1회 미만인 경우
    if state["retry_count"] < 1:
        return "reanalyze"
    
    # 재분석 횟수가 1회 이상인 경우
    # 웹 검색이 필요하고 아직 수행되지 않았다면 웹 검색 실행
    if state.get("needs_web_search") and not state.get("used_web_search"):
        return "web_search_supplement"
    
    # 재시도가 모두 소진되었거나 웹 검색을 이미 수행한 경우 RAGAS 검증으로 이동
    return "ragas_validation"

graph.add_conditional_edges(
    "llm_summarize",
    summarize_branch,
    {
        "ragas_validation": "ragas_validation",  # RAGAS 검증으로 변경
        "judge_recommendation": "judge_recommendation",
    },
)

graph.add_conditional_edges(
    "judge_recommendation", 
    judge_branch,
    {
        "ragas_validation": "ragas_validation",  # RAGAS 검증으로 변경
        "web_search_supplement": "web_search_supplement",  # 웹 검색 분기 추가
        "reanalyze": "reanalyze"
    }
)
graph.add_edge("web_search_supplement", "llm_summarize")  # 웹 검색 후 LLM으로
graph.add_edge("reanalyze", "llm_summarize")
graph.add_edge("ragas_validation", "output")  # RAGAS 검증 후 output으로
graph.add_edge("output", END)

graph.set_entry_point("input")

# 실행 함수
def run(state):
    print("\n\n===== Sales Agent 실행 시작 =====")
    app = graph.compile()
    result_state = app.invoke(state)
    return result_state

if __name__ == "__main__":
    # 판매처 에이전트 단독 실행용 코드
    print("=== 판매처 에이전트 단독 실행 모드 ===")
    
    # LangGraph를 컴파일하고 단독으로 실행
    app = graph.compile()

    # 판매처 에이전트 단독 실행, 그래프 시각화
    # try:
    #     graph_image_path = "sales_agent_workflow.png"
    #     with open(graph_image_path, "wb") as f:
    #         f.write(app.get_graph().draw_mermaid_png())
    #     print(f"\nLangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    # except Exception as e:
    #     print(f"그래프 시각화 중 오류 발생: {e}")
    result_state = app.invoke({"query": "경주에 위치한 농작물 판매처와 꽈리고추 시세를 알려주세요"})
    
    print("\n" + "=" * 50)
    if result_state.get('final_answer'):
        print(f"\n[최종 답변]")
        print(result_state['final_answer'])