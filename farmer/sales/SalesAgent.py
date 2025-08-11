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
groq_api_keys = [os.getenv(f"OPENAI_KEY{i}") for i in range(1, 4)]
milvus_host = os.getenv("MILVUS_HOST", "localhost")
milvus_port = os.getenv("MILVUS_PORT", "19530")

# Milvus 연결 및 컬렉션 설정
connections.connect("default", host=milvus_host, port=milvus_port)

collection_name = "market_price_docs"

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
            if keywords:
                if match_count > 0:
                    filtered_items.append((3, item))  # 완전 일치
                elif partial_count > 0:
                    filtered_items.append((2, item))  # 부분 일치
            else:
                filtered_items.append((0, item))

        filtered_items.sort(key=lambda x: x[0], reverse=True)
        filtered_items = [item for _, item in filtered_items]

        for item in filtered_items:
            category = safe_val(item.get('category_name', ''))
            if category not in ['수산물', '축산물'] and safe_val(item.get('product_cls_name', '')) != '소매':
                direction_raw = safe_val(item.get('direction', ''))
                value_raw = safe_val(item.get('value', ''))
                dpr1 = safe_val(item.get('dpr1', ''))
                dpr2 = safe_val(item.get('dpr2', ''))
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
                    f"{safe_val(item.get('day3', ''))}에는 {safe_val(item.get('dpr3', ''))}원, "
                    f"{safe_val(item.get('day4', ''))}에는 {safe_val(item.get('dpr4', ''))}원 이었습니다."
                )
                docs.append(doc)
    else:
        print("API 호출 실패:", response.status_code)
        print(response.text)
    return docs

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

# Milvus에서 문서 검색
def search_market_docs(query, top_k=3):
    # 전체 쿼리로 한 번만 검색
    all_results = []

    query_nouns = extract_keywords(query)

    # 미리 정의된 지역명 리스트와 명사 키워드를 비교하여 지역명만 추출
    predefined_locations = ['광주', '경산', '강동구', '태안', '성주', '창원', '용인', '울주', '순천', '경주', '양평', '울산광역', '영암', '김제', '고창', '전주', '하동', '제천', '홍성', '화성', '의왕', '담양', '진주', '사천', '남양주', '여수', '유성구', '정읍', '홍천', '남원', '동구', '달서구', '남해', '영동', '서구', '계룡', '고성', '고양', '평택', '남구', '울진', '나주', '전라북도', '익산', '부여', '청도', '합천', '포항', '봉화', '문경', '김해', '함양', '북구', '철원', '화순', '상주', '경북도', '안산', '청양', '충주', '김천', '영광', '성남', '전라남도', '달성', '인제', '천안', '제주', '원주', '가평', '완주', '제천시', '성주군', '고성군', '진천', '거창', '청주', '김포', '화성시', '완도', '함안', '옥천', '김해시', '해남', '무안', '예산', '금산', '강서구', '상당구', '송파구', '공도읍', '곡성', '울릉군', '서귀포', '정선', '평창', '양주', '포천', '진안', '세종']
    locations = [kw for kw in query_nouns if kw in predefined_locations or any(suffix in kw for suffix in ['시', '군', '구', '도'])]

    # 전체 쿼리 임베딩으로 검색
    query_vec = embedder.encode([query])[0]
    results = collection.search(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 20}},
        limit=200,
        output_fields=["text"],
    )

    if results and results[0]:
        all_results.extend([hit.entity.get("text") for hit in results[0]])

    all_results = list(dict.fromkeys(all_results))

    found_results = []
    # 검색된 문서 중에서 쿼리의 지역 키워드 또는 전체 쿼리가 포함된 결과만 선별
    for result_text in all_results:
        if any(loc in result_text for loc in locations) or any(q_part in result_text for q_part in query.split()):
            found_results.append(result_text)

    if not found_results:
        return []
    else:
        # 유사도 점수 계산 및 정렬 (기존 로직 유지)
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
    return {
        "실시간시세": kamis_results[:1],
        "판매처": sales_info_results
    }

# Groq LLM
class GroqLLM:
    def __init__(self, model="openai/gpt-oss-20b", api_keys=None):
        self.model = model
        self.api_keys = api_keys or [os.getenv(f"OPENAI_KEY{i}") for i in range(1, 4)]
        self.client = None

    def invoke(self, prompt: str):
        msg = ""
        for key in self.api_keys:
            if not key:
                continue
            try:
                self.client = Groq(api_key=key)
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
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
                msg = str(e)
                print(f"API key 실패: {e}")
                continue
        print("모든 키가 실패했습니다.")
        return f"LLM 호출 실패"

# 프롬프트 생성
def make_prompt(context, query):
    return (
        f"""
        [정보]
        {context}
        
        [질문]
        {query}
        
        [지시]
        - 반드시 [정보]에 나온 단위와 가격을 그대로 사용하세요.
        - [정보]에 없는 내용은 포함하지 마세요.
        - 답변은 한국어로, 간결하게 작성하세요.
        - 아래 순서대로 답변하세요:
            1. 품목과 등락율 (시세 정보가 없으면 "해당 작물에 대한 정보는 현재 없습니다."라고 답변 후 3번으로 넘어감)
            2. 당일, 1개월전, 1년전 가격 (정보 없으면 생략)
            3. 지역 매장 정보 및 주요 품목 ([질문]에서 물어본 주소와 일치하는 매장만 소개, 만약 판매처 정보가 없으면 "해당 지역에 판매점 정보가 없습니다."라고 답변)

        [예시]
        감자(20kg)의 가격은 어제보다 2.8%(1060원) 증가한 39,660원입니다. 1개월전에는 33,260원, 1년전에는 31,576원이었습니다. 해당 지역의 판매처는 충남 태안군 태안 로컬푸드 판매장(충남 태안군 남면 안면대로 1641 / 주요 품목: 채소, 과일, 서류) 등이 있습니다.
        """
    )

# LLM 호출
def ask_llm_groq(prompt, model="openai/gpt-oss-20b"):
    llm = GroqLLM(model=model, api_keys=groq_api_keys)
    return llm.invoke(prompt)

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

# LangGraph 노드 함수
def node_input_graph(state: GraphState) -> GraphState:
    query = input("작물 및 지역 정보를 입력하세요 (종료하려면 'exit'): ")
    if query.strip().lower() == "exit":
        state["exit"] = True
    else:
        state["query"] = query
        state["retry_count"] = 0 # 새로운 입력 시 재분석 카운트 초기화
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

    llm_prompt = make_prompt(context_str, query)
    pred_answer = ask_llm_groq(llm_prompt)
    
    state["pred_answer"] = pred_answer
    state["context_str_for_judge"] = context_str
    return state

def node_judge_recommendation_graph(state: GraphState) -> GraphState:
    pred_answer = state["pred_answer"]
    original_context_str = state["context_str_for_judge"]
    
    try:
        context_embedding = embedder.encode([original_context_str])[0].reshape(1, -1)
        answer_embedding = embedder.encode([pred_answer])[0].reshape(1, -1)
        
        similarity_score = cosine_similarity(context_embedding, answer_embedding)[0][0]
        
        ACCURACY_THRESHOLD = 0.8
        accuracy_ok = similarity_score > ACCURACY_THRESHOLD
        
        print(f"유사도 점수: {similarity_score:.4f}, 임계값: {ACCURACY_THRESHOLD}")
    except Exception as e:
        print(f"유사도 검사 중 오류 발생: {e}")
        accuracy_ok = False

    state["is_recommend_ok"] = accuracy_ok
    
    return state

def node_reanalyze_graph(state: GraphState) -> GraphState:
    state["retry_count"] += 1
    query = state["query"]
    print(f"재분석 {state['retry_count']}회차: 정보를 보완하여 재분석합니다.")
    results = hybrid_search(query, top_k=5)
    state["context"] = results
    return state

def node_output_graph(state: GraphState) -> GraphState:
    print("\n[최종 Agent 답변]")
    if state["retry_count"] >= 2 and not state["is_recommend_ok"]:
        print("해당 작물과 지역에 대한 시세 또는 판매처 정보가 없습니다. 혹시 다른 작물이나 지역을 찾아드릴까요?")
    else:
        print("\n--- Context (참고 정보) ---")
        print(state["context_str_for_judge"])
        print("---------------------------")
        print(state["pred_answer"])
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

graph.add_conditional_edges("judge_recommendation", judge_branch)
graph.add_edge("reanalyze", "llm_summarize")
graph.add_edge("output", END)

graph.set_entry_point("input")

# 실행 함수
def run_agent_langgraph():
    app = graph.compile()
    while True:
        try:
            state = app.invoke({"query": ""}) 
            if state.get("exit"):
                print("에이전트를 종료합니다.")
                break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            break

if __name__ == "__main__":
    embed_and_store_csv()
    collection.load()
    run_agent_langgraph()