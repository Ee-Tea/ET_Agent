# 주의 무시
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 임베딩 모델
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

# Milvus 연결 설정
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, list_collections
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
if collection_name not in list_collections():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
    ]
    schema = CollectionSchema(fields, "시장 가격 문서 컬렉션")
    collection = Collection(collection_name, schema)
else:
    collection = Collection(collection_name)

# api 요청 후 DB에 저장
def fetch_and_store_api_data():
    url = "http://www.kamis.or.kr/service/price/xml.do?action=dailySalesList"
    params = {
        "p_cert_key": api_key,
        "p_cert_id": api_id,
        "p_returntype": "json"
    }
    response = requests.get(url, params=params)
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

        docs = []
        for item in items:
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
                
                change_str = "변동 없음"
                if str(direction_raw) == "0":
                    change_str = f"{value_raw}%({diff}원) 감소"
                elif str(direction_raw) == "1":
                    change_str = f"{value_raw}%({diff}원) 증가"
                
                doc = (
                    f"{safe_val(item.get('item_name', ''))} ({safe_val(item.get('unit', ''))}) = "
                    f"{change_str}, "
                    f"{safe_val(item.get('day1', ''))}: {dpr1}원, "
                    f"{safe_val(item.get('day3', ''))}: {safe_val(item.get('dpr3', ''))}원, "
                    f"{safe_val(item.get('day4', ''))}: {safe_val(item.get('dpr4', ''))}원"
                )
                docs.append(doc)

        if docs:
            embeddings = embedder.encode(docs)
            collection.insert([embeddings.tolist(), docs], fields=["embedding", "text"])
            if not collection.has_index():
                index_params = {
                    "metric_type": "IP",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                collection.create_index(field_name="embedding", index_params=index_params)
    else:
        print("API 호출 실패:", response.status_code)
        print(response.text)

# CSV 파일에서 정보 검색
def search_csv(query, csv_path="data/info_20240812.csv", top_k=3):
    try:
        df = pd.read_csv(csv_path, encoding="euc-kr")
        df['품목'] = df['품목'].fillna("정보 없음")
        okt = Okt()
        keywords = set(okt.nouns(query))
        location_keywords = [kw for kw in keywords if any(kw in str(addr) for addr in df['주소'])]
        if location_keywords:
            location_mask = df['주소'].str.contains('|'.join(location_keywords), na=False)
            results = df[location_mask].copy()
        else:
            return ["해당 지역에 판매점 정보가 없습니다."]
        if not results.empty and len(keywords) > 0:
            results['점수'] = 0
            for kw in keywords:
                results['점수'] += results['품목'].str.contains(kw, na=False).astype(int) * 2
                results['점수'] += results['주소'].str.contains(kw, na=False).astype(int) * 1
            results = results.sort_values(by='점수', ascending=False, kind='stable')
        if not results.empty:
            return [
                f"판매장 이름: {row['판매장 이름']}, 주소: {row['주소']}, 주요 품목: {row['품목']}"
                for _, row in results.head(top_k).iterrows()
            ]
        else:
            return []
    except Exception as e:
        print("CSV 검색 오류:", e)
        return ["CSV 검색 오류가 발생했습니다."]

# Milvus에서 문서 검색
def search_market_docs(query, top_k=1):
    queries = [q.strip() for q in query.split('/')] if '/' in query else [query]
    all_results = []
    for q in queries:
        query_vec = embedder.encode([q])[0]
        results = collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 20}},
            limit=top_k * 200,
            output_fields=["text"]
        )
        if results and results[0]:
            all_results.extend([hit.entity.get("text") for hit in results[0]])
    all_results = list(dict.fromkeys(all_results))
    def overlap_score(result_text):
        item_part = result_text.split('(')[0].strip()
        item_names = [x.strip() for x in item_part.split('/')]
        okt = Okt()
        query_strip = query.strip()
        query_nouns = set(okt.nouns(query_strip))
        current_score = 0
        if query_strip in item_names:
            current_score += 10000
        for name in item_names:
            name_nouns = set(okt.nouns(name))
            if query_nouns.issubset(name_nouns):
                current_score += 1000
            current_score += len(query_nouns.intersection(name_nouns)) * 100
        for name in item_names:
            if any(qn in name for qn in query_nouns):
                if not any(qn in okt.nouns(name) for qn in query_nouns):
                    current_score += 1
        return current_score
    all_results.sort(key=overlap_score, reverse=True)
    keywords = extract_keywords(query)
    found = False
    for result_text in all_results:
        item_part = result_text.split('(')[0].strip()
        if any(kw in item_part for kw in keywords):
            found = True
            break
    if not all_results or not found:
        return ["해당 작물에 대한 정보는 현재 없습니다."]
    else:
        return all_results[:top_k]

# 키워드 추출
def extract_keywords(query):
    okt = Okt()
    return okt.nouns(query)

# 하이브리드 검색
def hybrid_search(query, top_k=3):
    milvus_results = search_market_docs(query, 1)
    csv_results = search_csv(query, top_k=top_k)
    return {
        "실시간시세": milvus_results,
        "판매처": csv_results
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
        - 실시간 시세 정보가 없으면 '해당 작물에 대한 정보는 현재 없습니다.'라고 답변하세요.
        - 답변은 한국어로, 간결하게 작성하세요.
        - 아래 순서대로 답변하세요:
            1. 품목(단위)의 가격 등락율 (없으면 '변동 없음')
            2. 당일, 1개월전, 1년전 가격 (정보 없으면 생략)
            3. 지역 매장 정보 및 주요 품목 ([질문]에서 물어본 주소와 일치하는 매장만 소개)

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

    no_info_keywords = ["해당 작물에 대한 정보는 현재 없습니다.", "해당 지역에 판매점 정보가 없습니다."]
    if not accuracy_ok or any(kw in pred_answer for kw in no_info_keywords):
        state["is_recommend_ok"] = False
    else:
        state["is_recommend_ok"] = True
    
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
        print("현재 정보로는 답변을 생성하기 어렵습니다. 지역과 작물명을 다시 작성해서 질문해주시겠어요?")
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
    fetch_and_store_api_data()
    collection.load()
    run_agent_langgraph()