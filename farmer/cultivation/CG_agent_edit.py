import os
import sys
import re
import pandas as pd
import asyncio
import threading
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict

# Langchain 및 LangGraph 관련 라이브러리
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus as LangChainMilvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from langchain.retrievers import EnsembleRetriever

from datasets import Dataset
# RAGAS 라이브러리
def evaluate_with_ragas(dataset, metrics):
    # 사용 직전에만 ragas import (지연 import)
    from ragas import evaluate
    return evaluate(dataset, metrics=metrics)

def get_ragas_metrics():
    # metrics도 내부에서 import
    from ragas.metrics import faithfulness, answer_relevancy, ContextUtilization, LLMContextPrecisionWithoutReference
    return faithfulness, answer_relevancy, ContextUtilization, LLMContextPrecisionWithoutReference


# --- 1. 환경 설정 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# (수정) 경고만 출력하고 플래그로 관리
USE_OPENAI = bool(OPENAI_API_KEY)
USE_TAVILY = bool(TAVILY_API_KEY)

if not USE_OPENAI:
    print("⚠️ OPENAI_API_KEY 미설정: LLM 호출 시 실패할 수 있습니다.")
if not USE_TAVILY:
    print("⚠️ TAVILY_API_KEY 미설정: 웹검색을 비활성화합니다.")

# Tavily 도구 생성 부분도 안전하게
tavily_tool = None
if USE_TAVILY:
    from langchain_tavily import TavilySearch
    tavily_tool = TavilySearch(max_results=5, api_key=TAVILY_API_KEY)
    
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME_INFO = "crop_info"
COLLECTION_NAME_GROW = "crop_grow"

# --- 2. LLM 및 프롬프트 설정 ---
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7
)

MULTI_CLASSIFY_PROMPT_TEMPLATE = """
당신은 사용자의 질문이 어떤 주제에 관한 것인지 분류하는 전문가입니다.

질문을 분석하여 다음 규칙에 따라 답변하세요.
- 질문에 '농작물' 재배 또는 관리 관련 내용이 있다면, 'crop_growth'를 포함하세요.
- 그 외의 모든 일반적인 질문이라면, 'general'를 포함하세요.
- 여러 주제가 포함된 경우, 쉼표(,)로 구분하여 답변하세요.
- 답변은 오직 주제 키워드만 포함해야 합니다.

질문: {question}
답변:
"""
multi_classify_prompt = ChatPromptTemplate.from_template(MULTI_CLASSIFY_PROMPT_TEMPLATE)

DB_AND_WEB_SEARCH_PROMPT_TEMPLATE = """
당신은 검색 전문가입니다.
다음 검색 결과들을 활용하여 사용자의 질문에 가장 정확하고 완전한 답변을 제공해 주세요.

# DB 검색 결과:
{db_context}

# 웹 검색 결과:
{web_search_results}

답변 규칙
1. **친절하고 자연스럽게**: 친근하고 명확한 문체로 작성해 주세요.
2. **정보의 출처 명시**: DB와 웹 검색 결과에 제시된 정보만을 사용하세요. 만약 질문에 대한 답변이 검색 결과에 없다면, '검색 결과에 해당 정보가 없습니다.'라고 명확하게 말해야 합니다.
3. **핵심 요약 및 정리**: 여러 검색 결과에서 중복되는 핵심 내용들을 종합하여 간결하게 요약해 주세요.
4. **구체적이고 상세하게**: 답변은 가능한 한 구체적인 정보(예: 날짜, 숫자, 기관명 등)를 포함하여 작성해 주세요.
5. **한글로만 답변**: 모든 답변은 한글로만 제공해야 합니다.
6. **내부 DB 정보를 우선적으로 활용**: 내부 DB에 관련된 내용이 있다면 이를 우선적으로 사용하고, 부족한 부분을 웹 검색 결과로 보충하세요.

질문: {question}
답변:
"""
db_and_web_search_prompt = ChatPromptTemplate.from_template(DB_AND_WEB_SEARCH_PROMPT_TEMPLATE)

VALIDATION_PROMPT = """
주어진 맥락만 사용하여 다음 질문에 대한 완전하고 상세한 답변을 생성할 수 있는지 여부를 '네' 또는 '아니오'로만 답변하세요.
질문: {question}
맥락: {db_context}
답변:
"""

tavily_tool = TavilyClient(api_key=TAVILY_API_KEY)

# 전역 retriever 변수 (직렬화 문제 해결을 위해)
_global_retriever = None

# --- 3. LangGraph 상태 정의 ---
class GraphState(TypedDict):
    question: Optional[str]
    # retriever: Optional[EnsembleRetriever]  # 직렬화 문제로 제거 - 전역 변수로 관리
    answer: Optional[str]
    topics: Optional[List[str]]
    db_context: Optional[str]
    web_sources: Optional[List[Dict[str, Any]]]
    db_sources: Optional[List[Dict[str, Any]]]
    attempts: int
    is_good_answer: Optional[str]

# --- 4. 핵심 기능 함수 정의 ---
def get_retriever() -> EnsembleRetriever:
    """전역 retriever를 가져오거나 새로 생성합니다."""
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = create_retriever()
    return _global_retriever

def create_retriever() -> EnsembleRetriever:
    """두 개의 Milvus 컬렉션에 연결하여 EnsembleRetriever를 생성합니다."""
    print("---기능: Milvus 컬렉션 연결 및 EnsembleRetriever 생성 시작---")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

        vectorstore_info = LangChainMilvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME_INFO,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            consistency_level="Bounded"
        )
        print(f"✅ '{COLLECTION_NAME_INFO}' 컬렉션에 연결했습니다.")

        vectorstore_grow = LangChainMilvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME_GROW,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            consistency_level="Bounded"
        )
        print(f"✅ '{COLLECTION_NAME_GROW}' 컬렉션에 연결했습니다.")

        retriever_info = vectorstore_info.as_retriever(search_kwargs={"k": 3})
        retriever_grow = vectorstore_grow.as_retriever(search_kwargs={"k": 3})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_info, retriever_grow],
            weights=[0.5, 0.5]
        )
        
        print("✅ EnsembleRetriever가 성공적으로 생성되었습니다.")
        return ensemble_retriever
    except Exception as e:
        print(f"Milvus 연결 또는 EnsembleRetriever 생성 중 오류 발생: {e}")
        raise

def retrieve_relevant_chunks(retriever: EnsembleRetriever, question: str) -> Dict[str, Any]:
    """EnsembleRetriever를 사용하여 두 컬렉션에서 관련 문서를 검색합니다."""
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    db_sources = [{"source": doc.metadata.get('source'), "page": doc.metadata.get('page'), "content": doc.page_content} for doc in docs]
    print(f"검색된 총 청크 수: {len(docs)}개")
    return {"context": context, "db_sources": db_sources}

# --- 5. LangGraph 노드 함수 정의 ---
def load_and_merge_dbs_node(state: GraphState) -> Dict[str, Any]:
    """Milvus의 EnsembleRetriever를 초기화합니다."""
    print("\n---노드: Milvus EnsembleRetriever 초기화 실행---")
    retriever = get_retriever()  # 전역 retriever 사용
    print("Milvus EnsembleRetriever 로드 완료.\n")
    return state  # retriever를 상태에 저장하지 않음

def multi_classify_question_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 복합 질문 분류 실행---")
    question = state["question"]
    chain = multi_classify_prompt | llm | StrOutputParser()
    classification_str = chain.invoke({"question": question}).strip()
    topics = [topic.strip() for topic in classification_str.split(',') if topic.strip()]
    print(f"질문이 다음 주제들로 분류되었습니다: {topics}")
    return {**state, "topics": topics}
    
def process_topics_and_retrieve_content_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 주제별 정보 검색 및 통합 실행---")
    question = state["question"]
    topics = state.get("topics", [])
    
    db_context = ""
    db_sources = []
    web_sources = []

    # 1. DB 검색 (농작물 재배 관련)
    if "crop_growth" in topics:
        print("🔍 '농작물 재배' 주제 관련 DB 정보 검색 중...")
        retriever = get_retriever()  # 전역 retriever 사용
        retrieval_result = retrieve_relevant_chunks(retriever, question)
        db_context = retrieval_result["context"]
        db_sources = retrieval_result["db_sources"]
        print("✅ DB 검색 완료.")

        is_sufficient = bool(db_context.strip())
        
        if is_sufficient:
            print("🔍 DB 내용 충분성 검증 중...")
            validation_chain = ChatPromptTemplate.from_template(VALIDATION_PROMPT) | llm | StrOutputParser()
            validation_result = validation_chain.invoke({"question": question, "db_context": db_context})
            if "아니오" in validation_result.strip():
                is_sufficient = False
                print("❗ DB 내용이 불충분하다고 판단되었습니다. 웹 검색을 추가합니다.")
        
        if not is_sufficient:
            print("🌐 웹 검색으로 정보 보충 중...")
            try:
                # TavilyClient 사용 - search 메서드 호출
                response = tavily_tool.search(query=question, max_results=5)
                
                # TavilyClient 응답 형식: {"results": [{"url": "", "content": "", "title": ""}]}
                if isinstance(response, dict) and "results" in response:
                    results = response["results"]
                    web_sources = []
                    for res in results:
                        if isinstance(res, dict):
                            web_sources.append({
                                "url": res.get("url", "N/A"), 
                                "content": res.get("content", res.get("snippet", ""))
                            })
                else:
                    # 예상치 못한 응답 형식
                    print(f"⚠️ 예상치 못한 Tavily 응답 형식: {type(response)}")
                    web_sources = [{"url": "N/A", "content": "웹 검색 응답 형식 오류"}]
                    
            except Exception as e:
                print(f"⚠️ Tavily 검색 오류: {e}")
                web_sources = [{"url": "N/A", "content": f"웹 검색 실패 - {str(e)}"}]
            
            print("✅ 웹 검색 완료.")

    # 2. 'general' 주제에 대한 웹 검색 (기존 로직 유지)
    if "general" in topics:
        print("🌐 '일반' 주제에 대한 웹 검색 중...")
        try:
            # TavilyClient 사용 - search 메서드 호출
            response = tavily_tool.search(query=question, max_results=5)
            
            # TavilyClient 응답 형식: {"results": [{"url": "", "content": "", "title": ""}]}
            if isinstance(response, dict) and "results" in response:
                results = response["results"]
                for res in results:
                    if isinstance(res, dict):
                        web_sources.append({
                            "url": res.get("url", "N/A"), 
                            "content": res.get("content", res.get("snippet", ""))
                        })
            else:
                # 예상치 못한 응답 형식
                print(f"⚠️ 예상치 못한 Tavily 응답 형식: {type(response)}")
                web_sources.extend([{"url": "N/A", "content": "웹 검색 응답 형식 오류"}])
                
        except Exception as e:
            print(f"⚠️ Tavily 검색 오류: {e}")
            web_sources.extend([{"url": "N/A", "content": f"웹 검색 실패 - {str(e)}"}])
        
        print("✅ 웹 검색 완료.")
    
    return {**state, "db_context": db_context, "db_sources": db_sources, "web_sources": web_sources}

def generate_final_answer_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 최종 답변 생성 실행---")
    question = state["question"]
    db_context = state.get("db_context", "내부 DB에서 검색된 정보가 없습니다.")
    web_search_results = "\n".join([str(res) for res in state.get("web_sources", [])])
    
    if not web_search_results:
        web_search_results = "웹 검색 결과가 없습니다."
        
    inputs = {
        "question": question,
        "db_context": db_context,
        "web_search_results": web_search_results
    }
    
    final_chain = db_and_web_search_prompt | llm | StrOutputParser()
    answer = final_chain.invoke(inputs)
    return {**state, "answer": answer}

def validate_and_regenerate_node(state: GraphState) -> Dict[str, Any]:
    print("\n---노드: 답변 품질 검증 실행---")
    question = state["question"]
    answer = state["answer"]
    db_sources = state.get('db_sources', [])
    web_sources = state.get('web_sources', [])
    
    contexts = [src.get('content') for src in db_sources] + [src.get('content') for src in web_sources]
    
    ragas_scores = run_ragas_evaluation(question, answer, contexts)
    
    RELEVANCY_THRESHOLD = 0.4
    FAITHFULNESS_THRESHOLD = 0.4
    
    answer_relevancy_score = ragas_scores['answer_relevancy'].iloc[0]
    faithfulness_score = ragas_scores['faithfulness'].iloc[0]
    
    attempts = state.get('attempts', 0)
    
    is_good_answer = (answer_relevancy_score >= RELEVANCY_THRESHOLD and faithfulness_score >= FAITHFULNESS_THRESHOLD)

    if is_good_answer:
        print("✅ 답변 품질이 양호합니다. 최종 답변을 제공합니다.")
        return {"is_good_answer": "yes", "attempts": attempts}
    else:
        print(f"❗ 답변 품질이 낮습니다. 재시도합니다. (현재 시도 횟수: {attempts + 1})")
        return {"is_good_answer": "no", "attempts": attempts + 1}

# --- 6. LangGraph 워크플로우 빌드 및 컴파일 ---
def build_initial_setup_graph():
    """초기 문서 로딩 및 벡터스토어 구축을 위한 그래프를 빌드합니다."""
    # 더 이상 필요하지 않음 - 전역 retriever 사용
    pass

def build_query_graph():
    """질문 분류, RAG, 웹 검색을 통합한 메인 질의 그래프를 빌드합니다."""
    query_builder = StateGraph(GraphState)
    
    query_builder.add_node("multi_classify_question", multi_classify_question_node)
    query_builder.add_node("process_topics_and_retrieve_content", process_topics_and_retrieve_content_node)
    query_builder.add_node("generate_final_answer", generate_final_answer_node)
    query_builder.add_node("validate_and_regenerate", validate_and_regenerate_node)

    query_builder.set_entry_point("multi_classify_question")
    query_builder.add_edge("multi_classify_question", "process_topics_and_retrieve_content")
    query_builder.add_edge("process_topics_and_retrieve_content", "generate_final_answer")
    query_builder.add_edge("generate_final_answer", "validate_and_regenerate")
    
    query_builder.add_conditional_edges(
        "validate_and_regenerate",
        lambda state: state.get("is_good_answer") == "yes",  # 조건문 수정
        {
            True: END,
            False: "process_topics_and_retrieve_content"
        }
    )

    return query_builder.compile()

# --- RAGAS 평가를 위한 LLM 및 임베딩 모델을 전역으로 정의 ---
ragas_llm = ChatOpenAI(model_name="gpt-4o-mini")
ragas_embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask"
)

# --- RAGAS 평가 함수 추가 (asyncio.gather 병렬 처리) ---
def run_ragas_evaluation(question: str, answer: str, contexts: List[str]):
    """
    주어진 질문, 답변, 맥락으로 RAGAS 평가를 실행합니다. (asyncio.gather 병렬 처리)
    """
    print("\n--- RAGAS 자동 평가 시작 ---")
    faithfulness, answer_relevancy , ContextUtilization, LLMContextPrecisionWithoutReference = get_ragas_metrics()
    data = {
        'question': [question],
        'answer': [answer],
        'contexts': [contexts]
    }
    dataset = Dataset.from_dict(data)

    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
        ContextUtilization(),
        LLMContextPrecisionWithoutReference()
    ]

    try:
        result = evaluate_with_ragas(
            dataset=dataset,
            metrics=metrics_to_evaluate,
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        # 🚀 asyncio.gather를 사용한 직접 병렬 처리
        async def evaluate_faithfulness():
            try:
                from ragas import SingleTurnSample
                from ragas.llms import LangchainLLMWrapper
                from ragas.metrics import Faithfulness
                
                faithfulness_scorer = Faithfulness(llm=LangchainLLMWrapper(ragas_llm))
                faithfulness_sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts
                )
                score = await faithfulness_scorer.single_turn_ascore(faithfulness_sample)
                return ("faithfulness", float(score) if score is not None else 0.0)
            except Exception as e:
                print(f"   - ⚠️ Faithfulness 평가 실패: {e}")
                return ("faithfulness", 0.0)
        
        async def evaluate_answer_relevancy():
            try:
                from ragas import SingleTurnSample
                from ragas.llms import LangchainLLMWrapper
                from ragas.embeddings import LangchainEmbeddingsWrapper
                from ragas.metrics import ResponseRelevancy
                
                answer_relevancy_scorer = ResponseRelevancy(
                    llm=LangchainLLMWrapper(ragas_llm), 
                    embeddings=LangchainEmbeddingsWrapper(ragas_embeddings)
                )
                relevancy_sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts
                )
                score = await answer_relevancy_scorer.single_turn_ascore(relevancy_sample)
                return ("answer_relevancy", float(score) if score is not None else 0.0)
            except Exception as e:
                print(f"   - ⚠️ Answer Relevancy 평가 실패: {e}")
                return ("answer_relevancy", 0.0)
        
        async def evaluate_context_utilization():
            try:
                from ragas import SingleTurnSample
                from ragas.llms import LangchainLLMWrapper
                from ragas.metrics import ContextUtilization
                
                context_utilization_scorer = ContextUtilization(llm=LangchainLLMWrapper(ragas_llm))
                context_sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts
                )
                score = await context_utilization_scorer.single_turn_ascore(context_sample)
                return ("context_utilization", float(score) if score is not None else 0.0)
            except Exception as e:
                print(f"   - ⚠️ Context Utilization 평가 실패: {e}")
                return ("context_utilization", 0.0)
        
        async def evaluate_context_precision():
            try:
                from ragas import SingleTurnSample
                from ragas.llms import LangchainLLMWrapper
                
                context_precision_scorer = LLMContextPrecisionWithoutReference(llm=LangchainLLMWrapper(ragas_llm))
                context_sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts
                )
                score = await context_precision_scorer.single_turn_ascore(context_sample)
                return ("context_precision", float(score) if score is not None else 0.0)
            except Exception as e:
                print(f"   - ⚠️ Context Precision 평가 실패: {e}")
                return ("context_precision", 0.0)
        
        # 스레드 격리로 4개 모두 병렬 실행
        def run_parallel_ragas_in_thread():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    # 4개 평가를 동시에 실행 (진짜 병렬 처리)
                    results = new_loop.run_until_complete(
                        asyncio.gather(
                            evaluate_faithfulness(),
                            evaluate_answer_relevancy(),
                            evaluate_context_utilization(),
                            evaluate_context_precision(),
                            return_exceptions=True
                        )
                    )
                    
                    # 결과 수집
                    scores = {}
                    for result in results:
                        if isinstance(result, Exception):
                            print(f"   - ⚠️ RAGAS 평가 중 예외 발생: {result}")
                            continue
                        metric_name, score = result
                        scores[metric_name] = score
                    
                    return scores
                finally:
                    new_loop.close()
                    asyncio.set_event_loop(None)
            except Exception as e:
                print(f"   - ⚠️ 스레드 내 병렬 RAGAS 평가 실패: {e}")
                return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_utilization": 0.0, "context_precision": 0.0}
        
        result_container = [None]
        def thread_target():
            result_container[0] = run_parallel_ragas_in_thread()
        
        print("   - 🔥 Faithfulness, Answer Relevancy, Context Utilization & Context Precision 4개 병렬 평가 시작!")
        thread = threading.Thread(target=thread_target)
        thread.start()
        thread.join()
        print("   - ✅ 4개 병렬 평가 완료!")
        
        scores = result_container[0]
        if scores is None:
            raise Exception("RAGAS 평가 실패")
        
        print("✅ RAGAS 평가 완료.")
        
        print("\n--- RAGAS 평가 점수 ---")
        print(f"faithfulness: {scores.get('faithfulness', 0.0):.4f}")
        print(f"answer_relevancy: {scores.get('answer_relevancy', 0.0):.4f}")
        print(f"context_utilization: {scores.get('context_utilization', 0.0):.4f}")
        print(f"context_precision: {scores.get('context_precision', 0.0):.4f}")
        
        # DataFrame 형태로 반환 (기존 호환성 유지)
        result_df = pd.DataFrame({
            'faithfulness': [scores.get('faithfulness', 0.0)],
            'answer_relevancy': [scores.get('answer_relevancy', 0.0)],
            'context_utilization': [scores.get('context_utilization', 0.0)],
            'context_precision': [scores.get('context_precision', 0.0)]
        })
        
        print("\n--- 전체 평가 데이터프레임 ---")
        print(result_df)
        
        return result_df

    except Exception as e:
        print(f"❌ RAGAS 평가 중 오류가 발생했습니다: {e}")
        # 오류 발생 시에도 일관된 DataFrame 구조를 반환
        return pd.DataFrame({'faithfulness': [0.0], 'answer_relevancy': [0.0], 'context_utilization': [0.0], 'context_precision': [0.0]})

# --- 7. OchestratorTest.py와 호환되는 run 함수 ---
async def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    OchestratorTest.py에서 호출되는 메인 실행 함수 (비동기)
    
    Args:
        state: OchestratorTest.py에서 전달받은 상태 딕셔너리
               - query: 사용자 질문 (필수)
               - 기타 필요한 상태 정보들
    
    Returns:
        dict: 실행 결과
            - agent_answer: 최종 응답
            - status: 실행 상태
            - error: 오류 정보 (있는 경우)
    """
    try:
        # 입력 검증
        if not state or not state.get("query"):
            return {
                "agent_answer": "질문이 제공되지 않았습니다.",
                "status": "error",
                "error": "query 필드가 없습니다."
            }
        
        query = state["query"]
        print(f"\n=== 🌱 작물재배_agent 실행 시작 ===")
        print(f"질문: {query}")
        
        # Milvus EnsembleRetriever 초기화
        print("Milvus EnsembleRetriever 초기화 중...")
        try:
            retriever = get_retriever()  # 전역 retriever 사용
            print("✅ Milvus EnsembleRetriever 초기화 완료")
        except Exception as e:
            return {
                "agent_answer": f"데이터베이스 연결에 실패했습니다: {e}",
                "status": "error",
                "error": f"Milvus EnsembleRetriever 초기화 실패: {e}"
            }
        
        # RAG 애플리케이션 빌드
        rag_app = build_query_graph()
        
        # 답변 생성 (최대 3회 시도)
        MAX_ATTEMPTS = 3
        attempts = 0
        final_response = None
        
        while attempts < MAX_ATTEMPTS:
            print(f"답변 생성 중... (시도 {attempts + 1}/{MAX_ATTEMPTS})")
            try:
                current_state = {
                    "question": query, 
                    "attempts": attempts
                }
                final_state = rag_app.invoke(current_state)
                
                # 최종 답변이 존재하면 루프 종료
                if final_state.get('answer'):
                    final_response = final_state['answer']
                    print("✅ 답변 생성 완료")
                    break
                else:
                    attempts += 1
                    print(f"❗ 답변 품질이 낮습니다. 재시도합니다.")
                    
            except Exception as e:
                print(f"❌ 답변 생성 중 오류: {e}")
                attempts += 1
                if attempts >= MAX_ATTEMPTS:
                    final_response = f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {e}"
        
        # 최종 응답 반환
        if final_response:
            print(f"=== 🎯 작물재배_agent 실행 완료 ===")
            return {
                "agent_answer": final_response,
                "status": "success",
                "error": None
            }
        else:
            return {
                "agent_answer": "죄송합니다. 답변을 생성하기 어렵습니다. 다시 시도해주세요.",
                "status": "error",
                "error": "최대 시도 횟수 초과"
            }
            
    except Exception as e:
        print(f"❌ 작물재배_agent 실행 중 치명적 오류: {e}")
        return {
            "agent_answer": f"작물재배_agent 실행 중 오류가 발생했습니다: {e}",
            "status": "error",
            "error": str(e)
        }

# --- 8. 메인 실행 로직 (독립 실행용) ---
if __name__ == "__main__":
    print("🌱 농작물 챗봇 에이전트 시작...")
    print("--------------------------------------------------")
    
    print("챗봇 시스템을 준비하는 중입니다... (Milvus EnsembleRetriever 초기화)")
    try:
        retriever = get_retriever()  # 전역 retriever 초기화
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        exit()
        
    print("챗봇 시스템 준비 완료!\n")
    
    rag_app = build_query_graph()

    print("이제 질문을 입력하세요. (종료하려면 'exit' 또는 'quit' 입력)")
    print("--------------------------------------------------")

    while True:
        prompt = input("질문을 입력하세요: ")
        if prompt.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다.")
            break
        
        MAX_ATTEMPTS = 3
        attempts = 0
        final_response = None
        
        while attempts < MAX_ATTEMPTS:
            print("답변을 생성하는 중...")
            try:
                current_state = {"question": prompt, "attempts": attempts}
                final_state = rag_app.invoke(current_state)
                
                # 최종 답변이 존재하면 바로 루프를 종료합니다.
                if final_state.get('answer'):
                    final_response = final_state['answer']
                    print("\n최종 답변이 확정되었습니다.\n")
                    break
                else:
                    attempts += 1
                    print(f"❗ 답변 품질이 낮거나 생성에 실패했습니다. 재시도합니다. (현재 시도 횟수: {attempts}/{MAX_ATTEMPTS})")
            except Exception as e:
                print(f"오류가 발생했습니다: {e}")
                final_response = "죄송합니다. 오류가 발생하여 답변을 생성할 수 없습니다."
                attempts = MAX_ATTEMPTS
        
        print("\n------------------- 답변 -------------------")
        if final_response:
            print(final_response)
        else:
            print("죄송합니다. 답변을 생성하기 어렵습니다. 다시 시도해주세요.")
        print("-------------------------------------------\n")

        if final_state:
            db_sources = final_state.get('db_sources', [])
            web_sources = final_state.get('web_sources', [])
            
            if db_sources:
                print("--- 참고한 DB 내용 ---")
                for i, source in enumerate(db_sources, 1):
                    file_name = os.path.basename(source.get('source', '')).rsplit('.', 1)[0]
                    page_num = source.get('page')
                    print(f"**[{i}]** 출처: {file_name}", end="")
                    if page_num is not None:
                        print(f", 페이지: {page_num + 1}", end="")
                    print(f"\n내용: {source.get('content', '내용 없음')[:100]}...\n")
            
            if web_sources:
                print("--- 참고한 웹 검색 결과 ---")
                for i, source in enumerate(web_sources, 1):
                    print(f"**[{i}]** URL: {source.get('url', 'URL 없음')}")
                    print(f"내용: {source.get('content', '내용 없음')[:100]}...\n")

            print("-------------------------------------------\n")