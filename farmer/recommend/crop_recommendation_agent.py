import os
import re
from pathlib import Path
import json
import time
import argparse
import unicodedata
import logging
import asyncio
import threading
from typing import List, Dict, Any, Optional, Tuple, TypedDict

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
import numpy as np
import math

from datasets import Dataset
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy, Faithfulness, LLMContextPrecisionWithoutReference
from ragas.llms import LangchainLLMWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import RateLimitError, APITimeoutError


load_dotenv(find_dotenv())

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "crop_info")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env에 설정되어야 합니다.")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus as MilvusVectorStore
from tavily import TavilyClient
from langchain_openai import ChatOpenAI

# =========[ 전역 변수 ]=========
_vectorstore = None
from langgraph.graph import StateGraph, END
from pymilvus import connections

RAG_PROMPT_TMPL = """
당신은 대한민국 농업 작물 추천 전문가입니다.
아래 '문맥'을 참고하여 사용자의 질문에 맞는 작물을 추천하고 재배 방법을 안내해주세요.

[문맥]
{context}

규칙:
- 문맥에 있는 정보만 사용하여 답변하세요.
- 추천 작물과 그 이유를 명확하게 설명하세요.
- 재배 조건, 시기, 관리 방법 등을 구체적으로 안내하세요.
- 한글로만 작성하고, 단계별로 설명하세요.
- **재배 방법은 각 단계마다 한 줄씩 띄워서 작성하세요.**
- 문맥에 근거가 없으면 "주어진 정보로는 답변할 수 없습니다."라고 답변하세요.

질문: {question}
답변:
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TMPL)

WEB_PROMPT_TMPL = """
당신은 대한민국 농업 작물 추천 전문가입니다.
아래 '웹 검색 결과'를 바탕으로 사용자의 질문에 맞는 작물을 추천하고 재배 정보를 종합하여 안내해주세요.

[웹 검색 결과]
{search_results}

규칙:
- 검색 결과를 바탕으로 작물 추천과 재배 방법을 종합하여 답변하세요.
- 추천 작물의 선택 이유와 장점을 명확하게 설명하세요.
- 재배 조건, 시기, 관리 방법 등을 단계별로 정리하세요.
- **재배 방법은 각 단계마다 한 줄씩 띄워서 작성하세요.**
- 검색 결과로 답변이 불가능할 때만 "관련 정보를 찾을 수 없습니다."라고 답변하세요.
- 모든 답변은 반드시 한국어로 작성하세요.

🟢 질문: {question}
✨ 답변:
"""
web_prompt = ChatPromptTemplate.from_template(WEB_PROMPT_TMPL)

class GraphState(TypedDict, total=False):
    question: Optional[str]
    db_context: Optional[str]
    web_context: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    answer_draft: Optional[str]
    log_file: Optional[str]
    answer_source: Optional[str]
    ragas_score: Optional[float]
    retry_count: int
    is_retrieval_sufficient: bool
    is_answer_sufficient: bool
    original_docs: List[str]
    web_contexts: List[str]
    ragas_details: Optional[Dict[str, Any]]


embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)
llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)



def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[\.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]

def build_extractive_reference_from_contexts(contexts: List[str], question: str, embedder: HuggingFaceEmbeddings, top_k: int = 5, max_chars: int = 1800) -> str:
    """
    컨텍스트 원문에서 문장들을 추출하여, 질문-문장 임베딩 유사도 상위 top_k 문장으로 reference를 구성.
    - LLM 생성/요약 없음 (추출식, extractive)
    """
    sents: List[str] = []
    for c in contexts:
        sents.extend(_split_sentences(c))

    if len(sents) > 2000:
        sents = sents[:2000]

    if not sents:
        return ""

    try:
        q_emb = embedder.embed_query(question)
        s_embs = embedder.embed_documents(sents)
        q = np.array(q_emb, dtype=np.float32)
        S = np.array(s_embs, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        S_norm = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
        scores = S_norm @ q_norm
        idx = np.argsort(-scores)[:max(top_k, 1)]
        picked = [sents[i] for i in idx]
        reference = " ".join(picked)
        return reference[:max_chars]
    except Exception as e:
        print(f"Reference 생성 오류: {e}")
        return " ".join(contexts)[:max_chars]

def _ragas_overall(result_obj: Any, metric_name: str) -> Optional[float]:
    try:
        val = None
        
        # 1. RAGAS 0.3.x _scores_dict 속성에서 직접 접근 (가장 일반적)
        if hasattr(result_obj, "_scores_dict") and isinstance(result_obj._scores_dict, dict):
            val = result_obj._scores_dict.get(metric_name)
            if val is not None:
                # 리스트인 경우 첫 번째 값 사용
                if isinstance(val, list) and len(val) > 0:
                    val = val[0]
                # JSON 문자열인 경우 파싱 시도
                elif isinstance(val, str) and val.startswith('{'):
                    try:
                        import json
                        json_data = json.loads(val)
                        if "statements" in json_data and isinstance(json_data["statements"], list):
                            # verdict 값들의 평균 계산
                            verdicts = []
                            for stmt in json_data["statements"]:
                                if isinstance(stmt, dict) and "verdict" in stmt:
                                    verdicts.append(float(stmt["verdict"]))
                            if verdicts:
                                val = sum(verdicts) / len(verdicts)
                                print(f"   - ✅ {metric_name}: {val:.4f} (JSON verdict 평균)")
                                return val
                    except:
                        pass
                val = float(val)
                if val == val:  # NaN 체크
                    print(f"   - ✅ {metric_name}: {val:.4f} (_scores_dict)")
                    return val
        
        # 2. RAGAS 0.3.x scores 속성에서 직접 접근
        if hasattr(result_obj, "scores") and hasattr(result_obj.scores, metric_name):
            val = getattr(result_obj.scores, metric_name)
            if val is not None:
                # JSON 문자열인 경우 파싱 시도
                if isinstance(val, str) and val.startswith('{'):
                    try:
                        import json
                        json_data = json.loads(val)
                        if "statements" in json_data and isinstance(json_data["statements"], list):
                            # verdict 값들의 평균 계산
                            verdicts = []
                            for stmt in json_data["statements"]:
                                if isinstance(stmt, dict) and "verdict" in stmt:
                                    verdicts.append(float(stmt["verdict"]))
                            if verdicts:
                                val = sum(verdicts) / len(verdicts)
                                print(f"   - ✅ {metric_name}: {val:.4f} (scores JSON verdict 평균)")
                                return val
                    except:
                        pass
                val = float(val)
                if val == val:  # NaN 체크
                    print(f"   - ✅ {metric_name}: {val:.4f} (scores 속성)")
                    return val
        
        # 3. to_dict() 시도
        if hasattr(result_obj, "to_dict"):
            d = result_obj.to_dict()
            if isinstance(d, dict):
                # scores 딕셔너리 내부 확인 (RAGAS 0.3.x)
                if "scores" in d and isinstance(d["scores"], dict):
                    val = d["scores"].get(metric_name)
                    if val is not None:
                        val = float(val)
                        if val == val:  # NaN 체크
                            print(f"   - ✅ {metric_name}: {val:.4f} (to_dict scores)")
                            return val
                
                # overall 딕셔너리 내부 확인 (구버전)
                if "overall" in d and isinstance(d["overall"], dict):
                    val = d["overall"].get(metric_name)
                    if val is not None:
                        val = float(val)
                        if val == val:  # NaN 체크
                            print(f"   - ✅ {metric_name}: {val:.4f} (to_dict overall)")
                            return val
                
                # 직접 키 접근
                if metric_name in d:
                    val = d[metric_name]
                    if val is not None:
                        val = float(val)
                        if val == val:  # NaN 체크
                            print(f"   - ✅ {metric_name}: {val:.4f} (to_dict 직접)")
                            return val
        
        # 4. __dict__ 시도
        if hasattr(result_obj, "__dict__"):
            d = result_obj.__dict__
            # _scores_dict 딕셔너리 내부 확인 (RAGAS 0.3.x)
            if "_scores_dict" in d and isinstance(d["_scores_dict"], dict):
                val = d["_scores_dict"].get(metric_name)
                if val is not None:
                    # 리스트인 경우 첫 번째 값 사용
                    if isinstance(val, list) and len(val) > 0:
                        val = val[0]
                    # JSON 문자열인 경우 파싱 시도
                    elif isinstance(val, str) and val.startswith('{'):
                        try:
                            import json
                            json_data = json.loads(val)
                            if "statements" in json_data and isinstance(json_data["statements"], list):
                                # verdict 값들의 평균 계산
                                verdicts = []
                                for stmt in json_data["statements"]:
                                    if isinstance(stmt, dict) and "verdict" in stmt:
                                        verdicts.append(float(stmt["verdict"]))
                                if verdicts:
                                    val = sum(verdicts) / len(verdicts)
                                    print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ JSON verdict 평균)")
                                    return val
                        except:
                            pass
                    val = float(val)
                    if val == val:  # NaN 체크
                        print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ _scores_dict)")
                        return val
            
            # scores 딕셔너리 내부 확인 (RAGAS 0.3.x)
            if "scores" in d and isinstance(d["scores"], dict):
                val = d["scores"].get(metric_name)
                if val is not None:
                    val = float(val)
                    if val == val:  # NaN 체크
                        print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ scores)")
                        return val
            
            if "overall" in d and isinstance(d["overall"], dict):
                val = d["overall"].get(metric_name)
                if val is not None:
                    val = float(val)
                    if val == val:  # NaN 체크
                        print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ overall)")
                        return val
            
            if metric_name in d:
                val = d[metric_name]
                if val is not None:
                    val = float(val)
                    if val == val:  # NaN 체크
                        print(f"   - ✅ {metric_name}: {val:.4f} (__dict__ 직접)")
                        return val
        
        # 5. 직접 속성 접근
        if hasattr(result_obj, metric_name):
            val = getattr(result_obj, metric_name)
            if val is not None:
                val = float(val)
                if val == val:  # NaN 체크
                    print(f"   - ✅ {metric_name}: {val:.4f} (직접 속성)")
                    return val
        
        print(f"   - ❌ {metric_name} 값을 찾을 수 없음")
        return None
        
    except Exception as e:
        print(f"   - ⚠️ RAGAS 결과 파싱 실패 ({metric_name}): {e}")
        return None

def load_milvus_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- 노드: Milvus 로드 ---")
    global _vectorstore
    
    if "default" not in connections.list_connections() or not connections.has_connection("default"):
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)
    try:
        # Milvus 객체를 전역 변수에 저장 (상태에 저장하지 않음)
        _vectorstore = MilvusVectorStore(
            embedding_function=embedding_model,
            collection_name=MILVUS_COLLECTION,
            connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN},
        )
        print(f"Milvus 로드 완료: {MILVUS_COLLECTION}")
        return {**state, "retry_count": 0, "ragas_score": None, "ragas_details": {}}  # vectorstore 제거
    except Exception as e:
        print(f"Milvus 로드 실패: {e}")
        raise ConnectionError("Milvus 벡터스토어 로드 실패")

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("--- 노드: 문서 검색 ---")
    question = state.get("question")
    global _vectorstore
    vectorstore = _vectorstore  # 전역 변수에서 가져오기
    if not question or not vectorstore: raise ValueError("질문 또는 벡터스토어가 누락되었습니다.")
    
    # 검색 쿼리를 더 구체적으로 만들어서 관련성 높은 문서 검색
    enhanced_query = f"{question} 재배 방법 키우기 팁"
    
    # 더 많은 문서를 검색하고 품질이 좋은 것만 선택
    docs_with_scores = vectorstore.similarity_search_with_score(enhanced_query, k=10)
    
    # 유사도 점수가 높고 내용이 충분한 문서만 필터링
    filtered_docs = []
    for doc, score in docs_with_scores:
        content = doc.page_content or ""
        # 점수가 높고 내용이 충분한 문서만 선택
        if score > 0.5 and len(content.strip()) > 100:
            filtered_docs.append((doc, score))
    
    # 상위 5개만 사용
    docs_with_scores = filtered_docs[:5]

    context = ""
    print(f"{len(docs_with_scores)}개 문서 검색 완료.")
    for i, (doc, score) in enumerate(docs_with_scores):
        preview = (doc.page_content or "")[:100].replace("\n", " ")
        print(f"문서 {i+1} (점수: {score:.4f}): '{preview}...'")
        context += f"\n\n{doc.page_content}"
    print(f"컨텍스트 길이: {len(context)}자")
    
    original_docs = [doc.page_content for doc, score in docs_with_scores]
    return {**state, "db_context": context, "original_docs": original_docs}

# 1차 검증: 검색된 문서의 문맥 정밀도(Context Precision)를 평가하는 노드
def retrieval_validation_node(state: GraphState) -> Dict[str, Any]:
    print("--- 노드: 1차 검증 (검색 품질) ---")
    question = state.get("question")
    db_context = state.get("db_context", "")

    if not db_context or "관련 문서를 찾을 수 없습니다." in db_context:
        print("   - ❌ 검색된 문서가 없어 불충분으로 판단합니다.")
        return {**state, "is_retrieval_sufficient": False}

    # RAGAS 평가
    ragas_scores = {"context_precision": 0.0}
    try:
        print("   - 📊 RAGAS 검색 품질 평가 중...")

        # 컨텍스트 최적화
        max_context_length = 2500
        optimized_context = db_context[:max_context_length] if len(db_context) > max_context_length else db_context

        # 임시 답변 생성 (LLMContextPrecisionWithoutReference용)
        temp_answer = optimized_context[:1200] if len(optimized_context) > 0 else "정보 부족"

        print(f"   - 📝 SingleTurnSample 준비: 질문={len(question)}자, 컨텍스트={len(optimized_context)}자")

        # SalesRAGAS 방식: SingleTurnSample 사용
        llm_wrapper = LangchainLLMWrapper(llm)
        context_precision_scorer = LLMContextPrecisionWithoutReference(llm=llm_wrapper)
        context_precision_scorer.max_tokens = 16000
        
        # SingleTurnSample 생성
        context_sample = SingleTurnSample(
            user_input=question,
            response=temp_answer,
            retrieved_contexts=[optimized_context] if optimized_context else [""]
        )

        print("   - 🔄 RAGAS 평가 실행 중...")
        
        # 진짜 병렬 처리: Context Precision만 먼저 시작
        def run_context_precision_in_thread():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(
                        context_precision_scorer.single_turn_ascore(context_sample)
                    )
                    return float(result) if result is not None else 0.0
                finally:
                    new_loop.close()
                    asyncio.set_event_loop(None)
            except Exception as e:
                print(f"   - ⚠️ 스레드 내 Context Precision 평가 실패: {e}")
                return 0.0
        
        # 스레드 시작하고 바로 다음으로 넘어감 (병렬 처리)
        context_precision_container = [None]
        def context_precision_target():
            context_precision_container[0] = run_context_precision_in_thread()
        
        context_precision_thread = threading.Thread(target=context_precision_target)
        context_precision_thread.start()
        
        # Context Precision 결과 기다리기
        context_precision_thread.join()
        context_precision_score = context_precision_container[0]
        ragas_scores["context_precision"] = float(context_precision_score)

    except Exception as e:
        print(f"   - ⚠️ RAGAS 검색 평가 실패: {e}")

    # 개별 임계값 평가
    precision_sufficient = ragas_scores["context_precision"] >= 0.7
    is_sufficient = precision_sufficient
    
    print(f"   - 🎯 개별 평가 결과:")
    print(f"     • Context Precision: {ragas_scores['context_precision']:.3f} (임계값: 0.7) {'✅' if precision_sufficient else '❌'}")
    print(f"     • 최종 결과: {'✅ 충분' if is_sufficient else '⚠️ 불충분'}")
    
    return {**state, "is_retrieval_sufficient": is_sufficient}

def combine_context_node(state: GraphState) -> Dict[str, Any]:
    print("--- 노드: 컨텍스트 ---")
    db_context = state.get("db_context", "")
    web_context = state.get("web_context", "")
    final_context = db_context
    if web_context:
        print("   - DB와 웹 컨텍스트를 결합합니다.")
        final_context = f"[DB 검색 결과]\n{db_context}\n\n[웹 검색 결과]\n{web_context}"
    else:
        print("   - DB 컨텍스트만 사용합니다.")
    return {**state, "context": final_context}

# 초안 답변을 생성하는 노드 함수를 정의합니다.
def generate_draft_node(state: GraphState) -> Dict[str, Any]:
    print("--- 노드: 초안 생성 ---")
    if not state.get("context"):
        raise ValueError("context 누락")
    
    # 프롬프트와 컨텍스트, 질문을 조합하여 LLM에 전달하고 응답을 받습니다.
    response = llm.invoke(rag_prompt.format(context=state.get("context", ""), question=state.get("question", "")))
    # 응답 내용(content)을 가져옵니다.
    ans = response.content
    # 생성된 답변의 앞부분을 출력합니다.
    print("답변 생성 완료.")
    
    # 상태에 답변, 출처, 재시도 횟수를 추가하여 반환합니다.
    return {**state, "answer_draft": ans, "answer_source": "내부 DB"}

# 답변 개선 및 최종 생성 노드
def refine_answer_node(state: GraphState) -> Dict[str, Any]:
    print("--- 노드: 답변 개선 및 최종 생성 ---")
    if not state.get("answer_draft"):
        raise ValueError("answer_draft 누락")
    
    # 간단한 개선: 초안을 그대로 최종 답변으로 사용
    # 필요시 더 복잡한 개선 로직 추가 가능
    answer = state.get("answer_draft", "")
    
    print("답변 개선 완료.")
    return {**state, "answer": answer}

# 2차 검증: 생성된 답변의 충실도와 관련성을 평가하는 노드
def answer_validation_node(state: GraphState) -> Dict[str, Any]:
    print("--- 노드: 2차 검증 (답변 품질) ---")
    retry_count = state.get("retry_count", 0) + 1

    question = state.get("question")
    context = state.get("context", "")
    answer = state.get("answer", "")

    # 컨텍스트 및 답변 최적화
    max_context_length = 3000
    optimized_context = context[:max_context_length] if len(context) > max_context_length else context
    max_answer_length = 1200
    optimized_answer = answer[:max_answer_length] if len(answer) > max_answer_length else answer

    try:
        print("   - 📊 RAGAS 답변 품질 평가 중...")
        
        scores = {}
        
        # 🚀 WeatherAgent 방식: asyncio.gather() 사용
        async def evaluate_faithfulness():
            try:
                faithfulness_scorer = Faithfulness(llm=LangchainLLMWrapper(llm))
                faithfulness_sample = SingleTurnSample(
                    user_input=question,
                    response=optimized_answer,
                    retrieved_contexts=[optimized_context] if optimized_context else [""]
                )
                result = await faithfulness_scorer.single_turn_ascore(faithfulness_sample)
                return ("faithfulness", float(result) if result is not None else 0.0)
            except Exception as e:
                print(f"   - ⚠️ Faithfulness 평가 실패: {e}")
                return ("faithfulness", 0.0)
        
        async def evaluate_answer_relevancy():
            try:
                answer_relevancy_scorer = ResponseRelevancy(
                    llm=LangchainLLMWrapper(llm), 
                    embeddings=embedding_model
                )
                relevancy_sample = SingleTurnSample(
                    user_input=question,
                    response=optimized_answer,
                    retrieved_contexts=[optimized_context] if optimized_context else [""]
                )
                result = await answer_relevancy_scorer.single_turn_ascore(relevancy_sample)
                return ("answer_relevancy", float(result) if result is not None else 0.0)
            except Exception as e:
                print(f"   - ⚠️ Answer Relevancy 평가 실패: {e}")
                return ("answer_relevancy", 0.0)
        
        # 스레드 격리로 병렬 실행
        def run_parallel_in_thread():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    # 2개 평가를 동시에 실행 (진짜 병렬 처리)
                    results = new_loop.run_until_complete(
                        asyncio.gather(
                            evaluate_faithfulness(),
                            evaluate_answer_relevancy(),
                            return_exceptions=True
                        )
                    )
                    
                    # 결과 수집
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
                return {"faithfulness": 0.0, "answer_relevancy": 0.0}
        
        result_container = [None]
        def thread_target():
            result_container[0] = run_parallel_in_thread()
        
        print("   - 🔥 Faithfulness & Answer Relevancy 병렬 평가 시작!")
        thread = threading.Thread(target=thread_target)
        thread.start()
        thread.join()
        print("   - ✅ 병렬 평가 완료!")
        
        parallel_scores = result_container[0]
        scores.update(parallel_scores)
        
    except Exception as e:
        print(f"   - ❌ 병렬 RAGAS 평가 실패: {e}")
        scores['faithfulness'] = 0.0
        scores['answer_relevancy'] = 0.0

    f_val = scores.get('faithfulness', 0.0)
    r_val = scores.get('answer_relevancy', 0.0)

    # 개별 임계값 평가
    faithfulness_sufficient = f_val >= 0.4
    relevancy_sufficient = r_val >= 0.5
    is_sufficient = faithfulness_sufficient and relevancy_sufficient

    print(f"   - 📈 답변 품질 지표:")
    print(f"     • Faithfulness: {f_val:.3f} (임계값: 0.4) {'✅' if faithfulness_sufficient else '❌'}")
    print(f"     • Answer Relevancy: {r_val:.3f} (임계값: 0.5) {'✅' if relevancy_sufficient else '❌'}")
    print(f"     • 최종 결과: {'✅ 충분' if is_sufficient else '⚠️ 불충분'}")

    # 충분한 경우 최종 출력 처리
    if is_sufficient:
        answer = state.get("answer", "답변 생성 실패")
        source = state.get("answer_source", "알 수 없음")

        print("\n" + "="*20)
        print("최종 답변")
        print("="*20)
        print("\n" + answer)
        print("="*20)

    return {**state, "is_answer_sufficient": is_sufficient, "retry_count": retry_count}

# 웹 검색을 수행하는 노드 함수를 정의합니다.
def web_search_node(state: GraphState) -> Dict[str, Any]:
    print("--- 노드: 웹 검색 ---")
    question = state.get("question")
    if not question:
        raise ValueError("질문이 누락되었습니다.")
    if not TAVILY_API_KEY:
        print("TAVILY API 키 없음. 웹 검색 건너뜁니다.")
        return {**state, "web_search_results": "웹 검색 비활성화", "web_contexts": []}
    
    search_tool = TavilyClient(api_key=TAVILY_API_KEY)
    
    web_contexts: List[str] = []
    web_context_parts = []
    
    try:
        # TavilyClient 사용 - search 메서드 호출
        response = search_tool.search(query=question, max_results=3)
        
        # TavilyClient 응답 형식: {"results": [{"url": "", "content": "", "title": ""}]}
        if isinstance(response, dict) and "results" in response:
            results = response["results"]
            for r in results:
                if isinstance(r, dict):
                    title = (r.get("title") or "").strip()
                    content = (r.get("content") or r.get("snippet") or "").strip()
                    url = (r.get("url") or "").strip()
                    passage = f"{title}\n{content}\nURL: {url}".strip()
                    web_contexts.append(passage)
                    web_context_parts.append(f"- 출처: {url or 'N/A'}\n 내용: {content}")
        else:
            # 예상치 못한 응답 형식
            print(f"⚠️ 예상치 못한 Tavily 응답 형식: {type(response)}")
            web_context_parts.append(f"- 출처: N/A\n 내용: 웹 검색 응답 형식 오류")
            
    except Exception as e:
        print(f"⚠️ Tavily 검색 오류: {e}")
        web_context_parts.append(f"- 출처: N/A\n 내용: 웹 검색 실패 - {str(e)}")

    # 웹 검색 결과를 web_context에 저장
    web_context = "\n\n".join(web_context_parts)
    
    return {**state, "web_context": web_context}

def fallback_answer_node(state: GraphState) -> Dict[str, Any]:
    fallback_msg = "죄송하지만 이 질문에 대한 답변은 제공할 수 없습니다. 다른 질문을 해주세요."
    
    print("\n" + fallback_msg)
    

    
    return {**state, "answer": fallback_msg, "answer_source": "Fallback"}

def build_graph():

    g = StateGraph(GraphState)
    g.add_node("load_milvus", load_milvus_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("retrieval_validation", retrieval_validation_node)
    g.add_node("web_search", web_search_node)
    g.add_node("combine_context", combine_context_node)
    g.add_node("generate_draft", generate_draft_node)
    g.add_node("refine_answer", refine_answer_node)
    g.add_node("answer_validation", answer_validation_node)
    g.add_node("fallback_answer", fallback_answer_node)

    g.set_entry_point("load_milvus")
    g.add_edge("load_milvus", "retrieve")
    g.add_edge("retrieve", "retrieval_validation")

    # 1차 검증 결과에 따라 웹 검색 여부 결정
    g.add_conditional_edges(
        "retrieval_validation",
        lambda state: "sufficient" if state["is_retrieval_sufficient"] else "insufficient",
        {"sufficient": "combine_context", "insufficient": "web_search"}
    )
    g.add_edge("web_search", "combine_context")
    g.add_edge("combine_context", "generate_draft")
    g.add_edge("generate_draft", "refine_answer")
    g.add_edge("refine_answer", "answer_validation")

    # 2차 검증 결과에 따라 종료/재시도/대체 답변 결정
    def decide_after_answer_validation(state: GraphState) -> str:
        # 웹 검색이 이미 수행된 경우 (web_context가 있음)
        if state.get("web_context"):
            if state["is_answer_sufficient"]:
                return "end"
            else:
                return "fallback"
        # 일반적인 경우
        else:
            if state["is_answer_sufficient"]:
                return "end"
            elif state["retry_count"] >= 3:
                return "fallback"
            else:
                return "retry"

    g.add_conditional_edges(
        "answer_validation",
        decide_after_answer_validation,
        {"end": END, "fallback": "fallback_answer", "retry": "web_search"}
    )
    g.add_edge("fallback_answer", END)
    
    return g.compile()

async def run(state: dict) -> dict:
    """
    오케스트레이터에서 호출되는 메인 실행 함수 (비동기)
    
    Args:
        state: 오케스트레이터에서 전달받은 상태 딕셔너리
               - query: 사용자 질문 (필수)
    
    Returns:
        dict: 실행 결과
            - agent_answer: 최종 응답
    """
    try:
        # 질문 추출
        query = state.get("query", "")
        if not query:
            return {"agent_answer": "질문이 제공되지 않았습니다. 작물추천 관련 질문을 해주세요."}
        
        print(f"[작물추천_agent] 질문 처리 시작: {query}")
        
        # 그래프 빌드 및 실행
        app = build_graph()
        final_state = app.invoke({"question": query})
        
        # 결과 추출 - 안전한 타입 체크
        if isinstance(final_state, dict):
            answer = final_state.get("answer", "답변 생성에 실패했습니다.")
        elif isinstance(final_state, str):
            answer = final_state
        else:
            answer = "답변 형식이 올바르지 않습니다."
        
        print(f"[작물추천_agent] 답변 생성 완료: {len(answer)}자")
        
        return {"agent_answer": answer}
        
    except Exception as e:
        error_msg = f"작물추천 에이전트 실행 중 오류가 발생했습니다: {e}"
        print(f"[작물추천_agent] 오류: {e}")
        return {"agent_answer": error_msg}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 파이프라인 - 농업 작물 추천 시스템")
    parser.add_argument("-q", "--query", type=str, help="한 번만 실행할 질문 (예: -q '주말농장에 키울 작물 추천해줘')")
    args = parser.parse_args()
    
    app = build_graph()

    if args.query:
        # -q 모드: 한 번만 실행
        print(f"\n질문: '{args.query}'")
        print("-" * 20)
        final_state = app.invoke({"question": args.query})
        
    else:
        # 대화 모드
        print("(종료: exit/quit)")
        while True:
            q = input("\n질문> ").strip()
            if not q or q.lower() in ("exit", "quit"):
                break
            app.invoke({"question": q})
