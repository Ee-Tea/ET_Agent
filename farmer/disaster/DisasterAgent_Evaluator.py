import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========[ RAGAS 평가 전용 파일 ]=========

# =========[ 표준/외부 라이브러리 ]=========
import os
import json
import asyncio
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
import sys

import numpy as np
from dotenv import load_dotenv

# =========[ LangChain / LLM ]=========
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# ===== RAGAS lazy import 유틸 =====
_HAS_RAGAS = False
_RAGAS_ERR = None

def _ragas_try_import():
    """필요 시점에만 ragas 로드. 성공 시 _HAS_RAGAS True로 세팅."""
    global _HAS_RAGAS, _RAGAS_ERR
    if _HAS_RAGAS:
        return True
    try:
        import ragas  # noqa: F401
        _HAS_RAGAS = True
        _RAGAS_ERR = None
        return True
    except Exception as e:
        _HAS_RAGAS = False
        _RAGAS_ERR = e
        return False

def _get_ragas_core():
    """evaluate, SingleTurnSample, Dataset 반환"""
    if not _ragas_try_import():
        return None, None, None
    from ragas import evaluate, SingleTurnSample  # type: ignore
    try:
        from datasets import Dataset  # type: ignore
    except Exception:
        Dataset = None
    return evaluate, SingleTurnSample, Dataset

def _get_ragas_metrics():
    """ResponseRelevancy, Faithfulness, LLMContextPrecisionWithoutReference 반환"""
    if not _ragas_try_import():
        return None, None, None
    from ragas.metrics import (  # type: ignore
        ResponseRelevancy, Faithfulness, LLMContextPrecisionWithoutReference
    )
    return ResponseRelevancy, Faithfulness, LLMContextPrecisionWithoutReference

def _get_ragas_wrappers():
    """LangchainLLMWrapper, LangchainEmbeddingsWrapper 반환"""
    if not _ragas_try_import():
        return None, None
    from ragas.llms import LangchainLLMWrapper  # type: ignore
    from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
    return LangchainLLMWrapper, LangchainEmbeddingsWrapper

# torch는 선택 사항
try:
    import torch
    print("   - 🚀 GPU 가속 활성화 (RAGAS)" if torch.cuda.is_available() else "   - 💻 CPU 모드 (RAGAS)")
except Exception:
    torch = None
    print("   - ℹ️ torch 미설치: CPU 모드 (RAGAS)")

load_dotenv()

# =========[ 환경설정 ]=========
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")

***REMOVED*** 설정
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# =========[ RAGAS 백엔드 설정 ]=========
RAGAS_BACKEND = os.getenv("RAGAS_BACKEND", "openai").lower()
RAGAS_OPENAI_LLM = os.getenv("RAGAS_OPENAI_LLM", "gpt-4o-mini")
RAGAS_OPENAI_EMBED = os.getenv("RAGAS_OPENAI_EMBED", "text-embedding-3-small")

_RAGAS_LLM = None
_RAGAS_EMB = None
_RAGAS_LLM_WRAPPER = None
_RAGAS_EMB_WRAPPER = None

def _init_ragas_backend():
    """RAGAS LLM/Embedding 백엔드 초기화. OpenAI LLM + HuggingFace Embeddings 사용."""
    global _RAGAS_LLM, _RAGAS_EMB, RAGAS_BACKEND, _RAGAS_LLM_WRAPPER, _RAGAS_EMB_WRAPPER
    if not _HAS_RAGAS:
        return

    try:
        if not OPENAI_API_KEY=REDACTED("   - ⚠️ OPENAI_API_KEY=REDACTED 비활성화")
            return
        
        # env 세팅
        os.environ["OPENAI_API_KEY=REDACTED
        if OPENAI_BASE_URL:
            os.environ["OPENAI_API_BASE"] = OPENAI_BASE_URL
        
        llm = ChatOpenAI(model=RAGAS_OPENAI_LLM, temperature=0)
        ***REMOVED*** 임베딩 대신 HuggingFace 임베딩 사용 (권한 문제 해결)
        emb = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True}
        )
        _RAGAS_LLM = llm
        _RAGAS_EMB = emb
        LangchainLLMWrapper, LangchainEmbeddingsWrapper = _get_ragas_wrappers()
        
        # RAGAS Wrapper 설정
        _RAGAS_LLM_WRAPPER = LangchainLLMWrapper(_RAGAS_LLM)
        _RAGAS_EMB_WRAPPER = LangchainEmbeddingsWrapper(_RAGAS_EMB)
        
        print(f"   - 🔑 RAGAS 백엔드=OpenAI LLM + HF Embeddings · LLM={RAGAS_OPENAI_LLM}, EMB={EMBED_MODEL_NAME}")
    except Exception as e:
        print(f"   - ⚠️ RAGAS 백엔드 초기화 실패: {e}")

_init_ragas_backend()

# =========[ RAGAS 결과 파싱 헬퍼 ]=========
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

# =========[ 평가 함수들 ]=========
class EvaluationResult:
    """평가 결과를 담는 클래스"""
    def __init__(self):
        self.context_precision: Optional[float] = None
        self.faithfulness: Optional[float] = None
        self.answer_relevancy: Optional[float] = None
        self.overall_score: Optional[float] = None
        self.evaluation_time: Optional[datetime] = None
        self.error_message: Optional[str] = None

def evaluate_retrieval_quality(question: str, context: str) -> EvaluationResult:
    """
    검색 품질 평가 (Context Precision)
    
    Args:
        question: 사용자 질문
        context: 검색된 컨텍스트
        
    Returns:
        EvaluationResult: 평가 결과
    """
    result = EvaluationResult()
    result.evaluation_time = datetime.now(ZoneInfo("Asia/Seoul"))
    
    if not _HAS_RAGAS or not _RAGAS_LLM_WRAPPER:
        result.error_message = "RAGAS 백엔드가 준비되지 않음"
        return result
    
    if not context or "관련 문서를 찾을 수 없습니다." in context:
        result.error_message = "검색된 문서가 없음"
        return result
    
    try:
        print("   - 📊 RAGAS 검색 품질 평가 중...")
        LLMContextPrecisionWithoutReference = _get_ragas_metrics().LLMContextPrecisionWithoutReference
        SingleTurnSample = _get_ragas_core().SingleTurnSample

        # 컨텍스트 최적화
        max_context_length = 2500
        optimized_context = context[:max_context_length] if len(context) > max_context_length else context

        # 임시 답변 생성 (LLMContextPrecisionWithoutReference용)
        temp_answer = optimized_context[:1200] if len(optimized_context) > 0 else "정보 부족"

        print(f"   - 📝 SingleTurnSample 준비: 질문={len(question)}자, 컨텍스트={len(optimized_context)}자")

        # RAGAS 평가 설정
        context_precision_scorer = LLMContextPrecisionWithoutReference(llm=_RAGAS_LLM_WRAPPER)
        
        # SingleTurnSample 생성
        context_sample = SingleTurnSample(
            user_input=question,
            response=temp_answer,
            retrieved_contexts=[optimized_context] if optimized_context else [""]
        )

        print("   - 🔄 RAGAS 평가 실행 중...")
        
        # 비동기 평가 실행
        async def evaluate_context_precision():
            try:
                score = await context_precision_scorer.single_turn_ascore(context_sample)
                return float(score) if score is not None else 0.0
            except Exception as e:
                print(f"   - ⚠️ Context Precision 평가 실패: {e}")
                return 0.0
        
        # 스레드 격리로 비동기 실행
        def run_in_thread():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(evaluate_context_precision())
                finally:
                    new_loop.close()
                    asyncio.set_event_loop(None)
            except Exception as e:
                print(f"   - ⚠️ 스레드 내 Context Precision 평가 실패: {e}")
                return 0.0
        
        result_container = [None]
        def thread_target():
            result_container[0] = run_in_thread()
        
        thread = threading.Thread(target=thread_target)
        thread.start()
        thread.join()
        
        result.context_precision = result_container[0]
        
        print(f"   - 📈 검색 품질 지표:")
        print(f"     • Context Precision (LLM-based): {result.context_precision:.3f}")

    except Exception as e:
        result.error_message = f"RAGAS 검색 평가 실패: {e}"
        print(f"   - ⚠️ {result.error_message}")
    
    return result

def evaluate_answer_quality(question: str, context: str, answer: str) -> EvaluationResult:
    """
    답변 품질 평가 (Faithfulness + Answer Relevancy)
    
    Args:
        question: 사용자 질문
        context: 검색된 컨텍스트
        answer: 생성된 답변
        
    Returns:
        EvaluationResult: 평가 결과
    """
    result = EvaluationResult()
    result.evaluation_time = datetime.now(ZoneInfo("Asia/Seoul"))
    
    if not _HAS_RAGAS or not (_RAGAS_LLM_WRAPPER and _RAGAS_EMB_WRAPPER):
        result.error_message = "RAGAS 백엔드가 준비되지 않음"
        return result
    
    if not all([question, context, answer]):
        result.error_message = "평가 정보가 부족함"
        return result
    
    # 컨텍스트 및 답변 최적화
    max_context_length = 3000
    optimized_context = context[:max_context_length] if len(context) > max_context_length else context
    max_answer_length = 1200
    optimized_answer = answer[:max_answer_length] if len(answer) > max_answer_length else answer

    if len(optimized_context.strip()) < 50 or len(optimized_answer.strip()) < 20:
        result.error_message = "컨텍스트/답변이 너무 짧음"
        return result

    print(f"   - 📝 답변 품질 평가 준비: 질문={len(question)}자, 컨텍스트={len(optimized_context)}자, 답변={len(optimized_answer)}자")

    try:
        print("   - 📊 RAGAS 답변 품질 평가 중...")
        Faithfulness = _get_ragas_metrics().Faithfulness
        ResponseRelevancy = _get_ragas_metrics().ResponseRelevancy
        SingleTurnSample = _get_ragas_core().SingleTurnSample
        
        scores = {}
        
        # Faithfulness와 Answer Relevancy 병렬 처리
        try:
            # Faithfulness 설정
            faithfulness_scorer = Faithfulness(llm=_RAGAS_LLM_WRAPPER)
            faithfulness_sample = SingleTurnSample(
                user_input=question,
                response=optimized_answer,
                retrieved_contexts=[optimized_context] if optimized_context else [""]
            )
            
            # Answer Relevancy 설정
            answer_relevancy_scorer = ResponseRelevancy(
                llm=_RAGAS_LLM_WRAPPER, 
                embeddings=_RAGAS_EMB_WRAPPER
            )
            relevancy_sample = SingleTurnSample(
                user_input=question,
                response=optimized_answer,
                retrieved_contexts=[optimized_context] if optimized_context else [""]
            )
            
            # 비동기 평가 함수들
            async def evaluate_faithfulness():
                try:
                    score = await faithfulness_scorer.single_turn_ascore(faithfulness_sample)
                    return ("faithfulness", float(score) if score is not None else 0.0)
                except Exception as e:
                    print(f"   - ⚠️ Faithfulness 평가 실패: {e}")
                    return ("faithfulness", 0.0)
            
            async def evaluate_answer_relevancy():
                try:
                    score = await answer_relevancy_scorer.single_turn_ascore(relevancy_sample)
                    return ("answer_relevancy", float(score) if score is not None else 0.0)
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

        result.faithfulness = scores.get('faithfulness', 0.0)
        result.answer_relevancy = scores.get('answer_relevancy', 0.0)
        
        # 전체 점수 계산 (가중 평균)
        if result.faithfulness is not None and result.answer_relevancy is not None:
            result.overall_score = (result.faithfulness * 0.4 + result.answer_relevancy * 0.6)

        print(f"   - 📈 답변 품질 지표:")
        print(f"     • Faithfulness: {result.faithfulness:.3f}")
        print(f"     • Answer Relevancy: {result.answer_relevancy:.3f}")
        print(f"     • Overall Score: {result.overall_score:.3f}")

    except Exception as e:
        result.error_message = f"RAGAS 답변 평가 실패: {e}"
        print(f"   - ❌ {result.error_message}")
    
    return result

def evaluate_full_pipeline(question: str, context: str, answer: str) -> EvaluationResult:
    """
    전체 파이프라인 평가 (검색 + 답변 품질)
    
    Args:
        question: 사용자 질문
        context: 검색된 컨텍스트
        answer: 생성된 답변
        
    Returns:
        EvaluationResult: 평가 결과
    """
    print("🔍 전체 파이프라인 평가 시작...")
    
    # 검색 품질 평가
    retrieval_result = evaluate_retrieval_quality(question, context)
    
    # 답변 품질 평가
    answer_result = evaluate_answer_quality(question, context, answer)
    
    # 결과 통합
    final_result = EvaluationResult()
    final_result.evaluation_time = datetime.now(ZoneInfo("Asia/Seoul"))
    
    if retrieval_result.error_message:
        final_result.error_message = f"검색 평가 오류: {retrieval_result.error_message}"
    elif answer_result.error_message:
        final_result.error_message = f"답변 평가 오류: {answer_result.error_message}"
    else:
        final_result.context_precision = retrieval_result.context_precision
        final_result.faithfulness = answer_result.faithfulness
        final_result.answer_relevancy = answer_result.answer_relevancy
        
        # 전체 점수 계산
        if all([final_result.context_precision, final_result.faithfulness, final_result.answer_relevancy]):
            final_result.overall_score = (
                final_result.context_precision * 0.3 + 
                final_result.faithfulness * 0.3 + 
                final_result.answer_relevancy * 0.4
            )
    
    print("✅ 전체 파이프라인 평가 완료!")
    return final_result

# =========[ 사용 예시 ]=========
if __name__ == "__main__":
    # 사용 예시
    sample_question = "2023년 태풍으로 인한 농작물 피해는 어땠나요?"
    sample_context = """
    [유사도:0.8234][text][농업재해보고서_2023.pdf p.15]
    2023년 태풍 카눈으로 인해 전국적으로 큰 피해가 발생했습니다.
    특히 경상남도와 전라남도에서 벼농사 피해가 심각했습니다.
    """
    sample_answer = """
    2023년 태풍 카눈으로 인해 전국적으로 농작물 피해가 발생했습니다.
    특히 경상남도와 전라남도에서 벼농사 피해가 심각했으며,
    총 피해면적은 1,234ha에 달했습니다.
    """
    
    print("🧪 RAGAS 평가 테스트 시작...")
    
    # 전체 평가 실행
    result = evaluate_full_pipeline(sample_question, sample_context, sample_answer)
    
    print("\n=== 평가 결과 ===")
    if result.error_message:
        print(f"❌ 오류: {result.error_message}")
    else:
        print(f"📊 Context Precision: {result.context_precision:.3f}")
        print(f"📊 Faithfulness: {result.faithfulness:.3f}")
        print(f"📊 Answer Relevancy: {result.answer_relevancy:.3f}")
        print(f"📊 Overall Score: {result.overall_score:.3f}")
        print(f"⏰ 평가 시간: {result.evaluation_time}")

