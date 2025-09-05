# 주의 무시
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 표준 라이브러리
import os
import sys
import json
import time
import asyncio
from datetime import datetime

# 서드파티 라이브러리
import pandas as pd
from dotenv import load_dotenv

# LangChain 관련
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# RAGAS 관련
def evaluate_with_ragas(dataset, metrics):
    # 사용 직전에만 ragas import (지연 import)
    from ragas import evaluate
    return evaluate(dataset, metrics=metrics)

def get_ragas_metrics():
    # metrics도 내부에서 import
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall  # 필요한 것만
    return faithfulness, answer_relevancy, context_precision, context_recall

from ragas import evaluate, SingleTurnSample
from ragas.metrics import (
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    Faithfulness
)


# RAGAS 래퍼 & 데이터 스키마
def get_ragas_wrappers():
    from ragas.llms import LangchainLLMWrapper as RagasLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper as RagasEmbWrapper
    return RagasLLMWrapper, RagasEmbWrapper


# HuggingFace 관련
from datasets import Dataset

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from SalesAgent import run as run_sales_agent

class SalesRAGASEvaluator:
    def __init__(self):
        """SalesAgent RAGAS 평가기 초기화"""
        # 공식 문서에 따라 모델 설정
        self._setup_models()
        
        self.test_questions = self._create_test_questions()
        self.evaluation_results = []
    
    def _setup_models(self):
        """공식 문서에 따른 모델 설정"""

        LangchainLLMWrapper, LangchainEmbeddingsWrapper = get_ragas_wrappers()

        try:
            # LLM 설정 (기본 설정으로 복원)
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini", 
                temperature=0.1,  # 매우 일관된 질문 생성
                api_key=OPENAI_API_KEY
            )
            self.evaluator_llm = LangchainLLMWrapper(self.llm)
            
            # 임베딩 모델 설정
            self.embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(
                    model_name="BAAI/bge-m3",
                    model_kwargs={'device': 'cpu'}  # GPU가 있으면 'cuda'로 변경
                )
            )
            
            # 모델 설정 완료
        except Exception as e:
            raise
    
    def _create_test_questions(self):
        """평가용 테스트 질문 생성"""
        return []
    
    def run_sales_agent_evaluation(self, question):
        """SalesAgent를 실행하여 답변과 컨텍스트 수집"""
        try:
            # SalesAgent 실행
            initial_state = {"query": question}
            result_state = run_sales_agent(initial_state)
            
            # 결과 추출
            answer = result_state.get('final_answer', '')
            context = result_state.get('context', {})
            classification = result_state.get('question_classification', '')
            used_web_search = result_state.get('used_web_search', False)
            
            # 컨텍스트를 문자열로 변환
            context_str = self._format_context(context)
            
            return {
                'answer': answer,
                'context': context_str,
                'classification': classification,
                'used_web_search': used_web_search,
                'success': True
            }
            
        except Exception as e:
            return {
                'answer': f"오류 발생: {str(e)}",
                'context': "",
                'classification': "",
                'used_web_search': False,
                'success': False
            }
    
    def _format_context(self, context):
        """컨텍스트를 문자열로 포맷팅"""
        if not context:
            return ""
        
        formatted_parts = []
        
        # 시세 정보를 더 상세하게 포맷팅
        if '실시간시세' in context:
            prices = context['실시간시세']
            if prices and len(prices) > 0:
                # 의미 있는 가격 정보만 필터링
                valid_prices = []
                for price in prices:
                    if isinstance(price, str) and price.strip():
                        # "정보가 없습니다" 같은 키워드가 포함되지 않은 경우만 추가
                        if not any(keyword in price for keyword in ['해당 작물에 대한 정보는 현재 없습니다.']):
                            valid_prices.append(price)
                
                # 유효한 가격 정보가 있는 경우만 추가
                if valid_prices:
                    formatted_parts.append(f"시세 정보: {' | '.join(valid_prices)}")
        
        # 판매처 정보를 더 상세하게 포맷팅
        if '판매처' in context:
            vendors = context['판매처']
            if vendors and len(vendors) > 0:
                # 의미 있는 판매처 정보만 필터링
                valid_vendors = []
                for vendor in vendors:
                    if isinstance(vendor, str) and vendor.strip():
                        # "정보가 없습니다" 같은 키워드가 포함되지 않은 경우만 추가
                        if not any(keyword in vendor for keyword in ['해당 지역에 위치한 판매점 정보가 없습니다.']):
                            valid_vendors.append(vendor)
                
                # 유효한 판매처 정보가 있는 경우만 추가
                if valid_vendors:
                    formatted_parts.append(f"판매처 정보: {' | '.join(valid_vendors)}")
        
        # 웹검색 결과를 더 상세하게 포맷팅
        if '웹검색' in context:
            web_results = context['웹검색']
            if web_results:
                formatted_parts.append(f"웹 검색 결과: {' | '.join(map(str, web_results))}")
        

        
        # 추가 정보가 없으면 빈 문자열 반환
        if not formatted_parts:
            return ""
        
        # 컨텍스트를 더 명확하게 구조화 (RAGAS 점수 향상)
        if formatted_parts:
            return "농작물 시세 및 판매처 정보:\n" + "\n".join(formatted_parts)
        else:
            return ""
    
    async def evaluate_single_question(self, test_case):
        """단일 질문에 대한 평가 실행 (개별 RAGAS 평가 포함) - 비동기"""
        
        # SalesAgent 실행
        agent_result = self.run_sales_agent_evaluation(test_case['question'])
        
        if not agent_result['success']:
            return None
        
        # 개별 RAGAS 평가 실행 (비동기)
        individual_ragas_score = await self._evaluate_single_ragas_simple(
            test_case['question'],
            agent_result['answer'],
            agent_result['context']
        )
        
        # 평가 결과 구성
        evaluation_result = {
            'question': test_case['question'],
            'answer': agent_result['answer'],
            'context': agent_result['context'],
            'classification': agent_result['classification'],
            'used_web_search': agent_result['used_web_search'],
            'individual_ragas_score': individual_ragas_score,
            'timestamp': datetime.now().isoformat()
        }
        
        return evaluation_result
    
    async def _evaluate_single_ragas_simple(self, question, answer, context):
        """개별 질문-답변에 대한 RAGAS 라이브러리 평가 (WeatherAgent 방식 병렬 처리)"""
        try:
            # 🚀 WeatherAgent 방식: 3개 RAGAS 평가 모두 병렬 처리
            async def evaluate_context_precision():
                try:
                    context_precision_scorer = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)
                    context_sample = SingleTurnSample(
                        user_input=question,
                        response=answer,
                        retrieved_contexts=[context] if context else [""]
                    )
                    score = await context_precision_scorer.single_turn_ascore(context_sample)
                    return ("context_precision", float(score) if score is not None else 0.0)
                except Exception as e:
                    print(f"   - ⚠️ Context Precision 평가 실패: {e}")
                    return ("context_precision", 0.0)
            
            async def evaluate_faithfulness():
                try:
                    faithfulness_scorer = Faithfulness(llm=self.evaluator_llm)
                    faithfulness_sample = SingleTurnSample(
                        user_input=question,
                        response=answer,
                        retrieved_contexts=[context] if context else [""]
                    )
                    score = await faithfulness_scorer.single_turn_ascore(faithfulness_sample)
                    return ("faithfulness", float(score) if score is not None else 0.0)
                except Exception as e:
                    print(f"   - ⚠️ Faithfulness 평가 실패: {e}")
                    return ("faithfulness", 0.0)
            
            async def evaluate_answer_relevancy():
                try:
                    response_relevancy_scorer = ResponseRelevancy(
                        llm=self.evaluator_llm, 
                        embeddings=self.embeddings
                    )
                    sample = SingleTurnSample(
                        user_input=question,
                        response=answer,
                        retrieved_contexts=[context] if context else [""]
                    )
                    score = await response_relevancy_scorer.single_turn_ascore(sample)
                    return ("answer_relevancy", float(score) if score is not None else 0.0)
                except Exception as e:
                    print(f"   - ⚠️ Answer Relevancy 평가 실패: {e}")
                    return ("answer_relevancy", 0.0)
            
            # 3개 평가를 동시에 실행 (진짜 병렬 처리)
            print("   - 🔥 Context Precision, Faithfulness & Answer Relevancy 3개 병렬 평가 시작!")
            results = await asyncio.gather(
                evaluate_context_precision(),
                evaluate_faithfulness(),
                evaluate_answer_relevancy(),
                return_exceptions=True
            )
            
            # 결과 수집
            scores = {}
            for result in results:
                if isinstance(result, Exception):
                    print(f"   - ⚠️ RAGAS 평가 중 예외 발생: {result}")
                    continue
                metric_name, score = result
                scores[metric_name] = score
            
            print("   - ✅ 3개 병렬 평가 완료!")
            return scores
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    async def run_full_evaluation(self):
        """전체 테스트 케이스에 대한 평가 실행 (개별 RAGAS 포함) - 비동기"""
        
        # 배치 크기 설정 (Rate Limit 방지)
        batch_size = 3
        
        for i, test_case in enumerate(self.test_questions, 1):
            result = await self.evaluate_single_question(test_case)
            if result:
                self.evaluation_results.append(result)
            
            # 배치 단위로 대기 시간 조정
            if i % batch_size == 0:
                time.sleep(2)
            else:
                time.sleep(3)  # 개별 질문 간 3초 대기
        
        return self.evaluation_results
    
    def create_ragas_dataset(self):
        """RAGAS 평가용 데이터셋 생성"""
        if not self.evaluation_results:
            return None
        
        # RAGAS 형식에 맞는 데이터 구성
        ragas_data = []
        
        for result in self.evaluation_results:
            ragas_data.append({
                'user_input': result['question'],
                'response': result['answer'],
                'retrieved_contexts': [result['context']] if result['context'] else [""],
            })
        
        return Dataset.from_list(ragas_data)
    
    def run_ragas_evaluation(self):
        """RAGAS 메트릭을 사용한 평가 실행"""
        
        # 데이터셋 생성
        dataset = self.create_ragas_dataset()
        if not dataset:
            return None
        
        # RAGAS 메트릭 실행
        try:
            results = evaluate(
                dataset,
                metrics=[
                    ResponseRelevancy(llm=self.evaluator_llm, embeddings=self.embeddings),
                    LLMContextPrecisionWithoutReference(llm=self.evaluator_llm),  # LLM 기반 Context Precision
                    Faithfulness(llm=self.evaluator_llm)  # SingleTurnSample 방식
                ],
                llm=self.evaluator_llm
            )
            
            return results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None