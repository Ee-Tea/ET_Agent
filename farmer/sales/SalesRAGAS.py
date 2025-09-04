# 주의 무시
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 표준 라이브러리
import os
import sys
import json
import time
from datetime import datetime

# 서드파티 라이브러리
import pandas as pd
from dotenv import load_dotenv

# LangChain 관련
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# RAGAS 관련
from ragas import evaluate, SingleTurnSample
from ragas.metrics import (
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    Faithfulness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# HuggingFace 관련
from datasets import Dataset

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED(os.path.dirname(os.path.abspath(__file__)))
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
        try:
            # LLM 설정 (기본 설정으로 복원)
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini", 
                temperature=0.1,  # 매우 일관된 질문 생성
                api_key=OPENAI_API_KEY=REDACTED = LangchainLLMWrapper(self.llm)
            
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
        """개별 질문-답변에 대한 RAGAS 라이브러리 평가 (비동기)"""
        try:
            # RAGAS 메트릭 실행
            scores = {}
            
            try:
                # Response Relevancy (기본 설정으로 복원)
                response_relevancy_scorer = ResponseRelevancy(
                    llm=self.evaluator_llm, 
                    embeddings=self.embeddings
                )
                
                # SingleTurnSample 생성
                sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=[context] if context else [""]
                )
                
                # await 키워드 추가
                answer_relevancy_score = await response_relevancy_scorer.single_turn_ascore(sample)
                scores['answer_relevancy'] = float(answer_relevancy_score)
                
            except Exception as e:
                scores['answer_relevancy'] = 0.0
            
            try:
                # Context Precision without reference (LLM 기반 방식)
                context_precision_scorer = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)
                
                # SingleTurnSample 생성 (Context Precision용)
                context_sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=[context] if context else [""]
                )
                
                # await 키워드 추가
                context_precision_score = await context_precision_scorer.single_turn_ascore(context_sample)
                scores['context_precision'] = float(context_precision_score)
                
            except Exception as e:
                scores['context_precision'] = 0.0
            
            try:
                # Faithfulness (SingleTurnSample 방식)
                faithfulness_scorer = Faithfulness(llm=self.evaluator_llm)
                
                # SingleTurnSample 생성 (Faithfulness용)
                faithfulness_sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=[context] if context else [""]
                )
                
                # await 키워드 추가
                faithfulness_score = await faithfulness_scorer.single_turn_ascore(faithfulness_sample)
                scores['faithfulness'] = float(faithfulness_score)
                
            except Exception as e:
                scores['faithfulness'] = 0.0
            
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