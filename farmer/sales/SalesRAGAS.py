# 주의 무시
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 표준 라이브러리
import os
import sys
import json
from datetime import datetime

# 서드파티 라이브러리
import pandas as pd
from dotenv import load_dotenv

# LangChain 관련
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# RAGAS 관련 모듈들을 지연 로딩으로 import
def import_ragas_modules():
    """RAGAS 관련 모듈들을 지연 로딩으로 import"""
    try:
        from ragas import evaluate, SingleTurnSample
        from ragas.metrics import (
            ResponseRelevancy,
            LLMContextPrecisionWithReference,
            Faithfulness,
            LLMContextRecall
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        
        return {
            'evaluate': evaluate,
            'SingleTurnSample': SingleTurnSample,
            'ResponseRelevancy': ResponseRelevancy,
            'LLMContextPrecisionWithReference': LLMContextPrecisionWithReference,
            'Faithfulness': Faithfulness,
            'LLMContextRecall': LLMContextRecall,
            'LangchainLLMWrapper': LangchainLLMWrapper,
            'LangchainEmbeddingsWrapper': LangchainEmbeddingsWrapper
        }
    except ImportError as e:
        print(f"❌ RAGAS 모듈 import 실패: {e}")
        raise

# RAGAS 래퍼 & 데이터 스키마 (기존 함수 유지)
def get_ragas_wrappers():
    from ragas.llms import LangchainLLMWrapper as RagasLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper as RagasEmbWrapper
    return RagasLLMWrapper, RagasEmbWrapper

# HuggingFace 관련
from datasets import Dataset

# 환경 변수 로드 - 프로젝트 루트의 .env 파일 로드
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from SalesAgent import run
from common.milvus_manager import MilvusDBManager

def __init__(self, csv_path="./farmer/sales/data/price_golden_dataset_20250912_160841.csv"):
    """SalesAgent RAGAS 평가기 초기화"""
    print(f"🔧 SalesRAGAS 평가기 초기화 시작...")
    print(f"📁 CSV 파일 경로: {csv_path}")
    
    try:
        # MilvusDB 관리자 초기화
        print("🔗 MilvusDB 관리자 초기화 중...")
        self.milvus_manager = MilvusDBManager()
        
        # MilvusDB 연결
        try:
            if not self.milvus_manager.connect():
                print("⚠️ MilvusDB 연결 실패 - 일부 기능이 제한될 수 있습니다.")
                self.milvus_manager = None  # 연결 실패 시 None으로 설정
            else:
                print("✅ MilvusDB 연결 성공")
        except Exception as e:
            print(f"❌ MilvusDB 연결 중 오류 발생: {e}")
            self.milvus_manager = None  # 오류 발생 시 None으로 설정
        
        # 공식 문서에 따라 모델 설정
        print("🤖 모델 설정 중...")
        self._setup_models()
        print("✅ 모델 설정 완료")
        
        self.csv_path = csv_path
        print("📊 테스트 질문 생성 중...")
        self.test_questions = self._create_test_questions()
        print(f"✅ 테스트 질문 생성 완료: {len(self.test_questions)}개")
        
        self.evaluation_results = []
        print("🎯 평가기 초기화 완료!")
        print(f"🔗 MilvusDB 연결 상태: {'✅ 연결됨' if self.milvus_manager and self.milvus_manager.is_connected else '❌ 연결 안됨'}")
        
    except Exception as e:
        print(f"❌ 평가기 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        raise

def _setup_models(self):
    """공식 문서에 따른 모델 설정"""
    print("📦 RAGAS 모듈들 로딩 중...")
    try:
        self.ragas_modules = import_ragas_modules()
        print("✅ RAGAS 모듈들 로딩 완료")
    except Exception as e:
        print(f"❌ RAGAS 모듈 로딩 실패: {e}")
        raise

    try:
        # LLM 설정 (기본 설정으로 복원)
        print("🤖 OpenAI LLM 설정 중...")
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0.1,
            api_key=OPENAI_API_KEY
        )
        print("✅ OpenAI LLM 설정 완료")
        
        print("🔧 LLM 래퍼 설정 중...")
        self.evaluator_llm = self.ragas_modules['LangchainLLMWrapper'](self.llm)
        print("✅ LLM 래퍼 설정 완료")
        
        # 임베딩 모델 설정
        print("🔤 임베딩 모델 설정 중...")
        self.embeddings = self.ragas_modules['LangchainEmbeddingsWrapper'](
            HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'}  # GPU가 있으면 'cuda'로 변경
            )
        )
        print("✅ 임베딩 모델 설정 완료")
        
        # 모델 설정 완료
        print("🎉 모든 모델 설정 완료!")
    except Exception as e:
        print(f"❌ 모델 설정 실패: {e}")
        import traceback
        traceback.print_exc()
        raise

def _create_test_questions(self):
    """평가용 테스트 질문 생성"""
    print(f"📊 테스트 질문 생성 중...")
    print(f"📁 CSV 파일 경로 확인: {self.csv_path}")
    
    if self.csv_path and os.path.exists(self.csv_path):
        print("✅ CSV 파일 존재 확인됨")
        return self._load_csv_data()
    else:
        print(f"⚠️ CSV 파일이 존재하지 않음: {self.csv_path}")
        return []

def _load_csv_data(self):
    """CSV 파일에서 user_input과 reference 컬럼을 읽어옵니다."""
    print(f"📖 CSV 파일 읽기 시작: {self.csv_path}")
    
    try:
        # CSV 파일 읽기
        print("📄 pandas로 CSV 파일 읽는 중...")
        df = pd.read_csv(self.csv_path)
        print(f"✅ CSV 파일 읽기 완료: {len(df)}행")
        
        # 컬럼 확인
        print(f"📋 CSV 컬럼들: {list(df.columns)}")
        
        # 필요한 컬럼 확인
        required_columns = ['user_input', 'reference']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 필수 컬럼이 없습니다: {missing_columns}")
            print(f"📋 사용 가능한 컬럼: {list(df.columns)}")
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
        
        print("✅ 필수 컬럼 확인 완료")
        
        # 데이터 추출
        print("🔄 데이터 추출 중...")
        test_cases = []
        for i, (_, row) in enumerate(df.iterrows()):
            test_cases.append({
                'question': str(row['user_input']).strip(),
                'reference': str(row['reference']).strip()
            })
            if i < 3:  # 처음 3개만 출력
                print(f"  📝 질문 {i+1}: {str(row['user_input']).strip()[:50]}...")
        
        print(f"✅ CSV 데이터 로드 완료: {len(test_cases)}개 질문")
        return test_cases
        
    except Exception as e:
        print(f"❌ CSV 파일 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_sales_agent_evaluation(self, question):
    """SalesAgent를 실행하여 답변과 컨텍스트 수집"""
    print(f"🤖 SalesAgent 실행 시작: {question[:50]}...")
    
    try:
        # MilvusDB 연결 정보 준비
        milvus_data = {}
        if self._is_milvus_connected():
            milvus_data = {
                "connection_status": True,
                "host": self.milvus_manager.host,
                "port": self.milvus_manager.port,
                "embedding_model_name": self.milvus_manager.embedding_model_name
            }
            print("🔗 MilvusDB 연결 정보 주입됨")
        else:
            milvus_data = {
                "connection_status": False,
                "error": "MilvusDB 연결되지 않음"
            }
            print("⚠️ MilvusDB 연결 정보 없음")
        
        # SalesAgent 실행 (동기 함수 호출)
        print("🔄 SalesAgent 실행 중...")
        initial_state = {
            "query": question,
            "milvus_data": milvus_data  # MilvusDB 연결 정보 주입
        }
        result_state = run(initial_state)
        print("✅ SalesAgent 실행 완료")
        
        # 결과 추출
        print("📊 결과 추출 중...")
        answer = result_state.get('agent_answer', '')
        context = result_state.get('context', {})
        
        print(f"📝 답변 길이: {len(answer)}자")
        print(f"💬 SalesAgent 응답: {answer}")
        
        # 컨텍스트를 문자열로 변환
        print("🔧 컨텍스트 포맷팅 중...")
        context_str = self._format_context(context)
        print(f"📄 컨텍스트 길이: {len(context_str)}자")
        if context_str:
            print(f"📋 컨텍스트 내용: {context_str[:200]}...")
        else:
            print("⚠️ 컨텍스트가 비어있음")
        
        print("✅ SalesAgent 평가 완료")
        return {
            'answer': answer,
            'context': context_str,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ SalesAgent 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return {
            'answer': f"오류 발생: {str(e)}",
            'context': "",
            'success': False
        }

def _is_milvus_connected(self):
    """MilvusDB 연결 상태 확인"""
    return self.milvus_manager is not None and self.milvus_manager.is_connected

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
                formatted_parts.append(' | '.join(valid_prices))
    
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
                formatted_parts.append(' | '.join(valid_vendors))
    
    # 웹검색 결과를 더 상세하게 포맷팅
    if '웹검색' in context:
        web_results = context['웹검색']
        if web_results:
            formatted_parts.append(' | '.join(map(str, web_results)))
    
    # 추가 정보가 없으면 빈 문자열 반환
    if not formatted_parts:
        return ""
    
    # 컨텍스트를 더 명확하게 구조화 (RAGAS 점수 향상)
    if formatted_parts:
        return "\n".join(formatted_parts)
    else:
        return ""

def evaluate_single_question(self, test_case):
    """단일 질문에 대한 평가 실행 (개별 RAGAS 평가 포함) - 동기"""
    
    question = test_case['question']
    reference = test_case.get('reference', '')
    
    print(f"\n📝 질문 평가 시작: {question[:50]}...")
    
    # SalesAgent 실행
    agent_result = self.run_sales_agent_evaluation(question)
    
    if not agent_result['success']:
        print(f"❌ SalesAgent 실행 실패")
        return None
    
    # 개별 RAGAS 평가 실행 (동기)
    individual_ragas_score = self._evaluate_single_ragas_simple(
        question,
        agent_result['answer'],
        agent_result['context'],
        reference
    )
    
    # 평가 결과 구성
    evaluation_result = {
        'question': question,
        'reference': reference,
        'answer': agent_result['answer'],
        'context': agent_result['context'],
        'individual_ragas_score': individual_ragas_score,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"✅ 질문 평가 완료")
    return evaluation_result

def _evaluate_single_ragas_simple(self, question, answer, context, reference=""):
    """개별 질문-답변에 대한 RAGAS 라이브러리 평가 (동기 처리)"""
    print(f"📊 RAGAS 평가 시작...")
    print(f"📝 질문: {question[:50]}...")
    print(f"📄 답변 길이: {len(answer)}자")
    print(f"📄 컨텍스트 길이: {len(context)}자")
    print(f"📄 참조 길이: {len(reference)}자")
    
    try:
        # 🚀 WeatherAgent 방식: 4개 RAGAS 평가 모두 병렬 처리
        def evaluate_context_precision():
            try:
                print("🔍 Context Precision 평가 중...")
                # Context Precision은 retrieved_contexts가 필요하므로 컨텍스트가 있을 때만 평가
                if not context or not context.strip():
                    print("   - ⚠️ Context Precision 건너뜀: 컨텍스트 없음")
                    return ("context_precision", 0.0)
                
                context_precision_scorer = self.ragas_modules['LLMContextPrecisionWithReference'](llm=self.evaluator_llm)
                
                # 컨텍스트가 비어있거나 None인 경우 처리
                contexts = [context] if context and context.strip() else [""]
                
                context_sample = self.ragas_modules['SingleTurnSample'](
                    user_input=question,
                    response=answer,
                    reference=reference,
                    retrieved_contexts=contexts
                )
                score = context_precision_scorer.single_turn_score(context_sample)
                print(f"✅ Context Precision: {float(score) if score is not None else 0.0:.3f}")
                return ("context_precision", float(score) if score is not None else 0.0)
            except Exception as e:
                print(f"   - ⚠️ Context Precision 평가 실패: {e}")
                return ("context_precision", 0.0)
        
        def evaluate_faithfulness():
            try:
                print("🔍 Faithfulness 평가 중...")
                faithfulness_scorer = self.ragas_modules['Faithfulness'](llm=self.evaluator_llm)
                
                # 컨텍스트가 비어있거나 None인 경우 처리
                contexts = [context] if context and context.strip() else [""]
                
                faithfulness_sample = self.ragas_modules['SingleTurnSample'](
                    user_input=question,
                    response=answer,
                    reference=reference,
                    retrieved_contexts=contexts
                )
                score = faithfulness_scorer.single_turn_score(faithfulness_sample)
                print(f"✅ Faithfulness: {float(score) if score is not None else 0.0:.3f}")
                return ("faithfulness", float(score) if score is not None else 0.0)
            except Exception as e:
                print(f"   - ⚠️ Faithfulness 평가 실패: {e}")
                import traceback
                traceback.print_exc()
                return ("faithfulness", 0.0)
        
        def evaluate_answer_relevancy():
            try:
                print("🔍 Answer Relevancy 평가 중...")
                response_relevancy_scorer = self.ragas_modules['ResponseRelevancy'](
                    llm=self.evaluator_llm, 
                    embeddings=self.embeddings
                )
                
                # ResponseRelevancy
                contexts = [context] if context and context.strip() else [""]
                sample = self.ragas_modules['SingleTurnSample'](
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts
                )
                score = response_relevancy_scorer.single_turn_score(sample)
                print(f"✅ Answer Relevancy: {float(score) if score is not None else 0.0:.3f}")
                return ("answer_relevancy", float(score) if score is not None else 0.0)
            except Exception as e:
                print(f"   - ⚠️ Answer Relevancy 평가 실패: {e}")
                return ("answer_relevancy", 0.0)
        
        def evaluate_context_recall():
            try:
                print("🔍 Context Recall 평가 중...")
                # Context Recall은 retrieved_contexts가 필요하므로 컨텍스트가 있을 때만 평가
                if not context or not context.strip():
                    print("   - ⚠️ Context Recall 건너뜀: 컨텍스트 없음")
                    return ("context_recall", 0.0)
                
                context_recall_scorer = self.ragas_modules['LLMContextRecall'](llm=self.evaluator_llm)
                
                # 컨텍스트가 비어있거나 None인 경우 처리
                contexts = [context] if context and context.strip() else [""]
                
                recall_sample = self.ragas_modules['SingleTurnSample'](
                    user_input=question,
                    response=answer,
                    reference=reference,
                    retrieved_contexts=contexts
                )
                score = context_recall_scorer.single_turn_score(recall_sample)
                print(f"✅ Context Recall: {float(score) if score is not None else 0.0:.3f}")
                return ("context_recall", float(score) if score is not None else 0.0)
            except Exception as e:
                print(f"   - ⚠️ Context Recall 평가 실패: {e}")
                return ("context_recall", 0.0)
        
        # 4개 평가를 순차적으로 실행 (동기 처리)
        print("   - 🔥 Context Precision, Faithfulness, Answer Relevancy & Context Recall 4개 순차 평가 시작!")
        results = [
            evaluate_context_precision(),
            evaluate_faithfulness(),
            evaluate_answer_relevancy(),
            evaluate_context_recall()
        ]
        
        # 결과 수집
        scores = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"   - ⚠️ RAGAS 평가 중 예외 발생: {result}")
                continue
            metric_name, score = result
            scores[metric_name] = score
        
        print("   - ✅ 4개 병렬 평가 완료!")
        return scores
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def run_full_evaluation(self, batch_size=3):
    """전체 테스트 케이스에 대한 평가 실행 (개별 RAGAS 포함) - 동기"""
    
    print(f"\n🚀 전체 평가 시작!")
    print(f"📊 평가할 질문 수: {len(self.test_questions)}")
    print(f"⚙️ 배치 크기: {batch_size}")
    
    if not self.test_questions:
        print("❌ 평가할 질문이 없습니다.")
        return []
    
    # 각 질문에 대해 평가 실행
    for i, test_case in enumerate(self.test_questions, 1):
        print(f"\n{'='*60}")
        print(f"📝 질문 {i}/{len(self.test_questions)} 평가 시작")
        print(f"{'='*60}")
        
        try:
            result = self.evaluate_single_question(test_case)
            if result:
                self.evaluation_results.append(result)
                print(f"✅ 질문 {i} 평가 완료")
            else:
                print(f"❌ 질문 {i} 평가 실패")
        except Exception as e:
            print(f"❌ 질문 {i} 평가 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        # 대기 시간 제거 (동기 처리로 변경했으므로 불필요)
    
    print(f"\n🎉 전체 평가 완료!")
    print(f"📊 성공한 평가: {len(self.evaluation_results)}개")
    print(f"📊 실패한 평가: {len(self.test_questions) - len(self.evaluation_results)}개")
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
            'reference': result.get('reference', ''),
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
        results = self.ragas_modules['evaluate'](
            dataset,
            metrics=[
                self.ragas_modules['ResponseRelevancy'](llm=self.evaluator_llm, embeddings=self.embeddings),
                self.ragas_modules['LLMContextPrecisionWithReference'](llm=self.evaluator_llm),  # LLM 기반 Context Precision
                self.ragas_modules['Faithfulness'](llm=self.evaluator_llm),  # SingleTurnSample 방식
                self.ragas_modules['LLMContextRecall'](llm=self.evaluator_llm)  # Context Recall 추가
            ],
            llm=self.evaluator_llm
        )
        
        return results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def save_results(self, output_path=None):
    """평가 결과를 JSON 파일로 저장합니다."""
    if not self.evaluation_results:
        print("❌ 저장할 평가 결과가 없습니다.")
        return None
    
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # ./farmer/sales/data 디렉토리에 저장
        data_dir = "./farmer/sales/data"
        os.makedirs(data_dir, exist_ok=True)  # 디렉토리가 없으면 생성
        output_path = os.path.join(data_dir, f"ragas_evaluation_results_{timestamp}.json")
    
    # 평균 점수 계산
    context_precision_scores = []
    faithfulness_scores = []
    answer_relevancy_scores = []
    context_recall_scores = []
    
    for result in self.evaluation_results:
        if 'individual_ragas_score' in result and result['individual_ragas_score']:
            scores = result['individual_ragas_score']
            if 'context_precision' in scores:
                context_precision_scores.append(scores['context_precision'])
            if 'faithfulness' in scores:
                faithfulness_scores.append(scores['faithfulness'])
            if 'answer_relevancy' in scores:
                answer_relevancy_scores.append(scores['answer_relevancy'])
            if 'context_recall' in scores:
                context_recall_scores.append(scores['context_recall'])
    
    # 평균 점수 계산
    avg_scores = {}
    if context_precision_scores:
        avg_scores['context_precision'] = sum(context_precision_scores) / len(context_precision_scores)
    if faithfulness_scores:
        avg_scores['faithfulness'] = sum(faithfulness_scores) / len(faithfulness_scores)
    if answer_relevancy_scores:
        avg_scores['answer_relevancy'] = sum(answer_relevancy_scores) / len(answer_relevancy_scores)
    if context_recall_scores:
        avg_scores['context_recall'] = sum(context_recall_scores) / len(context_recall_scores)
    
    # 전체 결과 구성
    full_results = {
        'evaluation_summary': {
            'total_questions': len(self.evaluation_results),
            'successful_evaluations': len([r for r in self.evaluation_results if r.get('individual_ragas_score')]),
            'failed_evaluations': len(self.evaluation_results) - len([r for r in self.evaluation_results if r.get('individual_ragas_score')]),
            'average_scores': avg_scores,
            'evaluation_timestamp': datetime.now().isoformat()
        },
        'detailed_results': self.evaluation_results
    }
    
    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 평가 결과 저장 완료: {output_path}")
    print(f"📊 평균 점수도 포함되어 저장되었습니다.")
    return output_path

def print_summary(self):
    """평가 결과 요약을 출력합니다."""
    if not self.evaluation_results:
        print("❌ 평가 결과가 없습니다.")
        return
    
    print("\n" + "="*60)
    print("📊 RAGAS 평가 결과 요약")
    print("="*60)
    
    # 전체 통계
    total_questions = len(self.evaluation_results)
    successful_evaluations = len([r for r in self.evaluation_results if r.get('individual_ragas_score')])
    
    print(f"총 질문 수: {total_questions}")
    print(f"성공한 평가: {successful_evaluations}")
    print(f"실패한 평가: {total_questions - successful_evaluations}")
    
    if successful_evaluations > 0:
        # RAGAS 점수 평균 계산
        context_precision_scores = [r['individual_ragas_score']['context_precision'] for r in self.evaluation_results if r.get('individual_ragas_score')]
        faithfulness_scores = [r['individual_ragas_score']['faithfulness'] for r in self.evaluation_results if r.get('individual_ragas_score')]
        answer_relevancy_scores = [r['individual_ragas_score']['answer_relevancy'] for r in self.evaluation_results if r.get('individual_ragas_score')]
        context_recall_scores = [r['individual_ragas_score']['context_recall'] for r in self.evaluation_results if r.get('individual_ragas_score')]
        
        print(f"\n📈 RAGAS 점수 평균:")
        print(f"  Context Precision: {sum(context_precision_scores)/len(context_precision_scores):.3f}")
        print(f"  Faithfulness: {sum(faithfulness_scores)/len(faithfulness_scores):.3f}")
        print(f"  Answer Relevancy: {sum(answer_relevancy_scores)/len(answer_relevancy_scores):.3f}")
        print(f"  Context Recall: {sum(context_recall_scores)/len(context_recall_scores):.3f}")
        
    
    print("="*60)


# 테스트 실행 코드
if __name__ == "__main__":
    print("🧪 SalesRAGAS 테스트 실행 시작")
    print("="*60)
    
    try:
        # 평가기 초기화 (함수들을 클래스에 바인딩)
        evaluator = type('Evaluator', (), {})()
        
        # 함수들을 evaluator 객체에 바인딩
        evaluator._setup_models = lambda: _setup_models(evaluator)
        evaluator._create_test_questions = lambda: _create_test_questions(evaluator)
        evaluator._load_csv_data = lambda: _load_csv_data(evaluator)
        evaluator.run_sales_agent_evaluation = lambda question: run_sales_agent_evaluation(evaluator, question)
        evaluator._format_context = lambda context: _format_context(evaluator, context)
        evaluator._is_milvus_connected = lambda: _is_milvus_connected(evaluator)
        evaluator.evaluate_single_question = lambda test_case: evaluate_single_question(evaluator, test_case)
        evaluator._evaluate_single_ragas_simple = lambda question, answer, context, reference="": _evaluate_single_ragas_simple(evaluator, question, answer, context, reference)
        evaluator.run_full_evaluation = lambda batch_size=3: run_full_evaluation(evaluator, batch_size)
        evaluator.create_ragas_dataset = lambda: create_ragas_dataset(evaluator)
        evaluator.run_ragas_evaluation = lambda: run_ragas_evaluation(evaluator)
        evaluator.save_results = lambda output_path=None: save_results(evaluator, output_path)
        evaluator.print_summary = lambda: print_summary(evaluator)
        
        __init__(evaluator)
        
        # 동기 평가 실행
        results = run_full_evaluation(evaluator)
        
        if results:
            print_summary(evaluator)
            save_results(evaluator)
        else:
            print("❌ 평가 결과가 없습니다.")
            
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()