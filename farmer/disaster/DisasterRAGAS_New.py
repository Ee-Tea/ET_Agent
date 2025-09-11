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
    Faithfulness,
    LLMContextRecall
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
from DisasterAgent_LLM import run

def __init__(self, csv_path="./golden_dataset_open_multi.csv"):
    """DisasterAgent_LLM RAGAS 평가기 초기화"""
    print(f"🔧 DisasterRAGAS 평가기 초기화 시작...")
    print(f"📁 CSV 파일 경로: {csv_path}")
    
    try:
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
        
    except Exception as e:
        print(f"❌ 평가기 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        raise

def _setup_models(self):
    """공식 문서에 따른 모델 설정"""
    print("🔧 RAGAS 래퍼 가져오는 중...")
    try:
        LangchainLLMWrapper, LangchainEmbeddingsWrapper = get_ragas_wrappers()
        print("✅ RAGAS 래퍼 가져오기 완료")
    except Exception as e:
        print(f"❌ RAGAS 래퍼 가져오기 실패: {e}")
        raise

    try:
        # LLM 설정
        print("🤖 OpenAI LLM 설정 중...")
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0.1,  # 매우 일관된 질문 생성
            api_key=OPENAI_API_KEY
        )
        print("✅ OpenAI LLM 설정 완료")
        
        print("🔧 LLM 래퍼 설정 중...")
        self.evaluator_llm = LangchainLLMWrapper(self.llm)
        print("✅ LLM 래퍼 설정 완료")
        
        # 임베딩 모델 설정 (한국어 특화)
        print("🔤 임베딩 모델 설정 중...")
        self.embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'},  # GPU가 있으면 'cuda'로 변경
                encode_kwargs={'normalize_embeddings': True}
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
    """CSV 파일에서 question, ground_truth, contexts 컬럼을 읽어옵니다."""
    print(f"📖 CSV 파일 읽기 시작: {self.csv_path}")
    
    try:
        # CSV 파일 읽기
        print("📄 pandas로 CSV 파일 읽는 중...")
        df = pd.read_csv(self.csv_path)
        print(f"✅ CSV 파일 읽기 완료: {len(df)}행")
        
        # 컬럼 확인
        print(f"📋 CSV 컬럼들: {list(df.columns)}")
        
        # 필요한 컬럼 확인
        required_columns = ['question', 'ground_truth', 'contexts']
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
            # contexts가 문자열로 저장된 리스트인 경우 파싱
            contexts_str = str(row['contexts']).strip()
            if contexts_str.startswith('[') and contexts_str.endswith(']'):
                try:
                    import ast
                    contexts = ast.literal_eval(contexts_str)
                except:
                    contexts = [contexts_str]
            else:
                contexts = [contexts_str]
            
            test_cases.append({
                'question': str(row['question']).strip(),
                'reference': str(row['ground_truth']).strip(),
                'contexts': contexts
            })
            if i < 3:  # 처음 3개만 출력
                print(f"  📝 질문 {i+1}: {str(row['question']).strip()[:50]}...")
        
        print(f"✅ CSV 데이터 로드 완료: {len(test_cases)}개 질문")
        return test_cases
        
    except Exception as e:
        print(f"❌ CSV 파일 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return []

def _extract_context_from_graph_state(self, result_state):
    """DisasterAgent 그래프 상태에서 컨텍스트 정보 추출"""
    context_parts = []
    
    # DB 컨텍스트 추출
    db_context = result_state.get('db_context', '')
    if db_context and db_context.strip() and "관련 문서를 찾을 수 없습니다" not in db_context:
        context_parts.append(f"DB 검색 결과: {db_context[:500]}...")
    
    # 웹 컨텍스트 추출
    web_context = result_state.get('web_context', '')
    if web_context and web_context.strip() and "[웹 검색 결과 없음]" not in web_context:
        context_parts.append(f"웹 검색 결과: {web_context[:500]}...")
    
    # 최종 컨텍스트 추출
    final_context = result_state.get('context', '')
    if final_context and final_context.strip():
        context_parts.append(f"최종 컨텍스트: {final_context[:500]}...")
    
    # 검색된 문서 정보 추출
    retrieved_docs = result_state.get('retrieved_docs', [])
    if retrieved_docs:
        doc_info = []
        for i, doc in enumerate(retrieved_docs[:5], 1):  # 최대 5개 문서만
            meta = getattr(doc, "metadata", {})
            fname = meta.get("file_name") or meta.get("source") or f"문서{i}"
            page = meta.get("page")
            doc_type = meta.get("type") or "text"
            doc_info.append(f"[문서{i}][{doc_type}][{fname}{f' p.{page}' if page else ''}]")
        
        if doc_info:
            context_parts.append(f"검색된 문서: {' | '.join(doc_info)}")
    
    # 컨텍스트를 더 명확하게 구조화 (RAGAS 점수 향상)
    if context_parts:
        return "농업재해 정보:\n" + "\n".join(context_parts)
    else:
        return ""

def run_disaster_agent_evaluation(self, question):
    """DisasterAgent_LLM을 실행하여 답변과 컨텍스트 수집"""
    print(f"🤖 DisasterAgent_LLM 실행 시작: {question[:50]}...")
    
    try:
        # DisasterAgent_LLM 실행 (비동기 함수를 동기적으로 실행)
        print("🔄 DisasterAgent_LLM 실행 중...")
        initial_state = {"query": question}
        
        # 비동기 함수를 동기적으로 실행
        import asyncio
        result_state = asyncio.run(run(initial_state))
        print("✅ DisasterAgent_LLM 실행 완료")
        
        # 결과 추출
        print("📊 결과 추출 중...")
        answer = result_state.get('agent_answer', '')
        
        print(f"📝 답변 길이: {len(answer)}자")
        print(f"💬 DisasterAgent_LLM 응답: {answer}")
        
        # 컨텍스트 정보 추출 (DisasterAgent_LLM의 그래프 상태에서)
        context_str = self._extract_context_from_graph_state(result_state)
        print(f"📄 컨텍스트 길이: {len(context_str)}자")
        if context_str:
            print(f"📋 컨텍스트 내용: {context_str[:200]}...")
        else:
            print("⚠️ 컨텍스트가 비어있음")
        
        print("✅ DisasterAgent_LLM 평가 완료")
        return {
            'answer': answer,
            'context': context_str,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ DisasterAgent_LLM 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return {
            'answer': f"오류 발생: {str(e)}",
            'context': "",
            'success': False
        }

def evaluate_single_question(self, test_case):
    """단일 질문에 대한 평가 실행 (개별 RAGAS 평가 포함) - 동기"""
    
    question = test_case['question']
    reference = test_case.get('reference', '')
    contexts = test_case.get('contexts', [])
    
    print(f"\n📝 질문 평가 시작: {question[:50]}...")
    
    # DisasterAgent_LLM 실행
    agent_result = self.run_disaster_agent_evaluation(question)
    
    if not agent_result['success']:
        print(f"❌ DisasterAgent_LLM 실행 실패")
        return None
    
    # 컨텍스트가 제공된 경우 사용, 그렇지 않으면 빈 문자열
    context_for_ragas = agent_result['context'] if agent_result['context'] else ""
    
    # 개별 RAGAS 평가 실행 (동기)
    individual_ragas_score = self._evaluate_single_ragas_simple(
        question,
        agent_result['answer'],
        context_for_ragas,
        reference
    )
    
    # 평가 결과 구성
    evaluation_result = {
        'question': question,
        'reference': reference,
        'answer': agent_result['answer'],
        'context': context_for_ragas,
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
        # 4개 RAGAS 평가 모두 순차 처리
        def evaluate_context_precision():
            try:
                print("🔍 Context Precision 평가 중...")
                # Context Precision은 retrieved_contexts가 필요하므로 컨텍스트가 있을 때만 평가
                if not context or not context.strip():
                    print("   - ⚠️ Context Precision 건너뜀: 컨텍스트 없음")
                    return ("context_precision", 0.0)
                
                context_precision_scorer = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)
                
                # 컨텍스트가 비어있거나 None인 경우 처리
                contexts = [context] if context and context.strip() else [""]
                
                context_sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
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
                faithfulness_scorer = Faithfulness(llm=self.evaluator_llm)
                
                # 컨텍스트가 비어있거나 None인 경우 처리
                contexts = [context] if context and context.strip() else [""]
                
                faithfulness_sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
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
                # ResponseRelevancy는 user_input과 response만으로 평가 가능 (retrieved_contexts 불필요)
                response_relevancy_scorer = ResponseRelevancy(
                    llm=self.evaluator_llm, 
                    embeddings=self.embeddings
                )
                
                # ResponseRelevancy는 retrieved_contexts 없이도 평가 가능
                sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=[]  # 빈 리스트로 설정
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
                
                context_recall_scorer = LLMContextRecall(llm=self.evaluator_llm)
                
                # 컨텍스트가 비어있거나 None인 경우 처리
                contexts = [context] if context and context.strip() else [""]
                
                recall_sample = SingleTurnSample(
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
        
        print("   - ✅ 4개 순차 평가 완료!")
        return scores
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def run_full_evaluation(self, batch_size=3, max_questions=None):
    """전체 테스트 케이스에 대한 평가 실행 (개별 RAGAS 포함) - 동기"""
    
    print(f"\n🚀 전체 평가 시작!")
    print(f"📊 평가할 질문 수: {len(self.test_questions)}")
    print(f"⚙️ 배치 크기: {batch_size}")
    
    if not self.test_questions:
        print("❌ 평가할 질문이 없습니다.")
        return []
    
    # 테스트할 질문 수 제한 (개발/테스트용)
    questions_to_test = self.test_questions
    if max_questions and max_questions < len(self.test_questions):
        questions_to_test = self.test_questions[:max_questions]
        print(f"🔧 테스트 질문 수를 {max_questions}개로 제한합니다.")
    
    # 각 질문에 대해 평가 실행
    for i, test_case in enumerate(questions_to_test, 1):
        print(f"\n{'='*60}")
        print(f"📝 질문 {i}/{len(questions_to_test)} 평가 시작")
        print(f"📊 진행률: {i/len(questions_to_test)*100:.1f}%")
        print(f"{'='*60}")
        
        try:
            result = self.evaluate_single_question(test_case)
            if result:
                self.evaluation_results.append(result)
                print(f"✅ 질문 {i} 평가 완료")
                
                # 현재까지의 성공률 표시
                success_rate = len(self.evaluation_results) / i * 100
                print(f"📈 현재 성공률: {success_rate:.1f}% ({len(self.evaluation_results)}/{i})")
            else:
                print(f"❌ 질문 {i} 평가 실패")
        except KeyboardInterrupt:
            print(f"\n⚠️ 사용자에 의해 중단되었습니다.")
            print(f"📊 중단 시점: {i-1}/{len(questions_to_test)} 완료")
            break
        except Exception as e:
            print(f"❌ 질문 {i} 평가 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 오류가 발생해도 다음 질문으로 계속 진행
            continue
        
        # 각 질문 사이에 대기 (API 제한 방지)
        if i < len(questions_to_test):  # 마지막 질문이 아닌 경우에만 대기
            print(f"⏳ 다음 질문까지 2초 대기...")
            time.sleep(2)
    
    print(f"\n🎉 전체 평가 완료!")
    print(f"📊 성공한 평가: {len(self.evaluation_results)}개")
    print(f"📊 실패한 평가: {len(questions_to_test) - len(self.evaluation_results)}개")
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
        results = evaluate(
            dataset,
            metrics=[
                ResponseRelevancy(llm=self.evaluator_llm, embeddings=self.embeddings),
                LLMContextPrecisionWithoutReference(llm=self.evaluator_llm),  # LLM 기반 Context Precision
                Faithfulness(llm=self.evaluator_llm),  # SingleTurnSample 방식
                LLMContextRecall(llm=self.evaluator_llm)  # Context Recall 추가
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
        # ./farmer/disaster/data 디렉토리에 저장
        data_dir = "./farmer/disaster/data"
        os.makedirs(data_dir, exist_ok=True)  # 디렉토리가 없으면 생성
        output_path = os.path.join(data_dir, f"disaster_ragas_evaluation_results_{timestamp}.json")
    
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
    print("📊 DisasterAgent_LLM RAGAS 평가 결과 요약")
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
    print("🧪 DisasterRAGAS 테스트 실행 시작")
    print("="*60)
    
    try:
        # 평가기 초기화 (함수들을 클래스에 바인딩)
        evaluator = type('Evaluator', (), {})()
        
        # 함수들을 evaluator 객체에 바인딩
        evaluator._setup_models = lambda: _setup_models(evaluator)
        evaluator._create_test_questions = lambda: _create_test_questions(evaluator)
        evaluator._load_csv_data = lambda: _load_csv_data(evaluator)
        evaluator.run_disaster_agent_evaluation = lambda question: run_disaster_agent_evaluation(evaluator, question)
        evaluator._extract_context_from_graph_state = lambda result_state: _extract_context_from_graph_state(evaluator, result_state)
        evaluator.evaluate_single_question = lambda test_case: evaluate_single_question(evaluator, test_case)
        evaluator._evaluate_single_ragas_simple = lambda question, answer, context, reference="": _evaluate_single_ragas_simple(evaluator, question, answer, context, reference)
        evaluator.run_full_evaluation = lambda batch_size=3, max_questions=None: run_full_evaluation(evaluator, batch_size, max_questions)
        evaluator.create_ragas_dataset = lambda: create_ragas_dataset(evaluator)
        evaluator.run_ragas_evaluation = lambda: run_ragas_evaluation(evaluator)
        evaluator.save_results = lambda output_path=None: save_results(evaluator, output_path)
        evaluator.print_summary = lambda: print_summary(evaluator)
        
        __init__(evaluator)
        
        # 동기 평가 실행 (전체 50개 질문)
        results = run_full_evaluation(evaluator, max_questions=50)
        
        if results:
            print_summary(evaluator)
            save_results(evaluator)
        else:
            print("❌ 평가 결과가 없습니다.")
            
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
