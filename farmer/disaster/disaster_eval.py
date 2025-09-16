# =============================================================================
# 환경 설정
# =============================================================================
# 프로젝트 루트 경로 설정 (.env가 있는 위치)
PROJECT_ROOT = r"C:\FinalPrj\ET_Agent"

# 평가할 데이터셋 파일 경로 설정
CSV_FILE_PATH = "./farmer/disaster/data/golden_dataset_open_single_20250916_181133.csv"

# 결과 저장 설정
OUTPUT_DIRECTORY = "./farmer/disaster/data"                # 결과 저장 디렉토리
OUTPUT_FORMAT = "csv"                                   # 출력 형식: "json" 또는 "csv"
OUTPUT_FILENAME_PREFIX = "disaster_deepeval_results"       # 파일명 접두사 자동으로 타임스탬프(YYYYMMDD_HHMM)가 붙음

# 골든셋 컬럼명 설정 (각 데이터셋에 맞게 오른쪽 수정)
GOLDEN_DATASET_COLUMNS = {
    'question': 'question',               # 사용자 질문 컬럼명
    'reference': 'contexts',      # 참조 컨텍스트 컬럼명
    'answer': 'ground_truth'                   # 참조 답변 컬럼명
}

# 평가 설정
MAX_EVALUATION_ROWS = None      # 평가할 최대 행 수 (None이면 전체 평가)
FILTER_BY_THRESHOLD = True      # True: 모든 메트릭 임계값을 넘는 것만 저장, False: 전체 저장

# 메트릭 임계값 설정
METRIC_THRESHOLDS = {
    'input_quality': 0.8,                   # 입력 질문 품질
    'reference_quality': 0.8,               # 참조 컨텍스트 품질
    'answer_quality': 0.8,                  # 답변 품질
    'input_reference_alignment': 0.8,       # 입력-참조 정렬도
    'reference_answer_alignment': 0.8,      # 참조-답변 정렬도
    'input_answer_alignment': 0.8           # 입력-답변 정렬도
}

# =============================================================================
# 메트릭 평가 단계 설정 (작물 재해 대응 전용)
# =============================================================================
EVALUATION_STEPS = {
    'input_quality': [
        "질문이 농작물 재해 대응(태풍, 폭우, 가뭄, 냉해, 폭염 등)과 직접적으로 관련이 있는지 엄격하게 평가합니다.",
        "재해 유형(예: 태풍, 호우, 냉해 등)이 명확히 언급되었는지 확인합니다.",
        "질문이 모호하거나 재해와 직접적인 관련이 없는 경우 점수를 크게 감점합니다.",
        "평가한 점수에 대한 이유를 한국어로 간단히 설명해주세요."
    ],
    'reference_quality': [
        "참조 컨텍스트가 해당 재해 상황에서 작물 보호 및 대응 방법을 구체적으로 포함하는지 확인합니다.",
        "구체적인 대응 단계(예: 차광망 설치, 배수로 정비, 방풍망 설치, 병해충 예방 등)가 포함되어 있는지 평가합니다.",
        "정보가 부족하거나 일반적 수준에 그친 경우 점수를 크게 감점합니다.",
        "평가한 점수에 대한 이유를 한국어로 간단히 설명해주세요."
    ],
    'answer_quality': [
        "답변이 해당 재해 상황에 대해 작물이 실제로 활용할 수 있는 대응 방법을 간결하고 명확하게 제시했는지 평가합니다.",
        "불필요하게 장황하거나 모호한 설명이 없는지 확인합니다.",
        "전문가가 조언하는 어조로 신뢰감 있게 작성되었는지 평가합니다.",
        "문장 구조가 명확하고 실제 농민이 이해하기 쉬운지 확인합니다.",
        "평가한 점수에 대한 이유를 한국어로 간단히 설명해주세요."
    ],
    'input_reference_alignment': [
        "사용자가 언급한 재해 유형(태풍, 폭우, 냉해 등)에 대해 참조 컨텍스트가 직접적인 대응 방안을 제공하는지 평가합니다.",
        "질문에서 언급된 작물과 참조의 작물이 일치하거나 유사한 상황인지 확인합니다.",
        "관련 없는 작물이나 재해 정보가 포함된 경우 점수를 크게 감점합니다.",
        "사용자의 질문 의도와 참조 내용이 불일치할 경우 엄격하게 평가합니다.",
        "평가한 점수에 대한 이유를 한국어로 간단히 설명해주세요."
    ],
    'reference_answer_alignment': [
        "답변이 참조 컨텍스트에 기반해 작성되었는지 엄격하게 확인합니다.",
        "참조에 없는 대응 방법이 추가된 경우 '환각'으로 간주하고 점수를 크게 감점합니다.",
        "참조의 핵심 대응 방법이 답변에서 누락되지 않았는지 확인합니다.",
        "참조 정보를 왜곡하거나 과장하지 않았는지 확인합니다.",
        "참조 정보를 충실하게 활용했는지 평가합니다.",
        "평가한 점수에 대한 이유를 한국어로 간단히 설명해주세요."
    ],
    'input_answer_alignment': [
        "사용자의 질문(특정 재해 상황과 작물)에 대해 답변이 직접적이고 구체적인 대응 방법을 제시했는지 확인합니다.",
        "질문에서 요구한 맥락과 맞지 않는 일반적이거나 불필요한 답변인 경우 점수를 크게 감점합니다.",
        "질문 의도와 답변 내용이 명확히 일치하는지 평가합니다.",
        "평가한 점수에 대한 이유를 한국어로 간단히 설명해주세요."
    ]
}


# DeepEval 모델 설정
DEEPEVAL_MODEL_NAME = "gpt-4o-mini"

# =============================================================================
# 라이브러리 import
# =============================================================================
# 표준 라이브러리
import json
from datetime import datetime

# 서드파티 라이브러리
import pandas as pd
import os
from dotenv import load_dotenv
env_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(env_path)

# DeepEval 관련 모듈들을 지연 로딩으로 import
def import_deepeval_modules():
    """DeepEval 관련 모듈들을 지연 로딩으로 import"""
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
        
        return {
            'GEval': GEval,
            'LLMTestCase': LLMTestCase,
            'LLMTestCaseParams': LLMTestCaseParams
        }
    except ImportError as e:
        print(f"❌ DeepEval 모듈 import 실패: {e}")
        raise


class DeepEvaluator:
    def __init__(self, csv_path=None):
        """골든 데이터셋 DeepEval 평가기 초기화"""
        print(f"🔧 골든 데이터셋 DeepEval 평가기 초기화 시작...")
        
        # CSV 경로 설정
        self.csv_path = csv_path if csv_path else CSV_FILE_PATH
        print(f"📁 CSV 파일 경로: {self.csv_path}")
        
        try:
            # DeepEval 모듈 설정
            print("🤖 DeepEval 모델 설정 중...")
            self._setup_models()
            print("✅ 모델 설정 완료")
            
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
        """DeepEval 모델 설정"""
        print("📦 DeepEval 모듈들 로딩 중...")
        try:
            self.deepeval_modules = import_deepeval_modules()
            print("✅ DeepEval 모듈들 로딩 완료")
        except Exception as e:
            print(f"❌ DeepEval 모듈 로딩 실패: {e}")
            raise

        try:
            # DeepEval은 문자열로 모델명을 전달해야 함
            print("🤖 DeepEval 모델 설정 중...")
            self.model_name = DEEPEVAL_MODEL_NAME  # 전역변수에서 모델명 가져오기
            print("✅ DeepEval 모델 설정 완료")
            
            # 커스텀 G-Eval 메트릭들 설정
            print("🔧 커스텀 G-Eval 메트릭들 설정 중...")
            self._setup_custom_metrics()
            print("✅ 커스텀 메트릭 설정 완료")
            
        except Exception as e:
            print(f"❌ 모델 설정 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _setup_custom_metrics(self):
        """커스텀 G-Eval 메트릭들 설정"""
        
        # 1. 입력 질문 품질 (Input Quality)
        self.input_quality_metric = self.deepeval_modules['GEval'](
            name="Input Quality",
            evaluation_steps=EVALUATION_STEPS['input_quality'],
            evaluation_params=[
                self.deepeval_modules['LLMTestCaseParams'].INPUT
            ],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['input_quality']
        )
        
        # 2. 참조 컨텍스트 품질 (Reference Quality)
        self.reference_quality_metric = self.deepeval_modules['GEval'](
            name="Reference Quality",
            evaluation_steps=EVALUATION_STEPS['reference_quality'],
            evaluation_params=[
                self.deepeval_modules['LLMTestCaseParams'].RETRIEVAL_CONTEXT
            ],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['reference_quality']
        )
        
        # 3. 답변 품질 (Answer Quality)
        self.answer_quality_metric = self.deepeval_modules['GEval'](
            name="Answer Quality",
            evaluation_steps=EVALUATION_STEPS['answer_quality'],
            evaluation_params=[
                self.deepeval_modules['LLMTestCaseParams'].EXPECTED_OUTPUT
            ],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['answer_quality']
        )
        
        # 4. 입력-참조 정렬도 (Input-Reference Alignment)
        self.input_reference_alignment_metric = self.deepeval_modules['GEval'](
            name="Input-Reference Alignment",
            evaluation_steps=EVALUATION_STEPS['input_reference_alignment'],
            evaluation_params=[
                self.deepeval_modules['LLMTestCaseParams'].INPUT,
                self.deepeval_modules['LLMTestCaseParams'].RETRIEVAL_CONTEXT
            ],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['input_reference_alignment']
        )
        
        # 5. 참조-답변 정렬도 (Reference-Answer Alignment)
        self.reference_answer_alignment_metric = self.deepeval_modules['GEval'](
            name="Reference-Answer Alignment",
            evaluation_steps=EVALUATION_STEPS['reference_answer_alignment'],
            evaluation_params=[
                self.deepeval_modules['LLMTestCaseParams'].RETRIEVAL_CONTEXT,
                self.deepeval_modules['LLMTestCaseParams'].EXPECTED_OUTPUT
            ],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['reference_answer_alignment']
        )
        
        # 6. 입력-답변 정렬도 (Input-Answer Alignment)
        self.input_answer_alignment_metric = self.deepeval_modules['GEval'](
            name="Input-Answer Alignment",
            evaluation_steps=EVALUATION_STEPS['input_answer_alignment'],
            evaluation_params=[
                self.deepeval_modules['LLMTestCaseParams'].INPUT,
                self.deepeval_modules['LLMTestCaseParams'].EXPECTED_OUTPUT
            ],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['input_answer_alignment']
        )
        
        print("✅ 6개 커스텀 G-Eval 메트릭 설정 완료")

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
        """CSV 파일에서 컬럼을 읽어옵니다."""
        print(f"📖 CSV 파일 읽기 시작: {self.csv_path}")
        
        try:
            df = pd.read_csv(self.csv_path)
            print(f"✅ CSV 파일 읽기 완료: {len(df)}행")
            
            # 전역변수에서 컬럼명 가져오기
            required_columns = list(GOLDEN_DATASET_COLUMNS.values())
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ 필수 컬럼이 없습니다: {missing_columns}")
                print(f"💡 사용 가능한 컬럼: {list(df.columns)}")
                print(f"💡 설정된 컬럼명: {GOLDEN_DATASET_COLUMNS}")
                raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
            
            test_cases = []
            max_rows = MAX_EVALUATION_ROWS if MAX_EVALUATION_ROWS else len(df)
            actual_rows = min(max_rows, len(df))
            
            for i, (_, row) in enumerate(df.iterrows()):
                if i >= actual_rows:
                    break
                    
                test_cases.append({
                    'question': str(row[GOLDEN_DATASET_COLUMNS['question']]).strip(),
                    'reference': str(row[GOLDEN_DATASET_COLUMNS['reference']]).strip(),
                    'answer': str(row[GOLDEN_DATASET_COLUMNS['answer']]).strip()
                })
                if i < 3:
                    print(f"  📝 질문 {i+1}: {str(row[GOLDEN_DATASET_COLUMNS['question']]).strip()[:50]}...")
            
            print(f"✅ CSV 데이터 로드 완료: {len(test_cases)}개 질문 (전체 {len(df)}개 중 {actual_rows}개 평가)")
            return test_cases
            
        except Exception as e:
            print(f"❌ CSV 파일 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return []


    def evaluate_single_question(self, test_case):
        """단일 질문에 대한 DeepEval 평가 실행 (골든 데이터셋 평가)"""
        
        question = test_case['question']
        answer = test_case.get('answer', '')
        reference = test_case.get('reference', '')
        
        print(f"\n📝 질문 평가 시작: {question[:50]}...")
        
        # DeepEval 테스트 케이스 생성 (골든 데이터셋 평가용)
        test_case_obj = self.deepeval_modules['LLMTestCase'](
            input=question,
            expected_output=answer,  # 골든 데이터셋의 답변
            retrieval_context=[reference] if reference else [""]  # 골든 데이터셋의 컨텍스트
        )
        
        # 모든 메트릭 평가
        print("🔍 DeepEval G-Eval 메트릭 평가 시작...")
        metrics = [
            self.input_quality_metric,
            self.reference_quality_metric,
            self.answer_quality_metric,
            self.input_reference_alignment_metric,
            self.reference_answer_alignment_metric,
            self.input_answer_alignment_metric
        ]
        
        metric_scores = {}
        for metric in metrics:
            try:
                print(f"  📊 {metric.name} 평가 중...")
                metric.measure(test_case_obj)
                metric_scores[metric.name] = {
                    'score': metric.score,
                    'reason': metric.reason,
                    'success': metric.success
                }
                print(f"  ✅ {metric.name}: {metric.score:.3f} ({'PASS' if metric.success else 'FAIL'})")
            except Exception as e:
                print(f"  ❌ {metric.name} 평가 실패: {e}")
                metric_scores[metric.name] = {
                    'score': 0.0,
                    'reason': f"평가 실패: {str(e)}",
                    'success': False
                }
        
        # 평가 결과 구성
        evaluation_result = {
            'question': question,
            'answer': answer,
            'reference': reference,
            'metric_scores': metric_scores,
            'overall_score': sum([m['score'] for m in metric_scores.values()]) / len(metric_scores),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ 질문 평가 완료 (전체 점수: {evaluation_result['overall_score']:.3f})")
        return evaluation_result

    def run_full_evaluation(self, batch_size=3):
        """전체 테스트 케이스에 대한 평가 실행"""
        
        print(f"\n🚀 전체 DeepEval 평가 시작!")
        print(f"📊 평가할 질문 수: {len(self.test_questions)}")
        
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
        
        print(f"\n🎉 전체 평가 완료!")
        print(f"📊 성공한 평가: {len(self.evaluation_results)}개")
        return self.evaluation_results

    def save_results(self, output_path=None):
        """평가 결과를 JSON 또는 CSV 파일로 저장합니다."""
        if not self.evaluation_results:
            print("❌ 저장할 평가 결과가 없습니다.")
            return None
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
            file_extension = "json" if OUTPUT_FORMAT.lower() == "json" else "csv"
            output_path = os.path.join(OUTPUT_DIRECTORY, f"{OUTPUT_FILENAME_PREFIX}_{timestamp}.{file_extension}")
        
        # 결과 필터링 (모든 메트릭 임계값을 넘는 것만 저장)
        if FILTER_BY_THRESHOLD:
            filtered_results = []
            for result in self.evaluation_results:
                metric_scores = result.get('metric_scores', {})
                
                # 모든 메트릭이 임계값을 넘는지 확인
                all_above_threshold = True
                for metric_name, threshold in METRIC_THRESHOLDS.items():
                    if metric_name in metric_scores:
                        score = metric_scores[metric_name].get('score', 0)
                        if score < threshold:
                            all_above_threshold = False
                            break
                    else:
                        all_above_threshold = False
                        break
                
                if all_above_threshold:
                    filtered_results.append(result)
            
            print(f"🔍 모든 메트릭 임계값을 넘는 것만 필터링: {len(self.evaluation_results)}개 중 {len(filtered_results)}개 저장")
            results_to_save = filtered_results
        else:
            results_to_save = self.evaluation_results
            print(f"📊 전체 결과 저장: {len(results_to_save)}개")
        
        # 평균 점수 계산
        metric_averages = {}
        for metric_name in ['Input Quality', 'Reference Quality', 'Answer Quality', 
                           'Input-Reference Alignment', 'Reference-Answer Alignment', 'Input-Answer Alignment']:
            scores = [r['metric_scores'][metric_name]['score'] for r in results_to_save 
                     if metric_name in r['metric_scores']]
            if scores:
                metric_averages[metric_name] = sum(scores) / len(scores)
        
        if OUTPUT_FORMAT.lower() == "json":
            # JSON 형식으로 저장
            full_results = {
                'evaluation_summary': {
                    'total_questions': len(self.evaluation_results),
                    'successful_evaluations': len([r for r in self.evaluation_results if r.get('metric_scores')]),
                    'filtered_questions': len(results_to_save),
                    'filter_applied': FILTER_BY_THRESHOLD,
                    'metric_thresholds': METRIC_THRESHOLDS if FILTER_BY_THRESHOLD else None,
                    'average_scores': metric_averages,
                    'overall_average': sum([r['overall_score'] for r in results_to_save]) / len(results_to_save) if results_to_save else 0,
                    'evaluation_timestamp': datetime.now().isoformat()
                },
                'detailed_results': results_to_save
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, ensure_ascii=False, indent=2)
        
        elif OUTPUT_FORMAT.lower() == "csv":
            # CSV 형식으로 저장
            csv_data = []
            for result in results_to_save:
                # 모든 메트릭이 임계값을 넘는지 확인
                metric_scores = result.get('metric_scores', {})
                all_pass = True
                for metric_name, threshold in METRIC_THRESHOLDS.items():
                    if metric_name in metric_scores:
                        score = metric_scores[metric_name].get('score', 0)
                        if score < threshold:
                            all_pass = False
                            break
                    else:
                        all_pass = False
                        break
                
                row = {
                    'question': result['question'],
                    'answer': result['answer'],
                    'reference': result['reference'],
                    'overall_score': result['overall_score'],
                    'all_metrics_pass': "PASS" if all_pass else "FAIL",
                    'timestamp': result['timestamp']
                }
                
                # 각 메트릭 점수 추가
                for metric_name in ['Input Quality', 'Reference Quality', 'Answer Quality', 
                                   'Input-Reference Alignment', 'Reference-Answer Alignment', 'Input-Answer Alignment']:
                    if metric_name in result['metric_scores']:
                        score = result['metric_scores'][metric_name]['score']
                        threshold = METRIC_THRESHOLDS.get(metric_name.lower().replace(' ', '_').replace('-', '_'), 0.9)
                        row[f'{metric_name}_score'] = score
                        row[f'{metric_name}_threshold'] = threshold
                        row[f'{metric_name}_pass'] = "PASS" if score >= threshold else "FAIL"
                        row[f'{metric_name}_reason'] = result['metric_scores'][metric_name]['reason']
                    else:
                        row[f'{metric_name}_score'] = 0.0
                        row[f'{metric_name}_threshold'] = METRIC_THRESHOLDS.get(metric_name.lower().replace(' ', '_').replace('-', '_'), 0.9)
                        row[f'{metric_name}_pass'] = "FAIL"
                        row[f'{metric_name}_reason'] = "평가 실패"
                
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        else:
            print(f"❌ 지원하지 않는 출력 형식입니다: {OUTPUT_FORMAT}")
            return None
        
        print(f"✅ 평가 결과 저장 완료 ({OUTPUT_FORMAT.upper()}): {output_path}")
        return output_path

    def print_summary(self):
        """평가 결과 요약을 출력합니다."""
        if not self.evaluation_results:
            print("❌ 평가 결과가 없습니다.")
            return
        
        print("\n" + "="*60)
        print("📊 DeepEval G-Eval 평가 결과 요약")
        print("="*60)
        
        total_questions = len(self.evaluation_results)
        successful_evaluations = len([r for r in self.evaluation_results if r.get('metric_scores')])
        
        print(f"총 질문 수: {total_questions}")
        print(f"성공한 평가: {successful_evaluations}")
        print(f"실패한 평가: {total_questions - successful_evaluations}")
        
        if successful_evaluations > 0:
            # 각 메트릭별 평균 점수 계산
            metric_names = ['Input Quality', 'Reference Quality', 'Answer Quality', 
                           'Input-Reference Alignment', 'Reference-Answer Alignment', 'Input-Answer Alignment']
            
            print(f"\n📈 G-Eval 메트릭 평균 점수:")
            for metric_name in metric_names:
                scores = [r['metric_scores'][metric_name]['score'] for r in self.evaluation_results 
                         if metric_name in r['metric_scores']]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"  {metric_name}: {avg_score:.3f}")
            
            # 전체 평균 점수
            overall_scores = [r['overall_score'] for r in self.evaluation_results]
            overall_avg = sum(overall_scores) / len(overall_scores)
            print(f"\n🎯 전체 평균 점수: {overall_avg:.3f}")
        
        print("="*60)

# 테스트 실행 코드
if __name__ == "__main__":
    print("🧪 SalesDeepEval 테스트 실행 시작")
    print("="*60)
    
    try:
        # 평가기 초기화 및 실행
        evaluator = DeepEvaluator()
        results = evaluator.run_full_evaluation()
        
        if results:
            evaluator.print_summary()
            evaluator.save_results()
        else:
            print("❌ 평가 결과가 없습니다.")
            
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
