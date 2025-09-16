# =============================================================================
# 환경 설정
# =============================================================================
PROJECT_ROOT = r"C:\FinalPrj\ET_Agent"

# 이 경로에 CSV / JSON / JSONL(=NDJSON) 아무거나 지정하면 자동 인식합니다.
CSV_FILE_PATH = "./teacher/agents/retrieve/goldensets/goldenset_20250916_124703.jsonl"

OUTPUT_DIRECTORY = "./teacher/agents/retrieve/eval_goldenset_deepeval"
OUTPUT_FORMAT = "json"   # "json" or "csv"
OUTPUT_FILENAME_PREFIX = "retrieve_deepeval_results"

# (CSV일 때만 사용) CSV 컬럼 매핑
GOLDEN_DATASET_COLUMNS = {
    'question': 'user_input',
    'reference': 'reference_contexts',
    'answer': 'reference'
}

# JSON/JSONL일 때 예상 키 (당신의 골든셋 포맷)
GOLDEN_JSON_KEYS = {
    'question': 'question',
    'answer': 'ground_truth',
    'reference': 'contexts'   # list[str] 권장
}

MAX_EVALUATION_ROWS = None
FILTER_BY_THRESHOLD = False

METRIC_THRESHOLDS = {
    'input_quality': 0.9,
    'reference_quality': 0.9,
    'answer_quality': 0.9,
    'input_reference_alignment': 0.9,
    'reference_answer_alignment': 0.9,
    'input_answer_alignment': 0.9
}

# 내부 표시명 매핑(필터 적용 시 키 불일치 보정용)
METRIC_NAME_MAP = {
    'input_quality': 'Input Quality',
    'reference_quality': 'Reference Quality',
    'answer_quality': 'Answer Quality',
    'input_reference_alignment': 'Input-Reference Alignment',
    'reference_answer_alignment': 'Reference-Answer Alignment',
    'input_answer_alignment': 'Input-Answer Alignment',
}

# 그대로 둡니다(원 스크립트 로직 유지) — 필요시만 나중에 교체
EVALUATION_STEPS = {
    'input_quality': [
        "질문이 정보처리기사 범위(데이터베이스, 자료구조/알고리즘, 운영체제, 소프트웨어공학, 컴퓨터구조, 네트워크, 정보보호 등)의 개념/정의/특징/비교/절차와 직접적으로 관련되어 있는지 평가합니다.",
        "핵심 용어가 명확한지 확인합니다(예: 제약조건, 엔터티/관계, 카디널리티, 도메인, 기본키/대체키, 정렬/해싱, 스케줄링, 페이지 교체, OSI/TCP-IP, 암호화 용어 등).",
        "질문 범위가 과도하게 넓거나 모호하면 감점합니다(정의·역할·차이·절차 등 요구 사항이 분명해야 함).",
        "시험형 표현(정의 요청, 비교/장단점/구성요소 나열 등)으로 충분히 구체화되어 있는지 봅니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],
    'reference_quality': [
        "컨텍스트가 해당 개념의 정확한 정의·조건·표기(기호/도식)·규칙·수식·절차·예시 등을 포함하는지 확인합니다.",
        "용어 사용이 일관되고 상충되는 진술이 없어야 합니다(복수 컨텍스트 간 모순 시 감점).",
        "시험과 직접 관련이 없는 주변 정보가 과도하거나 핵심 근거가 부족하면 감점합니다.",
        "핵심 키워드(정의-조건-예외-용도 등)가 드러나는지 평가합니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],
    'answer_quality': [
        "답변이 컨텍스트를 근거로 간결하고 정확하게 서술되었는지 평가합니다(시험 답안 스타일: 핵심어 중심, 불필요한 수식어/장황함 지양).",
        "필수 요소(정의/조건/관계/제약/특징/수식/절차 중 질문에 필요한 요소)가 명확히 포함되어야 합니다.",
        "용어 오용·왜곡·추측·환각이 없어야 하며, 표기(기호/수식) 오류가 없어야 합니다.",
        "한국어 문장이 명료하고 공손한 어조인지 확인합니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],
    'input_reference_alignment': [
        "컨텍스트가 질문한 개념(예: 제약조건, 엔터티, 카디널리티, 도메인, 기본키/대체키 등)과 직접적으로 대응하는지 평가합니다.",
        "동의어/표현 차이는 허용하되, 다른 영역의 개념으로 혼동되면 감점합니다.",
        "질문에 답하기에 필요한 핵심 근거(정의/기호/공식/절차)가 컨텍스트에 포함되어야 합니다.",
        "무관한 주제가 섞여 있으면 감점합니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],
    'reference_answer_alignment': [
        "답변이 컨텍스트의 정보만을 사용했는지(컨텍스트 밖 정보 추가 시 환각으로 간주) 엄격히 확인합니다.",
        "컨텍스트의 핵심 포인트를 누락하거나 왜곡하지 않았는지 평가합니다(복수 컨텍스트가 있을 경우 답변이 인용/요약한 근거와 일치해야 함).",
        "수식·기호·용어 표기가 컨텍스트와 일치하는지 확인합니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],
    'input_answer_alignment': [
        "답변이 질문 의도(정의 설명, 역할/특징 제시, 비교/차이, 절차/단계 등)에 직접적으로 반응하는지 평가합니다.",
        "질문 범위를 벗어나거나 비구체적/회피적 서술이 있으면 감점합니다.",
        "시험형 포맷에 맞게 핵심 키워드를 먼저 제시했는지 확인합니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ]
}


DEEPEVAL_MODEL_NAME = "gpt-4o-mini"

# =============================================================================
# 라이브러리 import
# =============================================================================
import json
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv
env_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(env_path)

def import_deepeval_modules():
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
        print(f"🔧 골든 데이터셋 DeepEval 평가기 초기화 시작...")
        self.data_path = csv_path if csv_path else CSV_FILE_PATH
        print(f"📁 데이터 파일 경로: {self.data_path}")
        try:
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
            import traceback; traceback.print_exc()
            raise

    def _setup_models(self):
        print("📦 DeepEval 모듈들 로딩 중...")
        try:
            self.deepeval_modules = import_deepeval_modules()
            print("✅ DeepEval 모듈들 로딩 완료")
        except Exception as e:
            print(f"❌ DeepEval 모듈 로딩 실패: {e}")
            raise

        try:
            print("🤖 DeepEval 모델 설정 중...")
            self.model_name = DEEPEVAL_MODEL_NAME
            print("✅ DeepEval 모델 설정 완료")

            print("🔧 커스텀 G-Eval 메트릭들 설정 중...")
            self._setup_custom_metrics()
            print("✅ 커스텀 메트릭 설정 완료")
        except Exception as e:
            print(f"❌ 모델 설정 실패: {e}")
            import traceback; traceback.print_exc()
            raise

    def _setup_custom_metrics(self):
        P = self.deepeval_modules['LLMTestCaseParams']
        GEval = self.deepeval_modules['GEval']

        self.input_quality_metric = GEval(
            name="Input Quality",
            evaluation_steps=EVALUATION_STEPS['input_quality'],
            evaluation_params=[P.INPUT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['input_quality']
        )
        self.reference_quality_metric = GEval(
            name="Reference Quality",
            evaluation_steps=EVALUATION_STEPS['reference_quality'],
            evaluation_params=[P.RETRIEVAL_CONTEXT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['reference_quality']
        )
        self.answer_quality_metric = GEval(
            name="Answer Quality",
            evaluation_steps=EVALUATION_STEPS['answer_quality'],
            evaluation_params=[P.EXPECTED_OUTPUT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['answer_quality']
        )
        self.input_reference_alignment_metric = GEval(
            name="Input-Reference Alignment",
            evaluation_steps=EVALUATION_STEPS['input_reference_alignment'],
            evaluation_params=[P.INPUT, P.RETRIEVAL_CONTEXT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['input_reference_alignment']
        )
        self.reference_answer_alignment_metric = GEval(
            name="Reference-Answer Alignment",
            evaluation_steps=EVALUATION_STEPS['reference_answer_alignment'],
            evaluation_params=[P.RETRIEVAL_CONTEXT, P.EXPECTED_OUTPUT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['reference_answer_alignment']
        )
        self.input_answer_alignment_metric = GEval(
            name="Input-Answer Alignment",
            evaluation_steps=EVALUATION_STEPS['input_answer_alignment'],
            evaluation_params=[P.INPUT, P.EXPECTED_OUTPUT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['input_answer_alignment']
        )
        print("✅ 6개 커스텀 G-Eval 메트릭 설정 완료")

    # ------------------------
    # 데이터 로딩 (CSV/JSON/JSONL)
    # ------------------------
    def _create_test_questions(self):
        print(f"📊 테스트 질문 생성 중...")
        print(f"📁 파일 경로 확인: {self.data_path}")
        if not (self.data_path and os.path.exists(self.data_path)):
            print(f"⚠️ 파일이 존재하지 않음: {self.data_path}")
            return []

        ext = os.path.splitext(self.data_path)[1].lower()
        if ext == ".csv":
            return self._load_csv_data()
        elif ext in (".jsonl", ".ndjson"):
            return self._load_jsonl_data()
        elif ext == ".json":
            # 배열 JSON 혹은 (실수로) JSONL 형식일 수도 있어 둘 다 시도
            items = self._try_load_json_array(self.data_path)
            if items is None:
                return self._load_jsonl_data()
            return self._normalize_json_items(items)
        else:
            # 확장자 모호 시: JSONL 시도 → JSON 배열 시도 → CSV 시도
            items = self._try_load_jsonl_fallback(self.data_path)
            if items is not None:
                return self._normalize_json_items(items)
            items = self._try_load_json_array(self.data_path)
            if items is not None:
                return self._normalize_json_items(items)
            return self._load_csv_data()

    def _load_csv_data(self):
        print(f"📖 CSV 파일 읽기 시작: {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
            print(f"✅ CSV 파일 읽기 완료: {len(df)}행")

            required_columns = list(GOLDEN_DATASET_COLUMNS.values())
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                print(f"❌ 필수 컬럼 없음: {missing}")
                print(f"💡 사용 가능 컬럼: {list(df.columns)}")
                print(f"💡 설정된 컬럼명: {GOLDEN_DATASET_COLUMNS}")
                raise ValueError(f"필수 컬럼이 없습니다: {missing}")

            test_cases = []
            max_rows = MAX_EVALUATION_ROWS if MAX_EVALUATION_ROWS else len(df)
            for i, (_, row) in enumerate(df.iterrows()):
                if i >= max_rows: break
                q = str(row[GOLDEN_DATASET_COLUMNS['question']]).strip()
                ref = row[GOLDEN_DATASET_COLUMNS['reference']]
                ans = str(row[GOLDEN_DATASET_COLUMNS['answer']]).strip()

                # CSV의 reference가 문자열이라면 그대로, 리스트 형태라면 변환 시도
                reference = ref
                if isinstance(ref, str):
                    reference = ref.strip()
                elif isinstance(ref, (list, tuple)):
                    reference = [str(x).strip() for x in ref if str(x).strip()]

                test_cases.append({'question': q, 'reference': reference, 'answer': ans})
                if i < 3: print(f"  📝 질문 {i+1}: {q[:50]}...")

            print(f"✅ CSV 로드 완료: {len(test_cases)}개")
            return test_cases
        except Exception as e:
            print(f"❌ CSV 파일 로드 실패: {e}")
            import traceback; traceback.print_exc()
            return []

    def _load_jsonl_data(self):
        print(f"📖 JSONL 파일 읽기 시작: {self.data_path}")
        items = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    items.append(obj)
                except Exception as e:
                    print(f"⚠️ JSONL 파싱 스킵: {e} :: {line[:80]}...")
        return self._normalize_json_items(items)

    def _try_load_json_array(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                # 단일 객체면 리스트로 감싼다
                return [data]
            if isinstance(data, list):
                return data
            return None
        except Exception:
            return None

    def _try_load_jsonl_fallback(self, path):
        # 확장자 모호 시 JSONL로 가정해서 읽어보기
        try:
            items = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    items.append(json.loads(line))
            return items
        except Exception:
            return None

    def _normalize_json_items(self, items):
        """당신의 포맷 {question, ground_truth, contexts[]}을 내부 공통 포맷으로 변환"""
        if not items:
            print("⚠️ JSON 항목이 비어있습니다.")
            return []

        qk = GOLDEN_JSON_KEYS['question']
        ak = GOLDEN_JSON_KEYS['answer']
        rk = GOLDEN_JSON_KEYS['reference']

        test_cases = []
        max_rows = MAX_EVALUATION_ROWS if MAX_EVALUATION_ROWS else len(items)
        for i, obj in enumerate(items):
            if i >= max_rows: break
            q = str(obj.get(qk, "")).strip()
            ans = str(obj.get(ak, "")).strip()
            ref = obj.get(rk, "")

            # contexts가 list[str]이면 그대로 사용, 문자열이면 단일 컨텍스트로 감싼다
            if isinstance(ref, list):
                reference = [str(x).strip() for x in ref if str(x).strip()]
            elif isinstance(ref, str):
                reference = ref.strip()
            else:
                reference = ""

            test_cases.append({'question': q, 'reference': reference, 'answer': ans})
            if i < 3: print(f"  📝 질문 {i+1}: {q[:50]}...")

        print(f"✅ JSON/JSONL 로드 완료: {len(test_cases)}개")
        return test_cases

    # ------------------------
    # 평가 실행
    # ------------------------
    def evaluate_single_question(self, test_case):
        question = test_case['question']
        answer = test_case.get('answer', '')
        reference = test_case.get('reference', '')

        print(f"\n📝 질문 평가 시작: {question[:50]}...")

        # retrieval_context는 List[str]을 권장
        if isinstance(reference, list):
            retrieval_ctx = reference if reference else [""]
        else:
            retrieval_ctx = [reference] if reference else [""]

        test_case_obj = self.deepeval_modules['LLMTestCase'](
            input=question,
            expected_output=answer,
            retrieval_context=retrieval_ctx
        )

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
        print(f"\n🚀 전체 DeepEval 평가 시작!")
        print(f"📊 평가할 질문 수: {len(self.test_questions)}")
        if not self.test_questions:
            print("❌ 평가할 질문이 없습니다.")
            return []

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
                import traceback; traceback.print_exc()

        print(f"\n🎉 전체 평가 완료!")
        print(f"📊 성공한 평가: {len(self.evaluation_results)}개")
        return self.evaluation_results

    def save_results(self, output_path=None):
        if not self.evaluation_results:
            print("❌ 저장할 평가 결과가 없습니다.")
            return None

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
            ext = "json" if OUTPUT_FORMAT.lower() == "json" else "csv"
            output_path = os.path.join(OUTPUT_DIRECTORY, f"{OUTPUT_FILENAME_PREFIX}_{timestamp}.{ext}")

        # (선택) 임계값 필터 적용 — 내부 이름 매핑으로 보정
        if FILTER_BY_THRESHOLD:
            filtered = []
            for result in self.evaluation_results:
                ms = result.get('metric_scores', {})
                below = False
                for k, th in METRIC_THRESHOLDS.items():
                    display = METRIC_NAME_MAP.get(k, k)
                    if display in ms and ms[display].get('score', 0) < th:
                        below = True
                        break
                if below:
                    filtered.append(result)
            print(f"🔍 임계값 필터: {len(self.evaluation_results)}개 중 {len(filtered)}개 저장")
            results_to_save = filtered
        else:
            results_to_save = self.evaluation_results
            print(f"📊 전체 결과 저장: {len(results_to_save)}개")

        # 평균 점수 계산
        metric_averages = {}
        metric_names = list(METRIC_NAME_MAP.values())
        for name in metric_names:
            scores = [r['metric_scores'][name]['score'] for r in results_to_save if name in r['metric_scores']]
            if scores:
                metric_averages[name] = sum(scores) / len(scores)

        if OUTPUT_FORMAT.lower() == "json":
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
            csv_data = []
            for r in results_to_save:
                ref = r['reference']
                ref_str = " | ".join(ref) if isinstance(ref, list) else str(ref)
                row = {
                    'question': r['question'],
                    'answer': r['answer'],
                    'reference': ref_str,
                    'overall_score': r['overall_score'],
                    'timestamp': r['timestamp']
                }
                for name in metric_names:
                    if name in r['metric_scores']:
                        row[f'{name}_score'] = r['metric_scores'][name]['score']
                        row[f'{name}_success'] = r['metric_scores'][name]['success']
                        row[f'{name}_reason'] = r['metric_scores'][name]['reason']
                    else:
                        row[f'{name}_score'] = 0.0
                        row[f'{name}_success'] = False
                        row[f'{name}_reason'] = "평가 실패"
                csv_data.append(row)
            pd.DataFrame(csv_data).to_csv(output_path, index=False, encoding='utf-8-sig')
        else:
            print(f"❌ 지원하지 않는 출력 형식: {OUTPUT_FORMAT}")
            return None

        print(f"✅ 평가 결과 저장 완료 ({OUTPUT_FORMAT.upper()}): {output_path}")
        return output_path

    def print_summary(self):
        if not self.evaluation_results:
            print("❌ 평가 결과가 없습니다.")
            return

        print("\n" + "="*60)
        print("📊 DeepEval G-Eval 평가 결과 요약")
        print("="*60)

        total = len(self.evaluation_results)
        success = len([r for r in self.evaluation_results if r.get('metric_scores')])

        print(f"총 질문 수: {total}")
        print(f"성공한 평가: {success}")
        print(f"실패한 평가: {total - success}")

        metric_names = list(METRIC_NAME_MAP.values())
        print(f"\n📈 G-Eval 메트릭 평균 점수:")
        for name in metric_names:
            scores = [r['metric_scores'][name]['score'] for r in self.evaluation_results if name in r['metric_scores']]
            if scores:
                print(f"  {name}: {sum(scores)/len(scores):.3f}")

        overall = [r['overall_score'] for r in self.evaluation_results]
        if overall:
            print(f"\n🎯 전체 평균 점수: {sum(overall)/len(overall):.3f}")
        print("="*60)

# 테스트 실행
if __name__ == "__main__":
    print("🧪 RetrieveDeepEval 테스트 실행 시작")
    print("="*60)
    try:
        evaluator = DeepEvaluator()
        results = evaluator.run_full_evaluation()
        if results:
            evaluator.print_summary()
            evaluator.save_results()
        else:
            print("❌ 평가 결과가 없습니다.")
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback; traceback.print_exc()
