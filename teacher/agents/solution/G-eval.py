# teacher/agents/solution/eval_goldenset_geval.py
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# =============================================================================
# 경로/출력 설정
# =============================================================================
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # -> teacher/
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# 너가 준 골든셋 경로
CSV_FILE_PATH = os.path.join(PROJECT_ROOT, "agents", "solution", "goldensets", "ipa_golden_20250916_152253.jsonl")

# 결과 저장 위치
OUTPUT_DIRECTORY = os.path.join(PROJECT_ROOT, "agents", "solution", "eval_goldenset_geval")
OUTPUT_FORMAT = "json"   # "json" or "csv"
OUTPUT_FILENAME_PREFIX = "solution_geval_results"

# JSON/JSONL 포맷 키 매핑 (문제풀이 에이전트 골든셋)
GOLDEN_JSON_KEYS = {
    'question': 'question',
    'answer': 'ground_truth',   # "정답:..\n풀이:..\n과목:.." 평문
    'reference': 'contexts'     # list[str] 권장
}

# 행 제한/필터 옵션
MAX_EVALUATION_ROWS = None    # None이면 전부
FILTER_BY_THRESHOLD = False

# 메트릭 임계값 (필요시 조정)
METRIC_THRESHOLDS = {
    'input_quality': 0.90,
    'reference_quality': 0.90,
    'answer_quality': 0.90,
    'input_reference_alignment': 0.90,
    'reference_answer_alignment': 0.90,
    'input_answer_alignment': 0.90
}

# 내부 표시명 매핑
METRIC_NAME_MAP = {
    'input_quality': 'Input Quality',
    'reference_quality': 'Reference Quality',
    'answer_quality': 'Answer Quality',
    'input_reference_alignment': 'Input-Reference Alignment',
    'reference_answer_alignment': 'Reference-Answer Alignment',
    'input_answer_alignment': 'Input-Answer Alignment',
}

# 문제풀이 에이전트용 G-Eval 지침(“정답/풀이/과목” 평문을 평가)
EVALUATION_STEPS = {
    'input_quality': [
        "질문이 정보처리기사 시험 스타일(정의/특징/비교/절차/적용 등)에 맞게 구체적이고 모호하지 않은지 평가합니다.",
        "핵심 용어가 명확하며 범위가 과도하게 넓지 않은지 확인합니다.",
        "한국어 문장이 명료한지, 시험형 표현으로 충분히 구체화되어 있는지 평가하세요.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],
    'reference_quality': [
        "컨텍스트가 해당 개념/규칙/절차의 정확한 정의·조건·예외·표기·예시 등을 포함하는지 확인합니다.",
        "용어 사용이 일관되고 상충 진술이 없는지 확인합니다.",
        "시험 풀이에 직접 필요한 핵심 근거가 충분한지 평가하세요.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],
    'answer_quality': [
        "expected_output(정답/풀이/과목 평문)이 컨텍스트를 근거로 간결·정확하게 서술되었는지 평가합니다.",
        "풀이에 불필요한 추측/환각이 없어야 하며, 핵심 근거(정의·조건·절차·키워드)가 명확해야 합니다.",
        "한국어 문장이 명료하고 시험형 답안 톤에 맞는지 확인합니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],
    'input_reference_alignment': [
        "컨텍스트가 질문의 핵심 개념/요구(정의/비교/절차 등)에 직접적으로 대응하는지 평가합니다.",
        "답변에 필요한 근거가 컨텍스트에 포함되어야 합니다.",
        "무관한 정보가 과도하면 감점합니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],
    'reference_answer_alignment': [
        "expected_output(정답/풀이/과목)이 컨텍스트 내부 정보만을 사용했는지(환각 금지) 평가합니다.",
        "컨텍스트 핵심 포인트를 누락하거나 왜곡하지 않았는지 확인합니다.",
        "표기/용어가 컨텍스트와 일치하는지 평가하세요.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],
    'input_answer_alignment': [
        "expected_output이 질문 의도(정의/특징/비교/절차 등)에 직접적으로 응답하는지 평가합니다.",
        "주제 이탈 없이 핵심 키워드 중심으로 작성되었는지 확인하세요.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ]
}

DEEPEVAL_MODEL_NAME = os.getenv("DEEPEVAL_MODEL_NAME", "gpt-4o-mini")

# =============================================================================
# 라이브러리 import
# =============================================================================
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
    def __init__(self, data_path=None):
        print("🔧 문제풀이 골든셋 DeepEval 평가기 초기화...")
        self.data_path = data_path if data_path else CSV_FILE_PATH
        print(f"📁 데이터 파일 경로: {self.data_path}")
        self._setup_models()
        self.test_questions = self._create_test_questions()
        self.evaluation_results = []
        print(f"✅ 초기화 완료 — 테스트 케이스: {len(self.test_questions)}개")

    def _setup_models(self):
        print("📦 DeepEval 모듈 로딩...")
        self.deepeval_modules = import_deepeval_modules()
        print("✅ 모듈 로딩 완료")

        self.model_name = DEEPEVAL_MODEL_NAME
        print(f"🤖 G-Eval 모델: {self.model_name}")

        # 메트릭 구성
        self._setup_custom_metrics()
        print("✅ 커스텀 메트릭 설정 완료")

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

    # ------------------------
    # 데이터 로딩 (JSON/JSONL/CSV)
    # ------------------------
    def _create_test_questions(self):
        path = self.data_path
        if not (path and os.path.exists(path)):
            print(f"⚠️ 파일이 존재하지 않음: {path}")
            return []

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return self._load_csv_data()
        elif ext in (".jsonl", ".ndjson"):
            return self._load_jsonl_data()
        elif ext == ".json":
            items = self._try_load_json_array(path)
            if items is None:
                return self._load_jsonl_data()
            return self._normalize_json_items(items)
        else:
            items = self._try_load_jsonl_fallback(path)
            if items is not None:
                return self._normalize_json_items(items)
            items = self._try_load_json_array(path)
            if items is not None:
                return self._normalize_json_items(items)
            return self._load_csv_data()

    def _load_csv_data(self):
        print(f"📖 CSV 파일 읽기: {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
            print(f"✅ CSV 로드: {len(df)}행")
        except Exception as e:
            print(f"❌ CSV 로드 실패: {e}")
            return []

        # CSV일 경우 컬럼명을 알 수 없으므로 간단 매핑(필요시 커스터마이즈)
        col_map = {'question': 'question', 'answer': 'ground_truth', 'reference': 'contexts'}
        for k, v in col_map.items():
            if v not in df.columns:
                print(f"❌ CSV에 '{v}' 컬럼이 필요합니다.")
                return []

        items = []
        for _, row in df.iterrows():
            items.append({
                'question': str(row[col_map['question']]),
                'ground_truth': str(row[col_map['answer']]),
                'contexts': row[col_map['reference']]
            })
        return self._normalize_json_items(items)

    def _load_jsonl_data(self):
        print(f"📖 JSONL 파일 읽기: {self.data_path}")
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
                return [data]
            if isinstance(data, list):
                return data
            return None
        except Exception:
            return None

    def _try_load_jsonl_fallback(self, path):
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
        """{question, ground_truth, contexts[]} → GEval용 공통 포맷으로 변환"""
        if not items:
            print("⚠️ JSON 항목이 비었습니다.")
            return []

        qk = GOLDEN_JSON_KEYS['question']
        ak = GOLDEN_JSON_KEYS['answer']
        rk = GOLDEN_JSON_KEYS['reference']

        test_cases = []
        max_rows = MAX_EVALUATION_ROWS if MAX_EVALUATION_ROWS else len(items)
        for i, obj in enumerate(items):
            if i >= max_rows: break
            q = str(obj.get(qk, "")).strip()
            ans = str(obj.get(ak, "")).strip()         # "정답/풀이/과목" 전체 평문
            ref = obj.get(rk, "")

            if isinstance(ref, list):
                reference = [str(x).strip() for x in ref if str(x).strip()]
            elif isinstance(ref, str):
                reference = ref.strip()
            else:
                reference = ""

            test_cases.append({'question': q, 'reference': reference, 'answer': ans})
            if i < 3:
                print(f"  📝 질문 {i+1}: {q[:50]}...")
        print(f"✅ JSON/JSONL 로드 완료: {len(test_cases)}개")
        return test_cases

    # ------------------------
    # 평가 실행
    # ------------------------
    def evaluate_single_question(self, test_case):
        question = test_case['question']
        expected_output = test_case.get('answer', '')       # = ground_truth 평문(정답/풀이/과목)
        reference = test_case.get('reference', '')

        print(f"\n📝 질문 평가 시작: {question[:50]}...")

        # retrieval_context는 List[str] 권장
        if isinstance(reference, list):
            retrieval_ctx = reference if reference else [""]
        else:
            retrieval_ctx = [reference] if reference else [""]

        LLMTestCase = self.deepeval_modules['LLMTestCase']
        test_case_obj = LLMTestCase(
            input=question,
            expected_output=expected_output,
            retrieval_context=retrieval_ctx
        )

        print("🔍 G-Eval 메트릭 평가...")
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
                print(f"  📊 {metric.name} ...")
                metric.measure(test_case_obj)
                metric_scores[metric.name] = {
                    'score': metric.score,
                    'reason': metric.reason,
                    'success': metric.success
                }
                print(f"  ✅ {metric.name}: {metric.score:.3f} ({'PASS' if metric.success else 'FAIL'})")
            except Exception as e:
                print(f"  ❌ {metric.name} 실패: {e}")
                metric_scores[metric.name] = {
                    'score': 0.0,
                    'reason': f"평가 실패: {str(e)}",
                    'success': False
                }

        overall = sum([m['score'] for m in metric_scores.values()]) / len(metric_scores)
        result = {
            'question': question,
            'answer_ground_truth': expected_output,
            'reference': reference,
            'metric_scores': metric_scores,
            'overall_score': overall,
            'timestamp': datetime.now().isoformat()
        }
        print(f"✅ 완료 — Overall: {overall:.3f}")
        return result

    def run_full_evaluation(self):
        print("\n🚀 전체 G-Eval 평가 시작")
        if not self.test_questions:
            print("❌ 평가할 항목이 없습니다.")
            return []
        for i, tc in enumerate(self.test_questions, 1):
            print("\n" + "="*60)
            print(f"📝 {i}/{len(self.test_questions)}")
            print("="*60)
            try:
                res = self.evaluate_single_question(tc)
                if res: self.evaluation_results.append(res)
            except Exception as e:
                print(f"❌ 평가 중 오류: {e}")
        print("\n🎉 전체 평가 완료")
        return self.evaluation_results

    def save_results(self, output_path=None):
        if not self.evaluation_results:
            print("❌ 저장할 결과가 없습니다.")
            return None

        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        if not output_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            ext = "json" if OUTPUT_FORMAT.lower() == "json" else "csv"
            output_path = os.path.join(OUTPUT_DIRECTORY, f"{OUTPUT_FILENAME_PREFIX}_{ts}.{ext}")

        # (선택) 임계값 필터
        if FILTER_BY_THRESHOLD:
            name_map = METRIC_NAME_MAP
            filtered = []
            for r in self.evaluation_results:
                ms = r.get('metric_scores', {})
                below = False
                for k, th in METRIC_THRESHOLDS.items():
                    disp = name_map.get(k, k)
                    if disp in ms and ms[disp].get('score', 0) < th:
                        below = True
                        break
                if below:
                    filtered.append(r)
            results_to_save = filtered
            print(f"🔍 임계값 필터: {len(self.evaluation_results)} → {len(results_to_save)}")
        else:
            results_to_save = self.evaluation_results
            print(f"📊 전체 저장: {len(results_to_save)}")

        # 평균들
        name_map = METRIC_NAME_MAP
        metric_avgs = {}
        metric_disp_names = list(name_map.values())
        for name in metric_disp_names:
            scores = [r['metric_scores'][name]['score'] for r in results_to_save if name in r['metric_scores']]
            if scores:
                metric_avgs[name] = sum(scores) / len(scores)

        if OUTPUT_FORMAT.lower() == "json":
            payload = {
                'evaluation_summary': {
                    'total_questions': len(self.evaluation_results),
                    'filtered_questions': len(results_to_save),
                    'filter_applied': FILTER_BY_THRESHOLD,
                    'metric_thresholds': METRIC_THRESHOLDS if FILTER_BY_THRESHOLD else None,
                    'average_scores': metric_avgs,
                    'overall_average': (sum([r['overall_score'] for r in results_to_save]) / len(results_to_save)) if results_to_save else 0,
                    'evaluation_timestamp': datetime.now().isoformat()
                },
                'detailed_results': results_to_save
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            rows = []
            for r in results_to_save:
                ref = r['reference']
                ref_str = " | ".join(ref) if isinstance(ref, list) else str(ref)
                row = {
                    'question': r['question'],
                    'answer_ground_truth': r['answer_ground_truth'],
                    'reference': ref_str,
                    'overall_score': r['overall_score'],
                    'timestamp': r['timestamp']
                }
                for name in metric_disp_names:
                    ms = r['metric_scores'].get(name, {'score': 0.0, 'success': False, 'reason': '평가 실패'})
                    row[f'{name}_score'] = ms['score']
                    row[f'{name}_success'] = ms['success']
                    row[f'{name}_reason'] = ms['reason']
                rows.append(row)
            pd.DataFrame(rows).to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"✅ 저장 완료: {output_path}")
        return output_path

    def print_summary(self):
        if not self.evaluation_results:
            print("❌ 요약할 결과가 없습니다.")
            return
        print("\n" + "="*60)
        print("📊 G-Eval 평가 결과 요약")
        print("="*60)
        total = len(self.evaluation_results)
        print(f"총 질문 수: {total}")
        disp_names = list(METRIC_NAME_MAP.values())
        for name in disp_names:
            scores = [r['metric_scores'][name]['score'] for r in self.evaluation_results if name in r['metric_scores']]
            if scores:
                print(f"  {name}: {sum(scores)/len(scores):.3f}")
        overall = [r['overall_score'] for r in self.evaluation_results]
        if overall:
            print(f"\n🎯 전체 평균 점수: {sum(overall)/len(overall):.3f}")
        print("="*60)

# 실행부
if __name__ == "__main__":
    print("🧪 Solution G-Eval 실행")
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
        print(f"❌ 실행 중 오류: {e}")
        import traceback; traceback.print_exc()
