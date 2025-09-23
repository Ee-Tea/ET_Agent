# -*- coding: utf-8 -*-

import os
import json
import re
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# =============================================================================
# 경로/출력 설정
# =============================================================================
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # -> teacher/
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ← 생성 에이전트 골든셋 기본 경로 (jsonl 권장)
GOLDEN_FILE_PATH = os.getenv(
    "GENERATOR_GOLDEN_PATH",
    os.path.join(PROJECT_ROOT, "agents", "TestGenerator", "goldensets", "generator_golden_ragas_style_20250922_004331.jsonl")
)

# 결과 저장 위치
OUTPUT_DIRECTORY = os.path.join(PROJECT_ROOT, "agents", "TestGenerator", "eval_goldenset_geval")
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "json")   # "json" or "csv"
OUTPUT_FILENAME_PREFIX = "generator_geval_results"

# JSON/JSONL 포맷 키 매핑 (문제 생성 에이전트 골든셋)
# question: 생성 요청 프롬프트(예: "정보처리기사 소프트웨어설계 객관식 12문제 만들어줘...")
# ground_truth: [{question, options[4]}, ...]
# contexts: str | list[str]
GOLDEN_JSON_KEYS = {
    'question': 'question',
    'golden_samples': 'ground_truth',
    'reference': 'contexts'
}

# 행 제한/필터 옵션
MAX_EVALUATION_ROWS = None    # None이면 전부
FILTER_BY_THRESHOLD = False

# 메트릭 임계값 (필요시 조정; 생성 품질 관점으로 소폭 완화)
METRIC_THRESHOLDS = {
    'input_quality': 0.8,
    'reference_quality': 0.8,
    'output_quality': 0.8,
    'input_reference_alignment': 0.80,  # from 0.85
    'reference_output_alignment': 0.80, # from 0.85
    'input_output_alignment': 0.85
}


# 내부 표시명 매핑
METRIC_NAME_MAP = {
    'input_quality': 'Input Quality',
    'reference_quality': 'Reference Quality',
    'output_quality': 'Generated Set Quality',
    'input_reference_alignment': 'Input-Reference Alignment',
    'reference_output_alignment': 'Reference-Output Alignment',
    'input_output_alignment': 'Input-Output Alignment',
}

# ── 생성 과업 전용 G-Eval 지침 ────────────────────────────────────────────────
EVALUATION_STEPS = {
    # 1) 입력 프롬프트 품질
    'input_quality': [
        "입력 지시문이 한국어로 자연스러우며, 객관식 생성과 형식 제약(보기 4개, 정답/해설 미포함)을 분명히 전달하는지 가볍게 확인합니다.",
        "과목 또는 범위(예: 특정 과목명 또는 '전체 범위')와 문항 수(k)가 대략적으로 파악 가능하면 충분합니다.",
        "사소한 중복표현이나 표현상의 군더더기는 감점하지 않습니다."
    ],

    # 2) 컨텍스트(개념/요약 + 공식 문제 조각들)의 유용성
    'reference_quality': [
        "컨텍스트는 여러 조각으로 제공될 수 있습니다. 각 조각이 전반적으로 프롬프트 범위와 관련되어 있으면 충분합니다.",
        "헤더(예: '### 개념/요약', '### 공식 문제 (컨텍스트)')나 섹션 표시는 정상으로 간주하며, 이로 인해 감점하지 않습니다.",
        "'전체 범위'일 때는 과목 혼입을 어느 정도 허용합니다. 특정 과목 지시 시에도 일부 혼입은 가볍게 넘어갑니다.",
        "소량의 중복·경미한 노이즈(오타/깨짐)가 있어도 큰 감점 없이 넘어갑니다."
         
    ],

    # 3) 생성될 샘플 문제 세트(ground_truth)의 품질
    'output_quality': [
        "각 문항이 명료하고 보기 4개가 제시되어 있으면 충분합니다(완벽한 난이도 균형까지는 요구하지 않습니다).",
        "문항 간 유사성이 일부 있어도 괜찮습니다. 다만 동일 문항 반복은 피하는 것이 좋습니다.",
        "정답/해설이 포함되지 않았는지만 가볍게 확인합니다."
    ],

    # 4) 입력 ↔ 컨텍스트 정렬
    'input_reference_alignment': [
        "컨텍스트가 입력 지시의 과목에 대한 내용으로 대체로 부합하면 충분합니다.",
    ],

    # 5) 컨텍스트 ↔ 출력(샘플) 정렬
    'reference_output_alignment': [
        "샘플 문항들이 컨텍스트에서 다루는 개념/용어/주제와 대체로 호흡이 맞으면 충분합니다.",
        "표현상의 경미한 차이나 용어 변형은 허용합니다."
    ],

    # 6) 입력 ↔ 출력 정렬
    'input_output_alignment': [
        "샘플 문항 수(k) 및 형식 제약(보기 4개, 정답/해설 미포함)이 대략 충족되면 충분합니다.",
        "'전체 범위' 지시에서 특정 과목 편중이 일부 있어도 큰 감점 없이 넘어갑니다."
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

# =============================================================================
# 유틸
# =============================================================================
def _format_ground_truth_samples(samples):
    """
    [{question, options[4]}, ...] → G-Eval이 읽기 쉬운 평문 형태로 직렬화
    """
    lines = []
    for idx, it in enumerate(samples or [], 1):
        q = str(it.get("question", "")).strip()
        opts = [str(o).strip() for o in (it.get("options") or [])][:4]
        if not q or len(opts) < 4:
            # 스킵
            continue
        lines.append(f"[문항 {idx}] {q}")
        for j, o in enumerate(opts, 1):
            lines.append(f"  - ({j}) {o}")
    return "\n".join(lines).strip()

def _stringify_reference(ref):
    if isinstance(ref, list):
        lst = [str(x) for x in ref if str(x).strip()]
        # ▼ 추가
        MAX_PER_REF = 2000
        return [x[:MAX_PER_REF] for x in lst]
    elif isinstance(ref, str):
        return [ref[:2000]]
    return [""]

# =============================================================================
# 평가기
# =============================================================================
class DeepEvaluator:
    def __init__(self, data_path=None):
        print("🔧 생성 에이전트 골든셋 DeepEval 평가기 초기화...")
        self.data_path = data_path if data_path else GOLDEN_FILE_PATH
        print(f"📁 데이터 파일 경로: {self.data_path}")
        self._setup_models()
        self.raw_items = []
        self.test_questions = self._create_test_questions()
        self.evaluation_results = []
        print(f"✅ 초기화 완료 — 테스트 케이스: {len(self.test_questions)}개")

    def _setup_models(self):
        print("📦 DeepEval 모듈 로딩...")
        self.deepeval_modules = import_deepeval_modules()
        print("✅ 모듈 로딩 완료")

        self.model_name = DEEPEVAL_MODEL_NAME
        print(f"🤖 G-Eval 모델: {self.model_name}")

        # 메트릭 구성 (생성 과업 버전)
        self._setup_custom_metrics()
        print("✅ 커스텀 메트릭 설정 완료")

    def _setup_custom_metrics(self):
        P = self.deepeval_modules['LLMTestCaseParams']
        GEval = self.deepeval_modules['GEval']

        self.input_quality_metric = GEval(
            name="Input Quality",
            evaluation_steps=EVALUATION_STEPS['input_quality'] + ["평가 코멘트는 반드시 한국어로 작성하세요."],
            evaluation_params=[P.INPUT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['input_quality']
        )
        self.reference_quality_metric = GEval(
            name="Reference Quality",
            evaluation_steps=EVALUATION_STEPS['reference_quality'] + ["평가 코멘트는 반드시 한국어로 작성하세요."],
            evaluation_params=[P.RETRIEVAL_CONTEXT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['reference_quality']
        )
        self.output_quality_metric = GEval(
            name="Generated Set Quality",
            evaluation_steps=EVALUATION_STEPS['output_quality'] + ["평가 코멘트는 반드시 한국어로 작성하세요."],
            evaluation_params=[P.EXPECTED_OUTPUT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['output_quality']
        )
        self.input_reference_alignment_metric = GEval(
            name="Input-Reference Alignment",
            evaluation_steps=EVALUATION_STEPS['input_reference_alignment'] + ["평가 코멘트는 반드시 한국어로 작성하세요."],
            evaluation_params=[P.INPUT, P.RETRIEVAL_CONTEXT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['input_reference_alignment']
        )
        self.reference_output_alignment_metric = GEval(
            name="Reference-Output Alignment",
            evaluation_steps=EVALUATION_STEPS['reference_output_alignment'] + ["평가 코멘트는 반드시 한국어로 작성하세요."],
            evaluation_params=[P.RETRIEVAL_CONTEXT, P.EXPECTED_OUTPUT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['reference_output_alignment']
        )
        self.input_output_alignment_metric = GEval(
            name="Input-Output Alignment",
            evaluation_steps=EVALUATION_STEPS['input_output_alignment'] + ["평가 코멘트는 반드시 한국어로 작성하세요."],
            evaluation_params=[P.INPUT, P.EXPECTED_OUTPUT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['input_output_alignment']
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

        # CSV일 경우 컬럼명 가정
        col_map = {'question': 'question', 'golden_samples': 'ground_truth', 'reference': 'contexts'}
        for _, v in col_map.items():
            if v not in df.columns:
                print(f"❌ CSV에 '{v}' 컬럼이 필요합니다.")
                return []

        items = []
        for _, row in df.iterrows():
            try:
                samples = row[col_map['golden_samples']]
                if isinstance(samples, str):
                    # CSV에서는 JSON 문자열로 들어올 수 있음
                    samples = json.loads(samples)
            except Exception:
                samples = []
            items.append({
                'question': str(row[col_map['question']]),
                'ground_truth': samples,
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
        """
        {question, ground_truth=[{question, options[4]}...], contexts[]} → GEval용 포맷
        """
        if not items:
            print("⚠️ JSON 항목이 비었습니다.")
            return []

        qk = GOLDEN_JSON_KEYS['question']
        sk = GOLDEN_JSON_KEYS['golden_samples']
        rk = GOLDEN_JSON_KEYS['reference']

        # 정규화된 원본 골든셋을 evaluator가 보관하도록 저장
        normalized_raw = []
        for obj in items:
            normalized_raw.append({
                'question': str(obj.get(qk, "")).strip(),
                'ground_truth': obj.get(sk, []) or [],
                'contexts': obj.get(rk, "")  # str | list[str]
            })
        self.raw_items = normalized_raw

        test_cases = []
        max_rows = MAX_EVALUATION_ROWS if MAX_EVALUATION_ROWS else len(items)
        for i, obj in enumerate(items):
            if i >= max_rows: break
            q = str(obj.get(qk, "")).strip()
            samples = obj.get(sk, []) or []
            ref = obj.get(rk, "")

            # expected_output으로 쓸 평문 세트 구성
            expected_output = _format_ground_truth_samples(samples)
            retrieval_ctx = _stringify_reference(ref)

            test_cases.append({
                'question': q,
                'expected_output': expected_output,
                'reference': retrieval_ctx,
                'raw_index': i,
            })
            if i < 3:
                print(f"  📝 프롬프트 {i+1}: {q[:60]}...")
                print(f"    • 샘플 {len(samples)}문항, 컨텍스트 len={sum(len(x) for x in retrieval_ctx)}")
        print(f"✅ JSON/JSONL 로드 완료: {len(test_cases)}개")
        return test_cases

    # ------------------------
    # 평가 실행
    # ------------------------
    def evaluate_single_question(self, test_case):
        question = test_case['question']
        expected_output = test_case.get('expected_output', '')   # = 샘플 문제 세트 평문
        reference_list = test_case.get('reference', [""])

        print(f"\n📝 프롬프트 평가 시작: {question[:60]}...")

        LLMTestCase = self.deepeval_modules['LLMTestCase']
        tc = LLMTestCase(
            input=question,
            expected_output=expected_output,
            retrieval_context=reference_list
        )

        print("🔍 G-Eval 메트릭 평가...")
        metrics = [
            self.input_quality_metric,
            self.reference_quality_metric,
            self.output_quality_metric,
            self.input_reference_alignment_metric,
            self.reference_output_alignment_metric,
            self.input_output_alignment_metric
        ]

        metric_scores = {}
        for metric in metrics:
            try:
                print(f"  📊 {metric.name} ...")
                metric.measure(tc)
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
            'expected_output_preview': expected_output[:500],
            'reference_used': reference_list[:1],
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
                for key, th in METRIC_THRESHOLDS.items():
                    disp = name_map.get(key, key)
                    if disp in ms and ms[disp].get('score', 0) < th:
                        below = True
                        break
                if not below:
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
                    'total_cases': len(self.evaluation_results),
                    'saved_cases': len(results_to_save),
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
                row = {
                    'question': r['question'],
                    'expected_output_preview': r['expected_output_preview'],
                    'reference_first_256': (r['reference_used'][0][:256] if r['reference_used'] else ""),
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
    
    def save_passed_goldensets(self, output_path=None):
        """
        모든 메트릭이 PASS인 케이스만 선별하여
        원본 골든셋(JSON 스키마)을 JSONL(1줄 1레코드)로 저장.
        """
        if not self.evaluation_results or not self.raw_items:
            print("❌ 저장할 PASS 골든셋이 없습니다.")
            return None

        # test_questions와 evaluation_results를 나란히 순회하여 raw_index 회수
        passed_items = []
        for tc, res in zip(self.test_questions, self.evaluation_results):
            ms = res.get('metric_scores', {})
            # 모든 메트릭 success=True 여야 PASS
            all_pass = True
            for disp_name in METRIC_NAME_MAP.values():
                if not ms.get(disp_name, {}).get('success', False):
                    all_pass = False
                    break
            if all_pass:
                ri = tc.get('raw_index')
                if ri is not None and 0 <= ri < len(self.raw_items):
                    passed_items.append(self.raw_items[ri])

        if not passed_items:
            print("ℹ️ PASS된 골든셋이 없습니다.")
            return None

        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        if not output_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            output_path = os.path.join(
                OUTPUT_DIRECTORY,
                f"generator_geval_passed_goldensets_{ts}.jsonl"
            )

        # ✅ JSONL로 저장 (1줄 1샘플)
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            for item in passed_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✅ PASS 골든셋 저장 완료: {len(passed_items)}개 → {output_path}")
        return output_path



    def print_summary(self):
        if not self.evaluation_results:
            print("❌ 요약할 결과가 없습니다.")
            return
        print("\n" + "="*60)
        print("📊 G-Eval 평가 결과 요약")
        print("="*60)
        total = len(self.evaluation_results)
        print(f"총 케이스 수: {total}")
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
    print("🧪 Generator G-Eval 실행")
    print("="*60)
    try:
        evaluator = DeepEvaluator()
        results = evaluator.run_full_evaluation()
        if results:
            evaluator.print_summary()
            evaluator.save_results()
            # 모든 항목 PASS 골든셋만 별도 JSONL 저장
            evaluator.save_passed_goldensets()
        else:
            print("❌ 평가 결과가 없습니다.")
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        import traceback; traceback.print_exc()

