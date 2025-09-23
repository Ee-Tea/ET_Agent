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
    os.path.join(PROJECT_ROOT, "agents", "TestGenerator", "goldensets", "generator_golden_20250917_152547.jsonl")
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
    'input_quality': 0.85,
    'reference_quality': 0.85,
    'output_quality': 0.85,
    'input_reference_alignment': 0.85,
    'reference_output_alignment': 0.85,
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
    # 1) 프롬프트 자체의 명확성/구체성
    'input_quality': [
        "생성 지시(input)가 과목/범위(예: '정보시스템구축관리', '전체 범위')와 문항 수(k), 형식 제약(보기 4개, 정답/해설 미포함)을 명확히 제시하는지 평가합니다.",
        "시험 스타일(객관식)과 한국어 표현이 자연스럽고 모호한 표현이 최소화되어 있는지 확인합니다.",
        "프롬프트의 제약을 혼동하게 만드는 불필요한 지시나 중복 지시가 없는지 확인합니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],

    # 2) 컨텍스트(개념+공식문제 2개)의 충분성/일관성/적합성
    'reference_quality': [
        "컨텍스트의 '개념/요약'이 프롬프트의 과목/범위에 직결되는 핵심 개념·정의·절차·예시를 충분히 포함하는지 평가합니다. (예: 과목이 '정보시스템구축관리'인데 '데이터베이스'만 과도하게 포함되면 감점)",
        "컨텍스트 내 용어·표기 일관성(약어, 단위, 기호)과 사실성, 상충 진술/중복/노이즈(오타, 깨짐 텍스트)가 최소화되어 있는지 확인합니다.",
        "컨텍스트의 '공식 문제 (2개)'가 프롬프트 과목/범위와 주제적으로 부합하고, 문항·보기 표기가 명확하며 불필요한 스포일러(정답 암시)가 없는지 평가합니다.",
        "과도한 분량으로 모델 입력을 방해하지 않도록 길이가 적절히 컷(요약/선별)되었는지 확인합니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],

    # 3) expected_output(샘플 문제 세트)의 품질
    'output_quality': [
        "각 문항이 명료하고 한 문항당 보기 4개가 상호 배타·동일 수준으로 구성되었는지, 난이도가 적정한지 평가합니다.",
        "문항 간 중복/유사도가 과도하지 않은지, 기술·용어의 사실성이 유지되는지 확인합니다. (정답/해설은 없어야 함)",
        "수치·절차형 문항은 단위/조건이 충분히 제시되어 재현 가능해야 합니다.",
        "한국어 표현의 자연스러움과 문항 포맷(번호/기호/줄바꿈) 일관성을 확인합니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],

    # 4) 프롬프트 ↔ 컨텍스트 정렬성
    'input_reference_alignment': [
        "컨텍스트의 '개념/요약'과 '공식 문제 2개'가 프롬프트의 과목/범위·문항 형식 요구를 충족하도록 적절히 대비되어 있는지 평가합니다.",
        "프롬프트가 특정 과목을 지시하면 해당 과목에 관한 근거 비중이 충분해야 하며, '전체 범위'일 경우 과도하게 한 과목에 치우치지 않도록 구성되었는지 확인합니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],

    # 5) 컨텍스트 ↔ 출력(골든 샘플) 정렬성
    'reference_output_alignment': [
        "expected_output의 문항들이 컨텍스트(개념/요약 + 공식 문제 2개)의 내용·용어·범위를 실제로 참조 가능한 수준으로 부합하는지 평가합니다. (컨텍스트에 없는 주제/용어만으로 구성되면 감점)",
        "핵심 개념 왜곡, 약어/용어 불일치, 과목/범위 이탈, 환각이 없는지 확인합니다.",
        "보기 표기 방식(기호/번호/단위)과 문항 형식이 컨텍스트의 사용 관례와 충돌하지 않는지 확인합니다.",
        "평가 사유를 한국어로 간결히 설명하세요."
    ],

    # 6) 프롬프트 ↔ 출력 정렬성
    'input_output_alignment': [
        "expected_output이 프롬프트의 제약(문항 수=k, 보기 4개, 정답/해설 미포함, 과목/범위)에 정확히 부합하는지 평가합니다.",
        "‘전체 범위’ 지시에 대해 한 과목 편중 없이 적절한 주제 다양성이 확보되었는지(가능한 경우) 확인합니다.",
        "톤/형식/표기 일관성이 프롬프트 요구를 벗어나지 않는지 확인합니다.",
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
        # 리스트면 첫 원소가 전체 blob인 케이스가 많음 → 그대로 사용
        return [str(x) for x in ref if str(x).strip()]
    elif isinstance(ref, str):
        return [ref]
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
        self.output_quality_metric = GEval(
            name="Generated Set Quality",
            evaluation_steps=EVALUATION_STEPS['output_quality'],
            evaluation_params=[P.EXPECTED_OUTPUT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['output_quality']
        )
        self.input_reference_alignment_metric = GEval(
            name="Input-Reference Alignment",
            evaluation_steps=EVALUATION_STEPS['input_reference_alignment'],
            evaluation_params=[P.INPUT, P.RETRIEVAL_CONTEXT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['input_reference_alignment']
        )
        self.reference_output_alignment_metric = GEval(
            name="Reference-Output Alignment",
            evaluation_steps=EVALUATION_STEPS['reference_output_alignment'],
            evaluation_params=[P.RETRIEVAL_CONTEXT, P.EXPECTED_OUTPUT],
            model=self.model_name,
            threshold=METRIC_THRESHOLDS['reference_output_alignment']
        )
        self.input_output_alignment_metric = GEval(
            name="Input-Output Alignment",
            evaluation_steps=EVALUATION_STEPS['input_output_alignment'],
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
                'reference': retrieval_ctx
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
        else:
            print("❌ 평가 결과가 없습니다.")
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        import traceback; traceback.print_exc()
