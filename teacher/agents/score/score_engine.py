from typing import Dict, Any, List, TypedDict, Literal, Union, TypeGuard
from agents.base_agent import BaseAgent

# 상태 TypedDict 정의
class ScoreSuccessResult(TypedDict):
    status: Literal["success"]
    results: List[int]

class ScoreInvalidTypeError(TypedDict):
    status: Literal["error"]
    error: Literal["invalid_type"]
    message: str

class ScoreLengthMismatchError(TypedDict):
    status: Literal["error"]
    error: Literal["length_mismatch"]
    message: str
    expected_total: int
    received_total: int

class ScoreUnexpectedError(TypedDict):
    status: Literal["error"]
    error: Literal["unexpected"]
    message: str

ScoreResult = Union[
    ScoreSuccessResult,
    ScoreInvalidTypeError,
    ScoreLengthMismatchError,
    ScoreUnexpectedError,
]

# TypedDict 생성 헬퍼
def _success(results: List[int]) -> ScoreSuccessResult:
    return {"status": "success", "results": results}

def _invalid_type(message: str) -> ScoreInvalidTypeError:
    return {"status": "error", "error": "invalid_type", "message": message}

def _length_mismatch(expected_total: int, received_total: int) -> ScoreLengthMismatchError:
    return {
        "status": "error",
        "error": "length_mismatch",
        "message": f"길이 불일치: user={received_total}, solution={expected_total}",
        "expected_total": expected_total,
        "received_total": received_total,
    }

def _unexpected(message: str) -> ScoreUnexpectedError:
    return {"status": "error", "error": "unexpected", "message": message}

# 호출 측 타입 내로잉을 위한 가드
def is_success(result: ScoreResult) -> TypeGuard[ScoreSuccessResult]:
    return result.get("status") == "success"

# --------- ScoreEngine ---------
class ScoreEngine(BaseAgent):
    @property
    def name(self) -> str:
        """에이전트의 고유한 이름을 반환합니다."""
        return "score"

    @property
    def description(self) -> str:
        """에이전트의 역할에 대한 설명을 반환합니다."""
        return "사용자 답안과 정답을 단순 비교하여 채점, JSON 결과를 반환합니다."

    # --------- 실행 메서드 ---------
    def execute(self, data: Dict[str, Any]) -> ScoreResult:
        """
        기대 입력:
          {
            "user_answer": [...],
            "solution_answer": [...]
          }
        반환 (성공):
          {
            "status": "success",
            "results": [...]  # [0,1] 이진 리스트
          }
        반환 (오류 예: 길이 불일치):
          {
            "status": "error",
            "error": "length_mismatch",
            "message": "...",
            "expected_total": int,
            "received_total": int
          }
        """
        try:
            user_answers: List[Any] = data.get("user_answer", []) or []
            solution_answers: List[Any] = data.get("solution_answer", []) or []

            if not isinstance(user_answers, list) or not isinstance(solution_answers, list):
                return _invalid_type("'user_answer'와 'solution_answer'는 리스트여야 합니다.")

            if len(user_answers) != len(solution_answers):
                return _length_mismatch(len(solution_answers), len(user_answers))

            bool_results = [u == s for u, s in zip(user_answers, solution_answers)]
            binary_results = [1 if r else 0 for r in bool_results]

            return _success(binary_results)
        except Exception as e:
            return _unexpected(str(e))

# 간단 출력 유틸리티
def print_score_result(result: ScoreResult) -> None:
    """
    ScoreEngine 결과를 요약 출력합니다.
    성공: 총 문항/정답/오답/오답 번호 목록(+ 소량일 때 O/X 행 출력)
    오류: 오류 유형/메시지 출력
    """
    print("\n=== 채점 결과 ===")
    if not is_success(result):
        err = getattr(result, "get", None) and result.get("error")
        msg = result.get("message") if hasattr(result, "get") else None
        print(f"❌ 오류: {err or 'unknown_error'}")
        if msg:
            print(f"이유: {msg}")
        if err == "length_mismatch":
            exp = result.get("expected_total")
            rec = result.get("received_total")
            print(f"기대 길이={exp}, 입력 길이={rec}")
        return

    results = result["results"]
    total = len(results)
    correct = sum(results)
    incorrect = total - correct
    wrong_indices = [i + 1 for i, r in enumerate(results) if r == 0]

    print(f"- 총 문항: {total}")
    print(f"- 정답: {correct}")
    print(f"- 오답: {incorrect}")
    if wrong_indices:
        print(f"- 오답 번호(1-based): {', '.join(map(str, wrong_indices))}")

    # 문항 수가 작을 때만 O/X 행 출력
    if total <= 30:
        print("\n[문항별 결과]")
        for i, r in enumerate(results, start=1):
            mark = "O" if r == 1 else "X"
            print(f"  #{i:02d} {mark}")