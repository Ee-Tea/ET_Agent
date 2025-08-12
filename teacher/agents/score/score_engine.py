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