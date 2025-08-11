from typing import Dict, Any, List
from agents.base_agent import BaseAgent

# --------- ScoreEngine ---------
class ScoreEngine(BaseAgent):
    @property
    def name(self) -> str:
        """에이전트의 고유한 이름을 반환합니다."""
        return "score"

    @property
    def description(self) -> str:
        """에이전트의 역할에 대한 설명을 반환합니다."""
        return "사용자 답안과 정답을 단순 비교하여 최소 JSON 결과를 반환합니다."

    # --------- 실행 메서드 ---------
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        기대 입력:
          {
            "user_answer": [...],
            "solution_answer": [...]
          }
        반환 (성공):
          {
            "status": "success",
            "total": int,
            "correct": int,
            "incorrect": int,
            "score": float,  # 0~100
            "answer_results": [
                {"index": 0, "user": ..., "solution": ..., "correct": bool},
                ...
            ]
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
                result = {
                    "status": "error",
                    "error": "invalid_type",
                    "message": "'user_answer'와 'solution_answer'는 리스트여야 합니다."
                }
                merged = dict(data)
                merged.update(result)
                return merged

            if len(user_answers) != len(solution_answers):
                result = {
                    "status": "error",
                    "error": "length_mismatch",
                    "message": f"길이 불일치: user={len(user_answers)}, solution={len(solution_answers)}",
                    "expected_total": len(solution_answers),
                    "received_total": len(user_answers)
                }
                merged = dict(data)
                merged.update(result)
                return merged

            results = [u == s for u, s in zip(user_answers, solution_answers)]
            correct = sum(1 for r in results if r)
            total = len(results)
            score = round((correct / total) * 100, 2) if total else 0.0

            answer_results = [
                {
                    "index": idx,
                    "user": u,
                    "solution": s,
                    "correct": r
                }
                for idx, (u, s, r) in enumerate(zip(user_answers, solution_answers, results))
            ]

            result = {
                "status": "success",
                "total": total,
                "correct": correct,
                "incorrect": total - correct,
                "score": score,
                "answer_results": answer_results
            }
            merged = dict(data)  # 입력 원본 포함
            merged.update(result)
            return merged
        except Exception as e:
            result = {
                "status": "error",
                "error": "unexpected",
                "message": str(e)
            }
            merged = dict(data)
            merged.update(result)
            return merged