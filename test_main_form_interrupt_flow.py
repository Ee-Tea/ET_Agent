from __future__ import annotations

import os
import sys
from typing import Dict, Any, List


# 프로젝트 루트 기준 실행 가정
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(CUR_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from main import MainOrchestrator


class StubGenerator:
    """Teacher.generator_runner 대체 스텁: 하드코딩된 3문제 반환"""

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        questions: List[Dict[str, Any]] = [
            {
                "question": "소프트웨어 공학에서 모델링(Modeling)과 관련한 설명으로 틀린 것은?",
                "options": [
                    "개발팀이 응용 문제를 이해하는 데 도움을 줄 수 있다.",
                    "유지보수 단계에서만 모델링 기법을 활용한다.",
                    "개발된 시스템에 대하여 여러 분야의 엔지니어들이 공동된 내용을 공유하는 데 도움을 준다.",
                    "절차적인 프로그램을 위한 자료 흐름도는 프로세스 위주의 모델링 방법이다.",
                ],
                "answer": "2",
                "explanation": (
                    "모델링은 요구분석~유지보수 전 단계에서 활용되며, 유지보수 단계에서만 사용하는 것이 아닙니다."
                ),
                "subject": "소프트웨어설계",
            },
            {
                "question": (
                    "UML 모델에서 한 객체가 다른 객체에게 오퍼레이션을 수행하도록 지정하는 의미적 관계로 옳은 것은?"
                ),
                "options": ["Dependency", "Realization", "Generalization", "Association"],
                "answer": "1",
                "explanation": (
                    "'Dependency'는 한 객체가 다른 객체의 오퍼레이션에 의존하는 관계입니다."
                ),
                "subject": "소프트웨어설계",
            },
            {
                "question": (
                    "분산 시스템을 위한 마스터-슬레이브(Master-Slave) 아키텍처에 대한 설명으로 틀린 것은?"
                ),
                "options": [
                    "일반적으로 실시간 시스템에서 사용된다.",
                    "마스터 프로세스는 일반적으로 연산, 통신, 조정을 책임진다.",
                    "슬레이브 프로세스는 데이터 유지 기능을 수행할 수 없다.",
                    "마스터 프로세스는 슬레이브 프로세스들을 제어할 수 있다.",
                ],
                "answer": "3",
                "explanation": (
                    "슬레이브도 데이터 유지/처리를 수행할 수 있으므로 3번은 틀린 설명입니다."
                ),
                "subject": "소프트웨어설계",
            },
        ]

        return {"success": True, "result": {"questions": questions}}


def main() -> None:
    # 오케스트레이터 생성 및 초기화
    orch = MainOrchestrator(user_id="u1", chat_id="c1")

    # 테스트 시작 전 상태 초기화
    try:
        orch.clear_short_term_memory()
    except Exception:
        pass

    # Teacher의 generator를 스텁으로 교체
    orch.teacher.generator_runner = StubGenerator()

    # 스레드/체크포인트 설정 (메인 오케스트레이터 규격: 최상위 키 사용)
    # LangGraph 최신 규격: configurable/thread_id 사용 (메인 오케스트레이터와 동일 식별자)
    config = {
        "configurable": {
            "thread_id": orch.thread_id,
            "checkpoint_id": orch.checkpoint_id,
        }
    }

    # Step 1) orchestrator.run() 사용: 초기 질의 실행 → await_output_mode에서 중단
    print("\n[STEP 1] orchestrator.run() → 출력 방식 선택 interrupt 발생 기대")
    _ = orch.run("문제 3개 만들어줘 (테스트)")

    # Step 2) 출력 방식 선택: form 으로 재개 → 폼 정답 입력 interrupt 발생 기대
    print("\n[STEP 2] resume('form') → 폼 정답 입력 interrupt 발생 기대")
    # 메인 오케스트레이터 래퍼로 재개 (thread/checkpoint 자동 일치)
    _ = orch.resume_workflow("form")

    # Step 3) 폼 정답 입력으로 재개 → 채점/분석/분석PDF 생성까지 진행
    print("\n[STEP 3] resume({user_answer}) → 채점→분석→PDF 생성까지 진행")
    user_answers = ["2", "1", "1"]
    final_state = orch.resume_workflow({"user_answer": user_answers})

    # 결과 확인
    shared = final_state.get("shared", {}) or final_state.get("teacher_state", {}).get("shared", {})
    artifacts = final_state.get("artifacts", {})
    correct = shared.get("correct_count")
    total = shared.get("total_count")
    gen_pdfs = artifacts.get("generated_pdfs", [])

    print("\n[RESULT]")
    print(f"correct/total: {correct}/{total}")
    print(f"generated_pdfs: {gen_pdfs}")

    # 간단한 검증
    assert isinstance(correct, int) and isinstance(total, int), "채점 결과가 있어야 합니다."
    assert total >= 3, "총 문제 수가 3 이상이어야 합니다."
    assert isinstance(gen_pdfs, list) and len(gen_pdfs) >= 1, "분석 PDF가 생성되어야 합니다."
    print("\n✅ 테스트 완료: 2회 interrupt → 채점/분석/분석PDF 생성")


if __name__ == "__main__":
    main()


