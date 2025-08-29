from __future__ import annotations

import sys
import os
from typing import Dict, Any, List


# 테스트는 프로젝트 루트에서 실행한다고 가정
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from teacher_graph import Orchestrator


class StubGenerator:
    """generator_agent 대체 스텁: 하드코딩된 3문제 반환"""

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # 사용자가 제공한 3문제/보기/정답/풀이/과목을 그대로 반환
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
                    "소프트웨어 공학에서 모델링은 시스템 개발의 여러 단계에서 사용되며, 특히 요구사항 분석, 설계, 구현, 테스트,\n"
                    "유지보수 단계에서 모두 활용됩니다. 따라서 \"유지보수 단계에서만 모델링 기법을 활용한다.\"라는 설명은 틀린\n"
                    "것입니다. 모델링은 개발팀이 응용 문제를 이해하고, 다양한 엔지니어들이 공동의 내용을 공유하는 데 도움을 주며, 또한\n"
                    "절차적인 프로그램을 위한 자료 흐름도는 프로세스 위주의 모델링 방법으로 분류됩니다. 이러한 이유로 2번 선택지가\n"
                    "틀린 설명으로 판단됩니다."
                ),
                "subject": "소프트웨어설계",
            },
            {
                "question": (
                    "UML 모델에서 한 객체가 다른 객체에게 오퍼레이션을 수행하도록 지정하는 의미적 관계로 옳은 것은?"
                ),
                "options": [
                    "Dependency",
                    "Realization",
                    "Generalization",
                    "Association",
                ],
                "answer": "1",
                "explanation": (
                    "UML에서 'Dependency'는 한 객체가 다른 객체의 오퍼레이션에 의존함을 의미합니다.\n"
                    "Realization은 인터페이스-구현, Generalization은 상속, Association은 단순 연결 관계입니다."
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
                    "마스터-슬레이브 구조에서 슬레이브는 데이터 유지 및 처리 기능을 수행할 수 있으므로,\n"
                    "\"슬레이브 프로세스는 데이터 유지 기능을 수행할 수 없다.\"는 설명은 틀렸습니다."
                ),
                "subject": "소프트웨어설계",
            },
        ]

        return {
            "success": True,
            "result": {
                # generator 노드가 지원하는 구조 중 하나(questions)를 사용
                "questions": questions
            },
        }


def main() -> None:
    # Orchestrator 생성(에이전트 초기화는 켜두되, generator는 스텁으로 교체)
    orch = Orchestrator(user_id="u1", service="svc", chat_id="c1", init_agents=True)
    orch.generator_runner = StubGenerator()

    # 스레드 식별자
    config = {"configurable": {"thread_id": "test_form_flow"}}

    # 초기 상태: intent 분류가 LLM을 쓰므로, 간단한 generate 의도 문장을 사용
    init_state: Dict[str, Any] = {
        "user_query": "문제 3개 만들어줘 (테스트)",
    }

    # 1) 최초 호출 → await_output_mode에서 interrupt 발생 기대
    print("\n[STEP 1] invoke() → 출력 방식 선택 interrupt 발생 기대")
    try:
        _ = orch.invoke(init_state, config)
        print("❗ 예상 외: interrupt 없이 종료됨")
    except Exception as e:
        msg = str(e)
        print(f"caught: {type(e).__name__}: {msg}")
        assert "interrupt" in msg.lower(), "첫 번째 interrupt가 발생해야 합니다."

    # 2) 출력 방식 선택: form 으로 재개 → await_form_answers에서 두 번째 interrupt 발생 기대
    print("\n[STEP 2] resume_workflow('form') → 폼 정답 입력 interrupt 발생 기대")
    try:
        _ = orch.resume_workflow("form", config)
        print("❗ 예상 외: 두 번째 interrupt 없이 종료됨")
    except Exception as e:
        msg = str(e)
        print(f"caught: {type(e).__name__}: {msg}")
        assert "interrupt" in msg.lower(), "두 번째 interrupt가 발생해야 합니다."

    # 3) 폼 정답 입력: 3문제 중 2개 정답으로 설정(정확히 80%는 3문제로 불가능)
    #    정답: ["2","1","3"], 사용자 입력: ["2","1","1"] → 2/3 정답 (≈66.7%)
    user_answers = ["2", "1", "1"]
    print("\n[STEP 3] resume_workflow({user_answer}) → 채점→분석→PDF 생성까지 진행")
    final_state = orch.resume_workflow({"user_answer": user_answers}, config)

    # 4) 결과 확인
    shared = final_state.get("shared", {})
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


