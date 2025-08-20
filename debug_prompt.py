#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프롬프트 괄호 문제 디버깅 스크립트
"""

def check_brackets_detailed(text):
    """괄호 균형을 자세히 검사하고 문제 위치를 찾습니다"""
    brackets = {'(': ')', '[': ']', '{': '}', '（': '）', '【': '】', '「': '」'}
    stack = []
    problems = []
    
    for i, char in enumerate(text):
        if char in brackets:
            stack.append((char, i))
        elif char in brackets.values():
            if not stack:
                problems.append(f"위치 {i}: 닫는 괄호 '{char}'가 열리는 괄호 없이 나타남")
            else:
                open_bracket, open_pos = stack.pop()
                if brackets[open_bracket] != char:
                    problems.append(f"위치 {open_pos}-{i}: 괄호 불일치 '{open_bracket}' vs '{char}'")
    
    # 남은 열린 괄호들
    for open_bracket, open_pos in stack:
        problems.append(f"위치 {open_pos}: 열린 괄호 '{open_bracket}'가 닫히지 않음")
    
    return problems

def debug_prompt():
    """프롬프트 내용을 확인하고 괄호 문제를 찾습니다"""
    
    # System prompt
    sys_prompt = (
        "너는 한국어 객관식 시험지 파서다. "
        "너의 유일한 출력은 JSON 배열이며, 각 원소는 { \"question\": string, \"options\": [string, string, string, string] } 형식이다. "
        "다음 규칙을 반드시 지켜라.\n\n"
        "역할/목표:\n"
        "입력은 페이지별 좌/우 컬럼 텍스트가 섞이거나 순서가 틀어질 수 있다. "
        "머리말/꼬리말/저작권/정답·해설 등 비문항 텍스트를 철저히 배제하고, 문항과 보기만 추출한다. "
        "오직 JSON만 출력한다. 설명, 코드블록(예: ```), 주석, 추가 문장은 금지한다. JSON이 아니면 실패로 간주된다.\n\n"
        "정규화(전처리):\n"
        "연속 공백·개행은 하나의 공백으로 정규화하되, 문항/보기 경계 판단에는 원래의 줄 시작 패턴을 사용한다. "
        "원형 번호 ①②③④⑤⑥⑦⑧⑨⑩와 숫자 표기 1) 2) 3) 4), (1) (2) (3) (4), 1. 2. 등은 동일 의미의 표지로 인식한다.\n\n"
        "문항(Question) 구분 알고리즘:\n"
        "A. 문항 시작 패턴: 줄 시작이 다음 중 하나이면 새 문항 시작으로 간주\n"
        "   - 숫자+점 : 예시: 7. 익스트림 프로그래밍에 대한 ...\n"
        "   - 문제 + 숫자 + 점 : 표기 변형이 있더라도 번호가 선행되면 동일하게 간주\n"
        "B. 문항 본문 수집: 시작 라인부터 다음 보기 시작 또는 다음 문항 시작 직전까지를 본문으로 취한다.\n"
        "C. 보기 시작 패턴(다음 중 하나로 시작하는 줄):\n"
        "   - 원문자 번호: ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨ ⑩\n"
        "   - 숫자 괄호: 1) 2) 3) 4)\n"
        "D. 보기 병합: 한 보기가 여러 줄로 이어지면 다음 보기 시작 전까지를 한 보기로 합친다.\n"
        "E. 문항 종료: 다음 문항 시작을 만나거나 입력 종료 시.\n"
        "F. 컬럼 뒤섞임 보정: 문항 번호가 역순으로 튀거나(예: 7 다음 3), 컬럼 전환 흔적이 보이면 수집된 결과를 문항 번호 오름차순으로 재정렬한다.\n\n"
        "제외 규칙(하드 필터):\n"
        "정답, 해설, 풀이, 해답, 참고, 저작권, 다음 문제, 답안카드, 회-쪽, 목차, 과목/장/절 제목, 머리말/꼬리말. "
        "보기에 정답: 같은 라벨이 섞여 있으면 해당 토큰은 삭제. "
        "보기 텍스트 중 영역 제목/그림 캡션/표 제목으로 보이는 라인은 제외.\n\n"
        "출력 제약/검증:\n"
        "각 문항은 옵션 4개만 유지한다(5개 이상이면 상위 4개, 3개 이하면 해당 문항 폐기). "
        "question과 각 options는 앞뒤 공백 제거, 내부 연속 공백 1칸으로 정규화. "
        "중복 보기는 제거하되 4개가 안 되면 문항 폐기. "
        "내용이 지나치게 짧거나(3자 미만) 의미가 없는 문항/보기는 폐기. "
        "출력은 유효한 하나의 JSON 배열이어야 하며, 만족하는 문항이 없다면 []만 출력한다."
    )
    
    print("=== System Prompt 괄호 검사 ===")
    print(f"길이: {len(sys_prompt)}")
    
    # 400번째 위치 주변 확인
    start = max(0, 400 - 20)
    end = min(len(sys_prompt), 400 + 20)
    print(f"\n400번째 위치 주변 (위치 {start}-{end}):")
    print(repr(sys_prompt[start:end]))
    
    # 괄호 문제 상세 검사
    problems = check_brackets_detailed(sys_prompt)
    if problems:
        print(f"\n발견된 괄호 문제들:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\n괄호 문제 없음")
    
    # User prompt도 확인
    user_prompt = (
        "다음 텍스트에서 문항들을 추출해 JSON 배열로 반환해줘.\n"
        "출력은 오직 문항 별 아래 형식의 단 하나의 JSON 코드블록이어야 한다:\n"
        "마크다운 코드블록(```)과 추가 설명은 절대 출력하지 마세요.\n"
        '{"question":"...", "options":["...","...","...","..."]}\n'
        "\n"
        "추출 규칙(아주 중요):\n"
        "1) 문항 시작(Question Start): 다음 중 하나로 시작하는 줄을 문항의 시작으로 간주\n"
        "   - 숫자+점 : 예시: 3. 익스트림 프로그래밍에 대한 ...\n"
        "   - 문제 + 숫자 + 점 : 표기 변형이 있더라도 번호가 선행되면 동일하게 간주\n"
        "2) 문항 종료(Question End): 다음 문항 시작 직전 또는 텍스트 끝, 문항은 보통 ?로 끝남\n"
        "   - ? 이후 보기가 시작하지 않고 텍스트가 존재할 경우 문제에 포함\n"
        "3) 보기 시작(Options Start): 문항 본문 뒤 연속되는 보기 라인. 다음 접두 중 하나로 시작\n"
        "   - 원문자 번호: ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨ ⑩\n"
        "   - 숫자 괄호: 1) 2) 3) 4)\n"
        "4) 보기 종료(Options End): 다음 보기 시작 전까지 또는 문항 종료까지. "
        "   보기 접두(번호/원문자/괄호 등)는 제거하고 내용만 담아라.\n"
        "5) 제외 대상:\n"
        "   - 정답, 해설, 풀이, 참고, ※, 저작권 안내, 과목/섹션 제목, 페이지 번호(예: 회1 - 2 -) 등 비문항 텍스트\n"
        "6) JSON 제약:\n"
        "   - 각 항목은 {\"question\": str, \"options\": list[str]} 형태\n"
        "   - question/option 문자열 내부의 개행/중복 공백은 하나의 공백으로 정규화\n"
        "   - 보기 개수는 4개로 자연스럽게 추출\n"
        "   - 애매하면 해당 보기/문항은 제외(환각 금지). 확실한 것만 넣기\n"
        "\n"
        "참고 예시(형태만 참고, 이 예시는 출력에 포함하지 말 것):\n"
        "원문:\n"
        "2. 애자일 방법론에 해당하지 않는 것은?\n"
        "① 기능 중심 개발\n"
        "② 개발 및 검증\n"
        "③ 익스트림 프로그래밍\n"
        "④ 칸반\n"
        "→ JSON 항목:\n"
        '{"question":"애자일 방법론에 해당하지 않는 것은?", "options":["기능 중심 개발","개발 및 검증","익스트림 프로그래밍","칸반"]}\n'
        "아래 텍스트에서 위 규칙에 맞는 문항만 추출해.\n"
        "텍스트 시작:\n테스트텍스트"
    )
    
    print("\n=== User Prompt 괄호 검사 ===")
    print(f"길이: {len(user_prompt)}")
    
    # 180번째 위치 주변 확인
    start = max(0, 180 - 20)
    end = min(len(user_prompt), 180 + 20)
    print(f"\n180번째 위치 주변 (위치 {start}-{end}):")
    print(repr(user_prompt[start:end]))
    
    # 괄호 문제 상세 검사
    problems = check_brackets_detailed(user_prompt)
    if problems:
        print(f"\n발견된 괄호 문제들:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\n괄호 문제 없음")

if __name__ == "__main__":
    debug_prompt()
