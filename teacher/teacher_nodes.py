from openai import OpenAI
from typing import List, Dict, Any, Optional
import os 
import json
import re
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
  raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 또는 환경변수에 키를 설정하세요.")

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"), api_key=openai_api_key)


def user_intent(user_question: str) -> dict:
    """
    사용자 질문에서 키워드를 추출합니다.
    """
    system_prompt = f"""다음 사용자 질문을 통해 사용자의 의도를 분석하세요:
    사용자 질문 : {user_question}
    질문의 의도 종류는 다음과 같습니다.
     - generate: 시험 문제 생성 - 새로운 문제를 만들거나 생성하는 것 (예: "문제 만들어줘", "5문제 출제해줘")
     - retrieve: 정보 검색 - 모르는 단어 및 용어에 대한 정보를 검색하는 것
     - analyze: 오답 분석 - 틀린 문제를 정리하고 유형을 분석하여 보완점 및 전략 생성을 추천하는 것
     - solution: 문제 풀이 - 기존 문제에 대한 답과 풀이, 해설을 제공하는 것 (예: "문제 풀어줘", "이거 풀어줘", "이거 해설 해줘", "PDF 풀이해줘")
     - score: 채점 - 문제 풀이에 대한 채점 및 합격 여부를 판단 하는 것
     
    중요한 구분:
    - PDF 파일이나 기존 문제를 풀어달라고 하면 "solution"
    - 새로운 문제를 만들어달라고 하면 "generate"
    - "풀어", "풀이", "해설" 등의 키워드가 있으면 대부분 "solution"
    - 사용자가 본인이 입력한 답에 대해 묻는 경우 "score"
     
    위 5가지 의도 중 하나로만 분류하고, 그 외에 단어나 문장은 포함되지 않게 답변하세요.
    string 형식으로만 출력하세요. 예시:
    "solution"
    """

    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=LLM_TEMPERATURE
    )
    result = response.choices[0].message.content
    
    return result.strip().lower()

def get_user_answer(user_question: str) -> list:
    """
    사용자 질문에서 답변을 추출합니다.
    """
    system_prompt = f"""다음 사용자 질문에서 답변을 추출하세요:
    사용자 질문 : {user_question}
    
    답변은 다음과 같은 형식으로 출력하세요. 모든 문제는 객관식이며, 사용자가 입력한 보기의 숫자만을 리스트(list[str]) 형태로 추출해야 합니다.
    "[4, 3, 1, 2, 4, 3, 3, 2, 1, 1, 4]"
    
    예시:
    사용자 입력: "정답은 4번, 3번, 1번, 2번, 4번, 3번, 3번, 2번, 1번, 1번, 4번입니다."
    추출할 내용: "[4, 3, 1, 2, 4, 3, 3, 2, 1, 1, 4]"
    추출할 수 있는 내용이 없으면 빈 리스트를 반환하세요. 예시: "[]"
    """

    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=LLM_TEMPERATURE
    )
    
    result = response.choices[0].message.content.strip()
    
    # LLM 응답을 파싱하여 리스트로 변환
    try:
        # JSON 형태의 리스트 문자열을 파싱
        import json
        if result.startswith('[') and result.endswith(']'):
            parsed_list = json.loads(result)
            if isinstance(parsed_list, list):
                # ★ 항상 문자열로 정규화해서 반환
                return [str(x).strip() for x in parsed_list]
        
        # JSON 파싱이 실패한 경우, 정규표현식으로 숫자 추출
        import re
        numbers = re.findall(r'\d+', result)
        if numbers:
            # ★ 여기서도 문자열 보장
            return [n.strip() for n in numbers]
        
        # 숫자도 없는 경우 빈 리스트 반환
        return []
        
    except Exception as e:
        print(f"⚠️ 답변 파싱 중 오류 발생: {e}")
        return []

def extract_problem_and_options(user_query: str) -> Dict[str, Any]:
    """
    사용자 질문에서 문제와 보기를 추출합니다.
    
    Args:
        user_query: 사용자 입력 문자열
        
    Returns:
        Dict containing:
        - problem: 문제 문자열
        - options: 보기 리스트
        - has_problem: 문제가 있는지 여부
    """
    print(f"🔍 [extract_problem_and_options] 입력: {user_query}")
    
    if not user_query or not user_query.strip():
        print("⚠️ [extract_problem_and_options] user_query가 비어있습니다.")
        return {
            "problem": "",
            "options": [],
            "has_problem": False
        }
    
    system_prompt = f"""다음 사용자 질문에서 문제와 보기를 추출하세요:
    사용자 질문: {user_query}
    
    다음 형식으로 JSON으로 응답하세요:
    {{
        "problem": "추출된 문제 내용",
        "options": ["보기1", "보기2", "보기3", "보기4"],
        "has_problem": true/false
    }}
    
    추출 규칙:
    1. 문제가 명확하게 보이면 "problem"에 추출
    2. 보기가 명확하게 보이면 "options"에 리스트로 추출
    3. 문제나 보기가 명확하지 않으면 "has_problem": false
    4. 보기는 보통 4개이지만, 3개나 5개일 수도 있음
    5. 문제나 보기가 없으면 빈 문자열이나 빈 리스트로 설정
    
    예시:
    - "소프트웨어 설계에서 사용되는 대표적인 추상화 기법이 아닌 것은? 자료 추상화, 제어 추상화, 과정 추상화, 강도 추상화"
    → {{
        "problem": "소프트웨어 설계에서 사용되는 대표적인 추상화 기법이 아닌 것은?",
        "options": ["자료 추상화", "제어 추상화", "과정 추상화", "강도 추상화"],
        "has_problem": true
    }}
    
    - "안녕하세요"
    → {{
        "problem": "",
        "options": [],
        "has_problem": false
    }}
    
    반드시 유효한 JSON 형식으로만 응답하세요.
    """

    try:
        print(f"🔍 [extract_problem_and_options] LLM 호출 시작...")
        response = client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=LLM_TEMPERATURE
        )
        
        result = response.choices[0].message.content.strip()
        print(f"🔍 [extract_problem_and_options] LLM 응답: {result}")
        
        # JSON 파싱 시도
        import json
        try:
            # 백틱(```)으로 감싸진 JSON 처리
            cleaned_result = result.strip()
            if cleaned_result.startswith('```') and cleaned_result.endswith('```'):
                # 첫 번째와 마지막 백틱 제거
                cleaned_result = cleaned_result[3:-3].strip()
                # 첫 번째 줄이 json이 아닌 경우 제거
                if not cleaned_result.startswith('{'):
                    lines = cleaned_result.split('\n')
                    cleaned_result = '\n'.join([line for line in lines if line.strip() and not line.strip().startswith('```')])
            
            parsed_result = json.loads(cleaned_result)
            print(f"🔍 [extract_problem_and_options] 파싱 결과: {parsed_result}")
            
            if isinstance(parsed_result, dict):
                final_result = {
                    "problem": str(parsed_result.get("problem", "")).strip(),
                    "options": parsed_result.get("options", []) if isinstance(parsed_result.get("options"), list) else [],
                    "has_problem": bool(parsed_result.get("has_problem", False))
                }
                
                # options가 문자열 리스트인지 확인하고 정규화
                if final_result["options"]:
                    normalized_options = []
                    for opt in final_result["options"]:
                        if isinstance(opt, str) and opt.strip():
                            normalized_options.append(opt.strip())
                    final_result["options"] = normalized_options
                
                print(f"🔍 [extract_problem_and_options] 최종 결과: {final_result}")
                return final_result
            else:
                print("⚠️ [extract_problem_and_options] 파싱된 결과가 딕셔너리가 아닙니다.")
                return {
                    "problem": "",
                    "options": [],
                    "has_problem": False
                }
                
        except json.JSONDecodeError as json_error:
            print(f"❌ [extract_problem_and_options] JSON 파싱 오류: {json_error}")
            print(f"🔍 파싱 실패한 응답: {result}")
            
            # JSON 파싱 실패 시 간단한 추출 시도
            if "?" in user_query and any(word in user_query for word in ["보기", "선택", "1", "2", "3", "4"]):
                # 문제와 보기가 있는 것 같음
                return {
                    "problem": user_query.split("?")[0] + "?",
                    "options": user_query.split("?")[1].strip().split(", ") if "?" in user_query else [],
                    "has_problem": True
                }
            else:
                return {
                    "problem": "",
                    "options": [],
                    "has_problem": False
                }
        
    except Exception as e:
        print(f"❌ [extract_problem_and_options] 문제/보기 추출 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {
            "problem": "",
            "options": [],
            "has_problem": False
        }

def parse_generator_input(user_question: str) -> dict:
    """
    사용자 질문에서 과목/문항수/난이도 파싱
    """
    system_prompt = f"""다음 사용자 질문에서 과목/문항수/난이도를 파싱하세요:
    지정된 형식의 응답 외에 문장이나 단어, 특수문자는 작성하지 마세요.
    사용자 질문 : {user_question}
    
    지원하는 모드:
    1. subject_quiz: 단일 과목 문제 생성
    2. partial_exam: 선택된 과목들에 대해 지정된 문제 수만큼 생성
    3. full_exam: 5과목 전체 문제 생성 (각 20문제씩)
    
    과목은 5가지 중 선택: (과목은 무조건 아래 5가지 중 선택을 해야 합니다. 또한, '과목 당', '전과목'을 요청할 경우 모든 과목을 포함해야 합니다.)
    - 소프트웨어설계, 소프트웨어개발, 데이터베이스구축, 프로그래밍언어활용, 정보시스템구축관리
    
    문항 수는 과목별 최대 40문제까지 가능
    난이도는 초급, 중급, 고급 중 하나, 언급 없으면 중급으로 간주
    과목 선정 없이 문제 수 만으로 문제 생성 요청 시 전체 과목 생성. 이 때, 과목 당 문제 수는 전체 문제 수의 1/5로 간주
    다른 변수 없이 단순한 문제 생성 요청은 전체 시험 문제 생성으로 간주 
    파싱 예시:
    1. 과목 당 3문제 만들어줘 -> 과목 당 3문제 생성
        {{
            "mode": "partial_exam",
            "selected_subjects": ["소프트웨어설계", "소프트웨어개발", "데이터베이스구축", "프로그래밍언어활용", "정보시스템구축관리"],
            "questions_per_subject": 3,
            "difficulty": "중급"
        }}
    
    2. 데이터베이스 10문제 만들어줘 -> 데이터베이스 과목 10문제 생성
        {{
            "mode": "subject_quiz",
            "subject_area": "데이터베이스구축",
            "target_count": 10,
            "difficulty": "중급"
        }}
    
    파싱 결과는 다음과 같은 형식으로 출력하세요:
    
    단일 과목의 경우:
    {{
        "mode": "subject_quiz",
        "subject_area": "과목명",
        "target_count": "문항 수",
        "difficulty": "난이도"
    }}
    
    여러 과목 선택의 경우:
    {{
        "mode": "partial_exam",
        "selected_subjects": ["과목1", "과목2", "과목3"],
        "questions_per_subject": "과목당 문항 수",
        "difficulty": "난이도"
    }}
    
    전체 과목의 경우:
    {{
        "mode": "full_exam",
        "difficulty": "난이도"
    }}
    """

    try:
        response = client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=LLM_TEMPERATURE
        )
        
        result = response.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        try:
            import json
            parsed = json.loads(result)
            return parsed
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 기본값 반환
            print(f"⚠️ JSON 파싱 실패: {result}")
            return {
                "mode": "full_exam",
                "difficulty": "중급"
            }
            
    except Exception as e:
        print(f"⚠️ parse_generator_input 오류: {e}")
        # 오류 시 기본값 반환
        return {
            "mode": "full_exam",
            "difficulty": "중급"
        }

# teacher_nodes.py
# 노드 내부에서 사용되는 헬퍼 함수들과 라우팅 로직

# ========== 라우팅 함수들 ==========
def route_solution(state: Dict[str, Any]) -> Dict[str, Any]:
    """solution 노드 라우팅 - 항상 preprocess를 먼저 거침"""
    from teacher_util import has_questions, extract_image_paths
    from pdf_preprocessor import extract_pdf_paths
    
    print(f"🔍 [route_solution] 상태 확인:")
    print(f"   user_query: {state.get('user_query', '')}")
    print(f"   artifacts: {state.get('artifacts', {})}")
    print(f"   has_questions: {has_questions(state)}")
    
    # 사용자 입력에서 파일 탐색
    user_query = state.get("user_query", "")
    current_artifacts = state.get("artifacts", {}) or {}
    
    # 이미지 파일 경로 추출
    extracted_images = extract_image_paths(user_query)
    if extracted_images:
        image_filenames = []
        for path in extracted_images:
            filename = path.split('\\')[-1].split('/')[-1]  # Windows/Unix 경로 모두 처리
            image_filenames.append(filename)
        
        current_artifacts["image_ids"] = image_filenames
        print(f"🖼️ [route_solution] 이미지 파일 발견: {image_filenames}")
    
    # PDF 파일 경로 추출
    extracted_pdfs = extract_pdf_paths(user_query)
    if extracted_pdfs:
        pdf_filenames = []
        for path in extracted_pdfs:
            filename = path.split('\\')[-1].split('/')[-1]  # Windows/Unix 경로 모두 처리
            pdf_filenames.append(filename)
        
        current_artifacts["pdf_ids"] = pdf_filenames
        print(f"📄 [route_solution] PDF 파일 발견: {pdf_filenames}")
    
    # 항상 preprocess를 먼저 거치도록 설정
    next_node = "preprocess"
    print("🔄 항상 preprocess를 먼저 거쳐서 문제를 확인합니다")
    
    print(f"🔍 [route_solution] 다음 노드: {next_node}")
    
    # artifacts 업데이트된 상태 반환
    new_state = {**state}
    new_state["artifacts"] = current_artifacts
    new_state.setdefault("routing", {})
    new_state["routing"]["solution_next"] = next_node
    
    print(f"🔍 [route_solution] 업데이트된 artifacts: {new_state['artifacts']}")
    print(f"🔍 [route_solution] 업데이트된 routing: {new_state['routing']}")
    return new_state

def route_score(state: Dict[str, Any]) -> Dict[str, Any]:
    """score 노드 라우팅"""
    from teacher_util import has_solution_answers
    
    next_node = "score" if has_solution_answers(state) else "mark_after_solution_score"
    new_state = {**state}
    new_state.setdefault("routing", {})
    new_state["routing"]["score_next"] = next_node
    return new_state

def route_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """analysis 노드 라우팅"""
    from teacher_util import has_score
    
    next_node = "analysis" if has_score(state) else "mark_after_score_analysis"
    new_state = {**state}
    new_state.setdefault("routing", {})
    new_state["routing"]["analysis_next"] = next_node
    return new_state

# ========== 마킹 함수들 ==========
def mark_after_generator_solution(state: Dict[str, Any]) -> Dict[str, Any]:
    """generator 후 solution 실행을 위한 마킹"""
    ns = {**state}
    ns.setdefault("routing", {})
    ns["routing"]["after_generator"] = "solution"
    return ns

def mark_after_solution_score(state: Dict[str, Any]) -> Dict[str, Any]:
    """solution 후 score 실행을 위한 마킹"""
    ns = {**state}
    ns.setdefault("routing", {})
    ns["routing"]["after_solution"] = "score"
    return ns

def mark_after_score_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """score 후 analysis 실행을 위한 마킹"""
    ns = {**state}
    ns.setdefault("routing", {})
    ns["routing"]["after_score"] = "analysis"
    return ns

# ========== 포스트 라우팅 함수들 ==========
def post_generator_route(state: Dict[str, Any]) -> str:
    """generator 실행 후 다음 노드 결정"""
    nxt = ((state.get("routing") or {}).get("after_generator") or "").strip()
    return nxt if nxt else "await_output_mode"  # 기본: PDF vs Form 선택 대기 노드로 이동

def post_solution_route(state: Dict[str, Any]) -> str:
    """solution 실행 후 다음 노드 결정"""
    nxt = ((state.get("routing") or {}).get("after_solution") or "").strip()
    return nxt if nxt else "generate_answer_pdf"  # 기본적으로 답안집 PDF 생성

def post_score_route(state: Dict[str, Any]) -> str:
    """score 실행 후 다음 노드 결정"""
    nxt = ((state.get("routing") or {}).get("after_score") or "").strip()
    return nxt if nxt else "analysis"  # 기본적으로 분석 진행

def post_analysis_route(state: Dict[str, Any]) -> str:
    """analysis 실행 후 다음 노드 결정"""
    nxt = ((state.get("routing") or {}).get("after_analysis") or "").strip()
    return nxt if nxt else "generate_analysis_pdf"  # 기본적으로 분석 리포트 PDF 생성

def generate_user_response(state: Dict[str, Any]) -> str:
    """
    사용자에게 실행 결과를 요약해서 답변하는 함수
    """
    system_prompt = f"""당신은 사용자 친화적인 챗봇입니다. 
    사용자의 질문과 실행 결과를 바탕으로 친근하고 이해하기 쉽게 답변해주세요.
    
    답변 형식:
    1. 사용자 질문에 대한 간단한 인사
    2. 실행된 작업들의 요약 (간결하게)
    3. 주요 결과 요약
    4. 추가 도움이 필요한 부분이 있다면 안내
    
    답변은 한국어로 작성하고, 친근하고 도움이 되는 톤으로 작성해주세요.
    """
    
    user_query = state.get("user_query", "")
    intent = state.get("intent", "")
    shared = state.get("shared", {})
    generation = state.get("generation", {})
    solution = state.get("solution", {})
    score = state.get("score", {})
    analysis = state.get("analysis", {})
    retrieval = state.get("retrieval", {})
    artifacts = state.get("artifacts", {})
    
    # 실행된 작업들 파악
    executed_tasks = []
    results_summary = []
    
    if intent == "retrieve" and retrieval:
        executed_tasks.append("정보 검색")
        if shared.get("retrieve_answer"):
            results_summary.append("관련 정보를 검색했습니다")
    
    if intent == "generate" and generation:
        executed_tasks.append("문제 생성")
        question_count = len(shared.get("question", []))
        if question_count > 0:
            results_summary.append(f"{question_count}개의 문제를 생성했습니다")
    
    if intent == "solution" or "solution" in executed_tasks:
        executed_tasks.append("문제 풀이")
        answer_count = len(shared.get("answer", []))
        if answer_count > 0:
            results_summary.append(f"{answer_count}개 문제의 답안과 해설을 생성했습니다")
    
    if intent == "score" or "score" in executed_tasks:
        executed_tasks.append("채점")
        correct_count = shared.get("correct_count", 0)
        total_count = shared.get("total_count", 0)
        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            results_summary.append(f"채점 결과: {correct_count}/{total_count} 정답 ({accuracy:.1f}%)")
    
    if intent == "analyze" or "analysis" in executed_tasks:
        executed_tasks.append("오답 분석")
        weak_types = shared.get("weak_type", [])
        if weak_types:
            results_summary.append(f"취약점 분석 완료: {', '.join(map(str, weak_types[:3]))}{'...' if len(weak_types) > 3 else ''}")
    
    # PDF 생성 확인
    generated_pdfs = artifacts.get("generated_pdfs", [])
    if generated_pdfs:
        executed_tasks.append("PDF 생성")
        pdf_count = len(generated_pdfs)
        results_summary.append(f"{pdf_count}개의 PDF 파일을 생성했습니다")
    
    # 사용자 친화적인 답변 생성
    user_prompt = f"""사용자 질문: {user_query}
    
실행된 작업들: {', '.join(executed_tasks) if executed_tasks else '없음'}
주요 결과: {'; '.join(results_summary) if results_summary else '결과 없음'}

위 정보를 바탕으로 사용자에게 친근하고 도움이 되는 답변을 해주세요."""

    try:
        response = client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=LLM_TEMPERATURE
        )
        
        result = response.choices[0].message.content.strip()
        return result if result else "작업이 완료되었습니다. 추가로 도움이 필요한 부분이 있으시면 말씀해 주세요."
        
    except Exception as e:
        print(f"답변 생성 중 오류: {e}")
        # 기본 답변 반환
        if executed_tasks:
            return f"안녕하세요! {', '.join(executed_tasks)} 작업을 완료했습니다. {'; '.join(results_summary)}"
        else:
            return "안녕하세요! 요청하신 작업을 처리했습니다. 추가로 도움이 필요한 부분이 있으시면 말씀해 주세요."