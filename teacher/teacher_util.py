# teacher_util.py
# 유틸리티 함수들을 모아놓은 모듈

import os
from typing import Dict, Any, List
from copy import deepcopy

# ========== 의도 정규화 ==========
CANON_INTENTS = {"retrieve", "generate", "analyze", "solution", "score"}

# ========== Shared State 관리 ==========
SHARED_DEFAULTS: Dict[str, Any] = {
    "question": [],
    "options": [],
    "answer": [],
    "explanation": [],
    "subject": [],
    "wrong_question": [],
    "weak_type": [],
    "user_answer": [],
    "retrieve_answer": "",
}
class SupportsExecute:
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...

def ensure_shared(state: Dict[str, Any]) -> Dict[str, Any]:
    """shared 키 및 타입을 보정하여 이후 노드에서 안정적으로 사용 가능하게 합니다."""
    ns = deepcopy(state) if state else {}
    ns.setdefault("shared", {})
    for key, default_val in SHARED_DEFAULTS.items():
        cur = ns["shared"].get(key, None)
        if not isinstance(cur, type(default_val)):
            ns["shared"][key] = deepcopy(default_val)
    return ns

def validate_qas(shared: Dict[str, Any]) -> None:
    """문항/보기/정답/해설/과목 길이 일관성 검증."""
    n = len(shared.get("question", []))
    if not all(len(shared.get(k, [])) == n for k in ("options", "answer", "explanation", "subject")):
        raise ValueError(
            f"[QA 정합성 오류] 길이 불일치: "
            f"q={len(shared.get('question', []))}, "
            f"opt={len(shared.get('options', []))}, "
            f"ans={len(shared.get('answer', []))}, "
            f"exp={len(shared.get('explanation', []))}, "
            f"subj={len(shared.get('subject', []))}"
        )

# ========== 에이전트 실행 ==========
class SupportsExecute:
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...

def safe_execute(agent: SupportsExecute, payload: Dict[str, Any]) -> Dict[str, Any]:
    """에이전트 실행 예외 방지 래퍼."""
    try:
        out = agent.execute(payload)
        return out if isinstance(out, dict) else {}
    except Exception as e:
        print(f"[WARN] agent {getattr(agent, 'name', type(agent).__name__)} failed: {e}")
        return {}

# ========== 의존성 체크 ==========
def has_questions(state: Dict[str, Any]) -> bool:
    sh = (state.get("shared") or {})
    return bool(sh.get("question")) and bool(sh.get("options"))

def has_solution_answers(state: Dict[str, Any]) -> bool:
    sh = (state.get("shared") or {})
    return bool(sh.get("answer")) and bool(sh.get("explanation"))

def has_score(state: Dict[str, Any]) -> bool:
    sc = state.get("score") or {}
    return bool(sc)

def has_files_to_preprocess(state: Dict[str, Any]) -> bool:
    # 파일 전처리 훅: 필요 시 사용자가 올린 파일/ID 기준으로 True 리턴
    art = state.get("artifacts") or {}
    
    # PDF 파일이 있으면 항상 전처리 수행 (새로운 파일이므로)
    pdf_ids = art.get("pdf_ids", [])
    
    # 이미지 파일도 체크 (새로 추가)
    image_ids = art.get("image_ids", [])
    
    # 디버깅 로그 추가
    print(f"🔍 [전처리 체크] PDF 파일: {pdf_ids}")
    print(f"🔍 [전처리 체크] 이미지 파일: {image_ids}")
    print(f"🔍 [전처리 체크] artifacts 전체: {art}")
    result = bool(pdf_ids) or bool(image_ids)
    print(f"🔍 [전처리 체크] 결과: {result} (PDF 있음: {bool(pdf_ids)}, 이미지 있음: {bool(image_ids)})")
    
    # PDF 또는 이미지 파일이 있으면 전처리 필요 (기존 문제 상관없이)
    return result

# ========== 파일 처리 ==========
def extract_image_paths(user_query: str) -> List[str]:
    """사용자 입력에서 이미지 파일 경로 추출"""
    import re
    
    # 이미지 파일 확장자 패턴 (문자열 끝 앵커 제거)
    image_extensions = r'\.(jpg|jpeg|png|gif|bmp|tiff|webp)'
    
    # 1. 따옴표로 둘러싸인 경로
    quoted_pattern = r'["\']([^"\']+' + image_extensions + r')["\']'
    # 2. 간단한 파일명 패턴 (확장자만 확인)
    filename_pattern = r'([^"\s]+' + image_extensions + r')'
    
    valid_paths = []
    
    # 따옴표로 둘러싸인 경로 먼저 찾기
    quoted_matches = re.findall(quoted_pattern, user_query, re.IGNORECASE)
    for match in quoted_matches:
        path = match.strip()
        if os.path.exists(path):
            valid_paths.append(path)
            print(f"🖼️ 따옴표로 둘러싸인 이미지 파일 발견: {path}")
        else:
            print(f"⚠️ 따옴표로 둘러싸인 이미지 파일을 찾을 수 없음: {path}")
    
    # 공백으로 구분된 파일명 찾기
    filename_matches = re.findall(filename_pattern, user_query, re.IGNORECASE)
    for match in filename_matches:
        # re.findall이 그룹을 반환할 때 튜플이 될 수 있음
        if isinstance(match, tuple):
            filename = match[0]  # 첫 번째 그룹 사용
        else:
            filename = match
        
        # 파일명에서 불필요한 공백 제거
        filename = filename.strip()
        
        # 현재 디렉토리에서 파일 찾기
        if os.path.exists(filename):
            valid_paths.append(filename)
            print(f"🖼️ 파일명으로 이미지 파일 발견: {filename}")
        else:
            # teacher/solution/pdf_outputs 디렉토리에서 찾기
            pdf_outputs_path = os.path.join("teacher", "solution", "pdf_outputs", filename)
            if os.path.exists(pdf_outputs_path):
                valid_paths.append(pdf_outputs_path)
                print(f"🖼️ pdf_outputs에서 이미지 파일 발견: {pdf_outputs_path}")
            else:
                # 현재 디렉토리의 하위 디렉토리들에서 찾기
                for root, dirs, files in os.walk("."):
                    if filename in files:
                        full_path = os.path.join(root, filename)
                        valid_paths.append(full_path)
                        print(f"🖼️ 하위 디렉토리에서 이미지 파일 발견: {full_path}")
                        break
                else:
                    print(f"⚠️ 이미지 파일을 찾을 수 없음: {filename}")
    
    return valid_paths

def extract_problems_from_pdf(pdf_preprocessor, file_paths: List[str]) -> List[Dict]:
    """PDF 파일에서 문제 추출 (pdf_preprocessor 사용)"""
    results: List[Dict] = []
    for p in file_paths:
        try:
            items = pdf_preprocessor.extract(p)  # [{question, options}]
            if isinstance(items, list):
                results.extend(items)
        except Exception as e:
            print(f"[WARN] PDF 추출 실패({p}): {e}")
    return results

def extract_problems_from_images(image_paths: List[str]) -> List[Dict]:
    """이미지 파일에서 문제 추출 (img2json_generation 사용)"""
    try:
        # img2json_generation 모듈 import
        from agents.solution.img2json_generation import call_gpt_on_images
        
        print(f"🖼️ 이미지에서 문제 추출 시작: {len(image_paths)}개 파일")
        
        # 이미지 파일 존재 여부 재확인
        valid_paths = []
        for path in image_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                print(f"⚠️ 이미지 파일을 찾을 수 없음: {path}")
        
        if not valid_paths:
            print("❌ 처리할 수 있는 이미지 파일이 없습니다.")
            return []
        
        print(f"🖼️ 유효한 이미지 파일: {len(valid_paths)}개")
        
        # call_gpt_on_images 함수 호출
        result = call_gpt_on_images(valid_paths)
        
        if not result:
            print("⚠️ 이미지에서 문제를 추출하지 못했습니다.")
            return []
        
        # 오류가 있는 경우 처리
        if "error" in result:
            print(f"⚠️ 이미지 처리 중 오류 발생: {result['error']}")
            if "raw_response" in result:
                print(f"🔍 원본 응답: {result['raw_response'][:200]}...")
            return []
        
        if "problems" not in result:
            print("⚠️ 이미지에서 문제를 추출하지 못했습니다.")
            return []
        
        problems = result["problems"]
        print(f"🖼️ 이미지에서 {len(problems)}개 문제 추출 성공")
        
        # img2json_generation의 결과를 teacher_graph 형식에 맞게 변환
        converted_problems = []
        skipped_count = 0
        
        for problem in problems:
            if isinstance(problem, dict):
                # skipped 문제는 제외하되 카운트
                if problem.get("skipped", False):
                    skipped_count += 1
                    print(f"⚠️ 문제 {problem.get('number', 'N/A')} 건너뜀: {problem.get('reason', '이유 없음')}")
                    continue
                
                # teacher_graph 형식으로 변환
                question = str(problem.get("question", "")).strip()
                options = problem.get("options", [])
                
                # options가 리스트가 아니면 변환
                if isinstance(options, str):
                    options = [x.strip() for x in options.splitlines() if x.strip()]
                elif not isinstance(options, list):
                    options = []
                
                # 빈 문제나 보기가 없는 문제는 제외
                if not question or not options:
                    print(f"⚠️ 문제 {problem.get('number', 'N/A')} 건너뜀: 질문 또는 보기가 비어있음")
                    continue
                
                converted_problem = {
                    "question": question,
                    "options": options
                }
                converted_problems.append(converted_problem)
        
        print(f"🖼️ 변환 완료: {len(converted_problems)}개 문제 (건너뜀: {skipped_count}개)")
        return converted_problems
        
    except ImportError as e:
        print(f"❌ img2json_generation 모듈을 불러올 수 없습니다: {e}")
        print("💡 이미지 처리를 위해서는 agents.solution.img2json_generation 모듈이 필요합니다.")
        return []
    except Exception as e:
        print(f"❌ 이미지 문제 추출 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return []
