import os
import json
import re
import sys
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

# Add the TestGenerator directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import DEFAULT_SAVE_DIR, DEFAULT_WEAKNESS_DIR

def extract_quiz_params(
    user_question: str,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    temperature: float = 0.2,
    groq_api_key: str = None,
    base_url: str = "https://api.groq.com/openai/v1"
) -> dict:
    """사용자 질문에서 save_to_file, filename, difficulty, mode를 LLM을 통해 추출합니다."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai 패키지가 설치되어 있지 않습니다. 'pip install openai'로 설치하세요.")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = groq_api_key or os.getenv("GROQAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQAI_API_KEY 환경변수 또는 groq_api_key 인자가 필요합니다.")
    
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    prompt = f"""다음 사용자 질문에서 아래 4가지 정보를 추출하여 JSON 형식으로 출력하세요.

- save_to_file: 문제 생성 결과를 파일로 저장할지 여부 (True/False)
- filename: 저장할 파일명 (사용자가 명시하지 않으면 null)
- difficulty: 난이도 ("초급", "중급", "고급" 중 하나, 명시 없으면 "중급")
- mode: "full_exam" 또는 "subject_quiz" 또는 "weakness_quiz" 중 하나

예시: {{"save_to_file": true, "filename": "내문제.json", "difficulty": "고급", "mode": "weakness_quiz"}}

사용자 질문: {user_question}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=temperature
        )
        content = response.choices[0].message.content
        match = re.search(r'\{[\s\S]*\}', content)
        return json.loads(match.group(0)) if match else {}
    except Exception:
        return {}

def save_to_json(exam_result: Dict[str, Any], filename: str = None, save_dir: str = None) -> str:
    """시험 결과를 JSON 파일로 저장"""
    if save_dir is None:
        save_dir = DEFAULT_SAVE_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if "exam_title" in exam_result:
            filename = f"정보처리기사_25문제_{timestamp}.json"
        elif "quiz_type" in exam_result and exam_result["quiz_type"] == "weakness_based_llm":
            concepts = "_".join([c[:10] for c in exam_result.get("weakness_concepts", ["취약점"])[:3]])
            count = exam_result.get("quiz_count", 0)
            filename = f"LLM취약점맞춤_{concepts}_{count}문제_{timestamp}.json"
        else:
            subject = exam_result.get("subject_area", "문제")
            count = exam_result.get("quiz_count", 0)
            filename = f"{subject}_{count}문제_{timestamp}.json"
    
    if not os.path.isabs(filename):
        filename = os.path.join(save_dir, filename)
    elif not filename.startswith(save_dir):
        filename = os.path.join(save_dir, os.path.basename(filename))
        
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(exam_result, f, ensure_ascii=False, indent=2)
    
    return filename

def save_weakness_result(result: Dict[str, Any], filename: str = None) -> str:
    """취약점 기반 문제 결과를 weakness 폴더에 저장"""
    os.makedirs(DEFAULT_WEAKNESS_DIR, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        concepts = "_".join([c[:10] for c in result.get("weakness_concepts", ["취약점"])[:3]])
        count = result.get("quiz_count", 0)
        filename = f"weakness_quiz_{concepts}_{count}문제_{timestamp}.json"
    
    if not os.path.isabs(filename):
        filename = os.path.join(DEFAULT_WEAKNESS_DIR, filename)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return filename

def generate_weakness_quiz_from_analysis_llm(
    agent,
    analysis_file_path: str,
    target_count: int = 10,
    difficulty: str = "중급",
    save_to_file: bool = True,
    filename: str = None
) -> Dict[str, Any]:
    """
    LLM을 활용하여 분석 결과 파일을 기반으로 취약점 맞춤 문제를 생성하는 편의 함수
    
    Args:
        agent: InfoProcessingExamAgent 인스턴스
        analysis_file_path: 분석 결과 JSON 파일 경로
        target_count: 생성할 문제 수
        difficulty: 난이도
        save_to_file: 파일 저장 여부
        filename: 저장할 파일명
        
    Returns:
        생성 결과
    """
    input_data = {
        "mode": "weakness_quiz",
        "analysis_file_path": analysis_file_path,
        "target_count": target_count,
        "difficulty": difficulty,
        "save_to_file": save_to_file,
        "filename": filename
    }
    
    return agent.execute(input_data)

def generate_weakness_quiz_from_text_llm(
    agent,
    analysis_text: str,
    target_count: int = 8,
    difficulty: str = "중급",
    save_to_file: bool = True,
    filename: str = None
) -> Dict[str, Any]:
    """
    LLM을 활용하여 분석 텍스트를 기반으로 취약점 맞춤 문제를 생성하는 편의 함수
    
    Args:
        agent: InfoProcessingExamAgent 인스턴스
        analysis_text: 분석 텍스트 내용
        target_count: 생성할 문제 수
        difficulty: 난이도
        save_to_file: 파일 저장 여부
        filename: 저장할 파일명
        
    Returns:
        생성 결과
    """
    input_data = {
        "mode": "weakness_quiz",
        "raw_analysis_text": analysis_text,
        "target_count": target_count,
        "difficulty": difficulty,
        "save_to_file": save_to_file,
        "filename": filename
    }
    
    return agent.execute(input_data)
