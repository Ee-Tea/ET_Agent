import os
import time
import sys
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from pathlib import Path

# Add the TestGenerator directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.base_agent import BaseAgent
from config import (
    DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, 
    DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES, SUBJECT_AREAS,
    DEFAULT_DATA_FOLDER, DEFAULT_SAVE_DIR, DEFAULT_WEAKNESS_DIR
)
from weakness.weakness_analyzer import WeaknessAnalyzer
from weakness.weakness_quiz_generator import WeaknessQuizGenerator
from core.quiz_workflow import QuizWorkflow
from utils.utils import save_to_json, save_weakness_result

class InfoProcessingExamAgent(BaseAgent):
    """
    정보처리기사 출제기준에 맞는 25문제 자동 출제 에이전트 (순차 처리 버전)
    LLM 기반 취약점 분석 및 맞춤형 문제 생성
    """
    
    def __init__(self, data_folder=DEFAULT_DATA_FOLDER, groq_api_key=None):
        """초기화"""
        self.data_folder = data_folder
        os.makedirs(self.data_folder, exist_ok=True)
        
        # Groq API 키 설정
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        elif not os.getenv("GROQ_API_KEY"):
            raise ValueError("Groq API 키가 필요합니다.")
        
        # RAG 엔진 초기화
        from rag.rag_engine import RAGEngine
        self.rag_engine = RAGEngine(data_folder=data_folder)
        
        # RAG 엔진 벡터 스토어 초기화
        print("🔨 RAG 엔진 벡터 스토어 초기화 중...")
        if not self.rag_engine.build_vectorstore_from_all_pdfs():
            raise ValueError(f"'{data_folder}' 폴더에서 PDF 파일을 로드할 수 없습니다.")
        print("✅ RAG 엔진 벡터 스토어 초기화 완료")
        
        self.llm = None
        self.workflow = None
        self.weakness_analyzer = None
        self.weakness_quiz_generator = None
        
        self._initialize_models()
        self._build_components()
            
    @property
    def name(self) -> str:
        """에이전트의 고유한 이름을 반환합니다."""
        return "InfoProcessingExamAgent"

    @property
    def description(self) -> str:
        """에이전트의 역할에 대한 설명을 반환합니다."""
        return "정보처리기사 출제기준에 맞는 25문제를 자동으로 생성하는 에이전트입니다. PDF 문서를 기반으로 5개 과목별로 문제를 생성하며, LLM을 활용하여 학습자의 취약점을 자동 분석하고 맞춤형 문제를 생성합니다."

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트의 주된 로직을 실행하는 메서드입니다.
        
        Args:
            input_data (Dict[str, Any]): 에이전트 실행에 필요한 입력 데이터
                - mode: "full_exam", "subject_quiz", "weakness_quiz"
                - difficulty: "초급", "중급", "고급" (기본값: "중급")
                - subject_area: 특정 과목명 (subject_quiz 모드일 때)
                - target_count: 생성할 문제 수 (subject_quiz, weakness_quiz 모드일 때)
                - save_to_file: JSON 파일 저장 여부 (기본값: False)
                - filename: 저장할 파일명 (선택사항)
                - analysis_file_path: 분석 결과 JSON 파일 경로 (weakness_quiz 모드일 때)
                - raw_analysis_text: 분석 텍스트 (weakness_quiz 모드일 때)
                
        Returns:
            Dict[str, Any]: 에이전트 실행 결과 데이터
                - success: 성공 여부
                - result: 생성된 시험 데이터
                - error: 오류 메시지 (실패시)
                - file_path: 저장된 파일 경로 (저장시)
        """
        try:
            # 입력 데이터 검증 및 기본값 설정
            mode = input_data.get("mode", "full_exam")
            difficulty = input_data.get("difficulty", "중급")
            save_to_file = input_data.get("save_to_file", False)
            filename = input_data.get("filename")
            
            # RAG 엔진 상태 확인
            vectorstore_info = self.rag_engine.get_vectorstore_info()
            if not vectorstore_info.get("is_initialized", False):
                return {
                    "success": False,
                    "error": f"RAG 엔진이 초기화되지 않았습니다."
                }
            
            if mode == "full_exam":
                # 전체 25문제 생성
                result = self._generate_full_exam(difficulty)
            elif mode == "subject_quiz":
                # 특정 과목 문제 생성
                subject_area = input_data.get("subject_area")
                target_count = input_data.get("target_count", 5)
                
                if not subject_area or subject_area not in SUBJECT_AREAS:
                    return {
                        "success": False,
                        "error": f"유효하지 않은 과목명입니다. 가능한 과목: {list(SUBJECT_AREAS.keys())}"
                    }
                
                result = self._generate_subject_quiz(subject_area, target_count, difficulty)
            elif mode == "weakness_quiz":
                # 취약점 기반 문제 생성
                result = self.weakness_quiz_generator.generate_weakness_quiz(input_data, difficulty)
            else:
                return {
                    "success": False,
                    "error": "유효하지 않은 모드입니다. 'full_exam', 'subject_quiz', 'weakness_quiz' 중 하나를 사용하세요."
                }
            
            # 오류 확인
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"]
                }
            
            response = {
                "success": True,
                "result": result
            }
            
            # 파일 저장 요청 시
            if save_to_file:
                try:
                    if mode == "weakness_quiz":
                        # 취약점 기반 문제는 weakness 폴더에 저장
                        file_path = save_weakness_result(result, filename)
                    else:
                        # 일반 문제는 test 폴더에 저장
                        file_path = save_to_json(result, filename)
                    response["file_path"] = file_path
                except Exception as e:
                    response["save_error"] = str(e)
            
            return response
            
        except Exception as e:
            return {
                "success": False,
                "error": f"에이전트 실행 중 오류 발생: {str(e)}"
            }

    def _initialize_models(self):
        """LLM 모델 초기화"""
        try:
            self.llm = ChatGroq(
                model=DEFAULT_MODEL,
                temperature=0.0,
                max_tokens=DEFAULT_MAX_TOKENS,
                timeout=DEFAULT_TIMEOUT,
                max_retries=DEFAULT_MAX_RETRIES
            )
            
            # 연결 테스트
            self.llm.invoke("안녕하세요")
            
        except Exception as e:
            raise ValueError(f"LLM 모델 초기화 중 오류 발생: {e}")

    def _build_components(self):
        """컴포넌트들 초기화"""
        # 워크플로우 초기화
        self.workflow = QuizWorkflow(self.llm, self.rag_engine)
        
        # 취약점 분석기 초기화
        self.weakness_analyzer = WeaknessAnalyzer(self.llm)
        
        # 취약점 문제 생성기 초기화
        self.weakness_quiz_generator = WeaknessQuizGenerator(self.llm, self.workflow)

    def _generate_subject_quiz(self, subject_area: str, target_count: int = 5, difficulty: str = "중급") -> Dict[str, Any]:
        """특정 과목의 문제를 순차로 생성"""
        if subject_area not in SUBJECT_AREAS:
            return {"error": f"유효하지 않은 과목: {subject_area}"}
        
        keywords = SUBJECT_AREAS[subject_area]["keywords"]
        
        all_validated_questions = []
        
        current_round = 0
        max_rounds = 10  # 최대 라운드 수
        
        while len(all_validated_questions) < target_count and current_round < max_rounds:
            current_round += 1
            remaining_needed = target_count - len(all_validated_questions)
            
            # 키워드를 2-3개씩 묶어서 순차 처리
            for i in range(0, len(keywords), 2):
                if len(all_validated_questions) >= target_count:
                    break
                    
                combo = " ".join(keywords[i:i+3])
                
                result = self._generate_with_keywords(combo, subject_area, remaining_needed, difficulty)
                
                if "questions" in result and result["questions"]:
                    new_questions = []
                    existing_questions = [q.get('question', '') for q in all_validated_questions]
                    
                    for q in result["questions"]:
                        if q.get('question', '') not in existing_questions:
                            new_questions.append(q)
                            existing_questions.append(q.get('question', ''))
                    
                    all_validated_questions.extend(new_questions)
                
                if len(all_validated_questions) >= target_count:
                    break
            
            if current_round < max_rounds and len(all_validated_questions) < target_count:
                time.sleep(2)
        
        final_questions = all_validated_questions[:target_count]
        
        return {
            "subject_area": subject_area,
            "difficulty": difficulty,
            "requested_count": target_count,
            "quiz_count": len(final_questions),
            "questions": final_questions,
            "status": "SUCCESS" if len(final_questions) >= target_count else "PARTIAL"
        }

    def _generate_with_keywords(self, query: str, subject_area: str, needed_count: int, difficulty: str) -> Dict[str, Any]:
        """특정 키워드로 문제 생성"""
        try:
            initial_state = {
                "query": query,
                "quiz_count": needed_count,
                "target_quiz_count": needed_count,
                "difficulty": difficulty,
                "generation_attempts": 0,
                "quiz_questions": [],
                "validated_questions": [],
                "subject_area": subject_area
            }
            
            result = self.workflow.invoke(initial_state)
            
            if result.get("error"):
                return {"error": result["error"]}
            
            return {
                "questions": result.get("validated_questions", []),
                "used_sources": result.get("used_sources", [])
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _generate_full_exam(self, difficulty: str = "중급") -> Dict[str, Any]:
        """정보처리기사 전체 25문제를 순차로 생성"""
        start_time = time.time()
        
        full_exam_result = {
            "exam_title": "정보처리기사 모의고사",
            "total_questions": 0,
            "difficulty": difficulty,
            "subjects": {},
            "all_questions": [],
            "generation_summary": {},
            "failed_subjects": [],
            "model_info": "Groq llama-4-scout-17b-16e-instruct with LLM weakness analysis"
        }
        
        total_generated = 0
        
        for i, (subject_area, subject_info) in enumerate(SUBJECT_AREAS.items(), 1):
            target_count = subject_info["count"]
            
            subject_result = self._generate_subject_quiz(
                subject_area=subject_area,
                target_count=target_count,
                difficulty=difficulty
            )
            
            if "error" in subject_result:
                full_exam_result["failed_subjects"].append({
                    "subject": subject_area,
                    "error": subject_result["error"]
                })
            else:
                questions = subject_result["questions"]
                actual_count = len(questions)
                total_generated += actual_count
                
                full_exam_result["subjects"][subject_area] = {
                    "requested_count": target_count,
                    "actual_count": actual_count,
                    "questions": questions,
                    "status": subject_result.get("status", "UNKNOWN")
                }
                
                full_exam_result["all_questions"].extend(questions)
            
            if i < 5:
                time.sleep(2)
        
        elapsed_time = time.time() - start_time
        
        full_exam_result["total_questions"] = total_generated
        full_exam_result["generation_summary"] = {
            "target_total": 25,
            "actual_total": total_generated,
            "success_rate": f"{total_generated/25*100:.1f}%",
            "successful_subjects": 5 - len(full_exam_result["failed_subjects"]),
            "failed_subjects": len(full_exam_result["failed_subjects"]),
            "completion_status": "COMPLETE" if total_generated >= 25 else "PARTIAL",
            "generation_time": f"{elapsed_time:.1f}초"
        }
        
        return full_exam_result
