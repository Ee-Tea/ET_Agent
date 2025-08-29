import os
import sys
from pathlib import Path

# Add the TestGenerator directory to the Python path
sys.path.append(str(Path(__file__).parent))

from interactive_interface import interactive_menu_llm, test_weakness_analysis

def main():
    """메인 실행 함수"""
    # Milvus 환경변수 설정 (localhost로 강제 설정)
    os.environ.setdefault("MILVUS_HOST", "localhost")
    os.environ.setdefault("MILVUS_PORT", "19530")
    os.environ.setdefault("MILVUS_COLLECTION", "info_exam_chunks")
    os.environ.setdefault("MILVUS_TOPK", "15")
    
    # Groq API 키 확인
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    # 사용 방법 선택
    print("🧠 LLM 기반 정보처리기사 맞춤형 문제 생성 에이전트")
    print("1. 대화형 인터페이스 사용")
    print("2. LLM 취약점 분석 기능 테스트")
    
    choice = input("선택하세요 (1 또는 2): ").strip()
    
    if choice == "1":
        interactive_menu_llm()
    elif choice == "2":
        test_weakness_analysis()
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()
