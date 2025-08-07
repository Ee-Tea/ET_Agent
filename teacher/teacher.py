import json
from pathlib import Path
import sys
from typing import Dict, Any
import os
from dotenv import load_dotenv
from langsmith import traceable

from agents.analisys.analysis_agent import AnalysisAgent, print_analysis_result
from agents.base_agent import BaseAgent

class Orchestrator:
    """
    전체 워크플로우를 관리하고, 사용자 요청에 따라 
    적절한 에이전트를 선택하고 실행하는 오케스트레이터 클래스입니다.
    """
    
    def __init__(self):
        # .env 파일에서 환경 변수 로드
        load_dotenv()
        # LangSmith 추적 환경 변수 확인
        if not os.getenv("LANGCHAIN_API_KEY"):
            print("경고: LANGCHAIN_API_KEY 환경 변수가 설정되지 않았습니다.")
            print(".env 파일에 키를 추가하거나 직접 환경 변수를 설정해주세요.")

        self.agents: Dict[str, BaseAgent] = {
            "analysis": AnalysisAgent(),
            # 다른 에이전트들을 여기에 추가할 수 있습니다.
            # "problem_generation": ProblemGenerationAgent(), 
        }

    @traceable(name="Orchestrator Run")
    def run(self, agent_name: str, input_file_path: str):
        """
        지정된 에이전트를 실행하고, 파일 입출력을 처리합니다.
        
        Args:
            agent_name (str): 실행할 에이전트의 이름입니다.
            input_file_path (str): 에이전트에 전달할 입력 데이터 파일 경로입니다.
        """
        # 1. 에이전트 선택
        agent = self.agents.get(agent_name)
        if not agent:
            print(f"오류: '{agent_name}'이라는 이름의 에이전트를 찾을 수 없습니다.")
            sys.exit(1)
            
        # 2. 입력 파일 로드
        input_file = Path(input_file_path)
        if not input_file.exists():
            print(f"오류: 입력 파일을 찾을 수 없습니다. ({input_file})")
            sys.exit(1)
            
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            print(f"파일 '{input_file}' 로드 성공")
        except json.JSONDecodeError:
            print(f"오류: '{input_file}'이 올바른 JSON 형식이 아닙니다.")
            sys.exit(1)
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {e}")
            sys.exit(1)

        # 3. 에이전트 실행
        try:
            print(f"🚀 '{agent_name}' 에이전트 실행 시작...")
            result = agent.execute(input_data)
            
            # 결과 검증
            if not isinstance(result, dict):
                raise ValueError("에이전트가 올바른 형식의 결과를 반환하지 않았습니다.")
                
            if result.get("status") == "error":
                print(f"⚠️  에이전트 실행 중 내부 오류 발생: {result.get('error_message', '알 수 없는 오류')}")
            else:
                print(f"✅ '{agent_name}' 에이전트 실행 완료.")
                
        except Exception as e:
            print(f"❌ 에이전트 실행 중 오류 발생: {e}")
            # 오류 발생 시에도 기본 구조로 결과 생성
            result = {
                "status": "error",
                "error_message": str(e),
                "metadata": {
                    "total_problems": 0,
                    "correct_count": 0,
                    "score": 0
                },
                "grading": {"results": [], "details": []},
                "analysis": {},
                "raw_data": {}
            }
            
        # 4. 결과 처리 및 저장
        self.handle_result(result, agent_name, input_file)

    def handle_result(self, result: Dict[str, Any], agent_name: str, input_file: Path):
        """
        에이전트 실행 결과를 처리하고 저장합니다.
        
        Args:
            result (Dict[str, Any]): 에이전트 실행 결과.
            agent_name (str): 실행된 에이전트의 이름.
            input_file (Path): 원본 입력 파일의 경로.
        """
        # 결과 출력
        if agent_name == "analysis":
            print_analysis_result(result)
        else:
            # 다른 에이전트들의 결과 출력 로직
            print("\n--- 실행 결과 ---")
            if result.get("status") == "success":
                print("✅ 성공적으로 완료되었습니다.")
                if "metadata" in result:
                    print("\n📊 메타데이터:")
                    for key, value in result["metadata"].items():
                        print(f"  - {key}: {value}")
            else:
                print("❌ 오류가 발생했습니다.")
                if "error_message" in result:
                    print(f"오류 메시지: {result['error_message']}")
            
            print("\n전체 결과:")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        # 결과 파일 저장
        output_file = input_file.with_name(f"{input_file.stem}_{agent_name}_result.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 결과가 '{output_file}'에 저장되었습니다.")
            
            # 요약 정보 출력
            if result.get("status") == "success" and result.get("metadata"):
                metadata = result["metadata"]
                print(f"\n📋 저장된 결과 요약:")
                print(f"  - 상태: {result.get('status', '알 수 없음')}")
                if agent_name == "analysis":
                    print(f"  - 총 문제 수: {metadata.get('total_problems', 0)}")
                    print(f"  - 정답률: {metadata.get('score', 0)}%")
                    print(f"  - 오답 여부: {'있음' if metadata.get('has_mistakes', False) else '없음'}")
                    
        except Exception as e:
            print(f"❌ 결과 저장 중 오류 발생: {e}")
            print("결과가 저장되지 않았지만, 분석은 완료되었습니다.")


if __name__ == "__main__":
    # 명령줄 인자 파싱
    # 예: python teacher.py analysis "path/to/your/input.json"
    if len(sys.argv) < 3:
        print("🔧 사용법: python teacher.py [agent_name] [input_file_path]")
        print("\n📋 사용 가능한 에이전트:")
        print("  - analysis: 학습자 답안을 분석하고 피드백을 생성합니다")
        print("\n💡 예시:")
        print("  python teacher.py analysis ./test_sample/analysis_sample.json")
        print("  python teacher.py analysis C:/path/to/student_answers.json")
        sys.exit(1)
        
    agent_to_run = sys.argv[1]
    file_path = sys.argv[2]
    
    print("🎓 ET_Agent Teacher System")
    print("=" * 40)
    
    orchestrator = Orchestrator()
    orchestrator.run(agent_name=agent_to_run, input_file_path=file_path)
