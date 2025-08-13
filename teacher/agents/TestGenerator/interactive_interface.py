import os
import json
import sys
from pathlib import Path

# Add the TestGenerator directory to the Python path
sys.path.append(str(Path(__file__).parent))

from core.main_agent import InfoProcessingExamAgent
from utils.utils import generate_weakness_quiz_from_analysis_llm, generate_weakness_quiz_from_text_llm

def interactive_menu_llm():
    """LLM 기반 취약점 분석을 포함한 대화형 메뉴 시스템"""
    try:
        agent = InfoProcessingExamAgent()
        
        print(f"\n🧠 {agent.name} 초기화 완료")
        print(f"📖 설명: {agent.description}")
        
        while True:
            print("\n" + "="*70)
            print("  🧠 LLM 기반 정보처리기사 맞춤형 문제 생성 에이전트")
            print("="*70)
            print("1. 전체 25문제 생성")
            print("2. 특정 과목만 문제 생성")
            print("3. 🧠 LLM 취약점 분석 + 맞춤 문제 생성 (파일)")
            print("4. 🧠 LLM 취약점 분석 + 맞춤 문제 생성 (텍스트)")
            print("5. 사용 가능한 PDF 목록 보기")
            print("0. 종료")
            print("-"*70)
            
            choice = input("선택하세요: ").strip()
            
            if choice == "1":
                # 전체 25문제 생성
                difficulty = input("난이도 (초급/중급/고급, 기본값: 중급): ").strip()
                if difficulty not in ["초급", "중급", "고급"]:
                    difficulty = "중급"
                
                save_option = input("JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower()
                save_to_file = save_option == 'y'
                
                filename = None
                if save_to_file:
                    filename_input = input("파일명 (엔터: 자동생성): ").strip()
                    if filename_input:
                        filename = filename_input
                
                input_data = {
                    "mode": "full_exam",
                    "difficulty": difficulty,
                    "save_to_file": save_to_file,
                    "filename": filename
                }
                
                print("\n전체 25문제 생성 중...")
                result = agent.execute(input_data)
                
                if result["success"]:
                    exam_data = result["result"]
                    summary = exam_data.get("generation_summary", {})
                    
                    print(f"\n✅ 생성 완료!")
                    print(f"전체 문제 수: {summary.get('actual_total', 0)}/25문제")
                    print(f"성공률: {summary.get('success_rate', '0%')}")
                    print(f"소요 시간: {summary.get('generation_time', 'N/A')}")
                    
                    if "file_path" in result:
                        print(f"📁 저장 경로: {result['file_path']}")
                else:
                    print(f"❌ 실패: {result['error']}")
            
            elif choice == "2":
                # 특정 과목 문제 생성
                print("\n[정보처리기사 과목 선택]")
                from config import SUBJECT_AREAS
                subjects = list(SUBJECT_AREAS.keys())
                for i, subject in enumerate(subjects, 1):
                    count = SUBJECT_AREAS[subject]["count"]
                    print(f"{i}. {subject} ({count}문제)")
                
                try:
                    subject_choice = int(input("과목 번호 선택: "))
                    if 1 <= subject_choice <= len(subjects):
                        selected_subject = subjects[subject_choice - 1]
                        default_count = SUBJECT_AREAS[selected_subject]["count"]
                        
                        count_input = input(f"생성할 문제 수 (기본값: {default_count}): ").strip()
                        target_count = int(count_input) if count_input.isdigit() else default_count
                        
                        difficulty = input("난이도 (초급/중급/고급, 기본값: 중급): ").strip()
                        if difficulty not in ["초급", "중급", "고급"]:
                            difficulty = "중급"
                        
                        save_option = input("JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower()
                        save_to_file = save_option == 'y'
                        
                        filename = None
                        if save_to_file:
                            filename_input = input("파일명 (엔터: 자동생성): ").strip()
                            if filename_input:
                                filename = filename_input
                        
                        input_data = {
                            "mode": "subject_quiz",
                            "subject_area": selected_subject,
                            "target_count": target_count,
                            "difficulty": difficulty,
                            "save_to_file": save_to_file,
                            "filename": filename
                        }
                        
                        print(f"\n{selected_subject} 과목 {target_count}문제 생성 중...")
                        result = agent.execute(input_data)
                        
                        if result["success"]:
                            subject_data = result["result"]
                            print(f"✅ 생성 완료!")
                            print(f"{subject_data['subject_area']}: {subject_data['quiz_count']}/{subject_data['requested_count']}문제")
                            print(f"상태: {subject_data.get('status', 'UNKNOWN')}")
                            
                            if "file_path" in result:
                                print(f"📁 저장 경로: {result['file_path']}")
                        else:
                            print(f"❌ 실패: {result['error']}")
                    else:
                        print("유효하지 않은 과목 번호입니다.")
                except ValueError:
                    print("숫자를 입력해주세요.")
            
            elif choice == "3":
                # LLM 기반 취약점 분석 + 맞춤 문제 생성 (파일)
                print("\n🧠 [LLM 기반 취약점 분석 + 맞춤 문제 생성 - 파일]")
                
                analysis_file_path = input("분석 결과 JSON 파일 경로를 입력하세요: ").strip()
                
                if not os.path.exists(analysis_file_path):
                    print("❌ 파일이 존재하지 않습니다.")
                    continue
                
                try:
                    # 분석 파일 미리보기
                    with open(analysis_file_path, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    
                    print(f"\n📋 분석 파일 로드 완료")
                    
                    count_input = input("생성할 문제 수 (기본값: 10): ").strip()
                    target_count = int(count_input) if count_input.isdigit() else 10
                    
                    difficulty = input("난이도 (초급/중급/고급, 기본값: 중급): ").strip()
                    if difficulty not in ["초급", "중급", "고급"]:
                        difficulty = "중급"
                    
                    save_option = input("JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower()
                    save_to_file = save_option == 'y'
                    
                    filename = None
                    if save_to_file:
                        filename_input = input("파일명 (엔터: 자동생성): ").strip()
                        if filename_input:
                            filename = filename_input
                    
                    print(f"\n🧠 LLM이 취약점을 분석하고 맞춤 문제 {target_count}개를 생성 중...")
                    
                    result = generate_weakness_quiz_from_analysis_llm(
                        agent=agent,
                        analysis_file_path=analysis_file_path,
                        target_count=target_count,
                        difficulty=difficulty,
                        save_to_file=save_to_file,
                        filename=filename
                    )
                    
                    if result["success"]:
                        weakness_data = result["result"]
                        print(f"\n✅ LLM 취약점 분석 및 맞춤 문제 생성 완료!")
                        print(f"🧠 LLM이 분석한 취약점 개념: {weakness_data.get('weakness_concepts', [])}")
                        print(f"📚 집중 추천 과목: {weakness_data.get('weakness_analysis', {}).get('subject_focus', [])}")
                        print(f"🎯 추천 난이도: {weakness_data.get('weakness_analysis', {}).get('difficulty_level', '중급')}")
                        print(f"📊 생성된 문제 수: {weakness_data.get('quiz_count', 0)}/{weakness_data.get('requested_count', 0)}")
                        print(f"📈 성공률: {weakness_data.get('generation_summary', {}).get('success_rate', '0%')}")
                        
                        # 학습 우선순위 표시
                        learning_priorities = weakness_data.get('weakness_analysis', {}).get('learning_priorities', [])
                        if learning_priorities:
                            print(f"📝 추천 학습 순서:")
                            for i, priority in enumerate(learning_priorities[:5], 1):
                                print(f"  {i}. {priority}")
                        
                        # 문제 미리보기
                        questions = weakness_data.get("questions", [])
                        if questions and input("\n생성된 문제를 미리보시겠습니까? (y/n): ").strip().lower() == 'y':
                            for i, q in enumerate(questions[:2], 1):
                                weakness_concept = q.get('weakness_concept', '일반')
                                weakness_focus = q.get('weakness_focus', weakness_concept)
                                print(f"\n[🎯 취약점 집중: {weakness_focus}] [문제 {i}]")
                                print(f"❓ {q.get('question', '')}")
                                for option in q.get('options', []):
                                    print(f"{option}")
                                print(f"✅ 정답: {q.get('answer', '')}")
                                print(f"💡 해설: {q.get('explanation', '')}")
                                if i < 2 and i < len(questions):
                                    input("다음 문제를 보려면 Enter를 누르세요...")
                            
                            if len(questions) > 2:
                                print(f"\n... 외 {len(questions)-2}개 문제가 더 있습니다.")
                        
                        if "file_path" in result:
                            print(f"📁 저장 경로: {result['file_path']}")
                    else:
                        print(f"❌ 실패: {result['error']}")
                        
                except Exception as e:
                    print(f"❌ 분석 파일 처리 중 오류: {e}")
            
            elif choice == "4":
                # LLM 기반 취약점 분석 + 맞춤 문제 생성 (텍스트)
                print("\n🧠 [LLM 기반 취약점 분석 + 맞춤 문제 생성 - 텍스트 입력]")
                
                print("학습자의 취약점이나 분석 내용을 자유롭게 입력하세요.")
                print("예: '자료흐름도 구성요소 이해 부족, SQL 조인 연산 실수 많음, UML 다이어그램 해석 어려움'")
                print("(여러 줄 입력 가능, 완료 후 빈 줄에서 Enter)")
                
                analysis_lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    analysis_lines.append(line)
                
                analysis_text = "\n".join(analysis_lines)
                
                if not analysis_text.strip():
                    print("❌ 분석 내용이 입력되지 않았습니다.")
                    continue
                
                print(f"\n📝 입력된 분석 내용:")
                print(f"{analysis_text[:200]}...")
                
                count_input = input("\n생성할 문제 수 (기본값: 8): ").strip()
                target_count = int(count_input) if count_input.isdigit() else 8
                
                difficulty = input("난이도 (초급/중급/고급, 기본값: 중급): ").strip()
                if difficulty not in ["초급", "중급", "고급"]:
                    difficulty = "중급"
                
                save_option = input("JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower()
                save_to_file = save_option == 'y'
                
                filename = None
                if save_to_file:
                    filename_input = input("파일명 (엔터: 자동생성): ").strip()
                    if filename_input:
                        filename = filename_input
                
                print(f"\n🧠 LLM이 입력 내용을 분석하고 맞춤 문제 {target_count}개를 생성 중...")
                
                result = generate_weakness_quiz_from_text_llm(
                    agent=agent,
                    analysis_text=analysis_text,
                    target_count=target_count,
                    difficulty=difficulty,
                    save_to_file=save_to_file,
                    filename=filename
                )
                
                if result["success"]:
                    weakness_data = result["result"]
                    print(f"\n✅ LLM 텍스트 분석 및 맞춤 문제 생성 완료!")
                    print(f"🧠 LLM이 추출한 취약점: {weakness_data.get('weakness_concepts', [])}")
                    print(f"📚 집중 추천 과목: {weakness_data.get('weakness_analysis', {}).get('subject_focus', [])}")
                    print(f"🎯 LLM 추천 난이도: {weakness_data.get('weakness_analysis', {}).get('difficulty_level', '중급')}")
                    print(f"📊 생성된 문제 수: {weakness_data.get('quiz_count', 0)}/{weakness_data.get('requested_count', 0)}")
                    print(f"📈 성공률: {weakness_data.get('generation_summary', {}).get('success_rate', '0%')}")
                    
                    # 추천 문제 유형 표시
                    question_types = weakness_data.get('weakness_analysis', {}).get('question_types', [])
                    if question_types:
                        print(f"📋 추천 문제 유형: {', '.join(question_types)}")
                    
                    # 문제 미리보기
                    questions = weakness_data.get("questions", [])
                    if questions and input("\n생성된 문제를 미리보시겠습니까? (y/n): ").strip().lower() == 'y':
                        for i, q in enumerate(questions[:2], 1):
                            weakness_concept = q.get('weakness_concept', '일반')
                            weakness_focus = q.get('weakness_focus', weakness_concept)
                            print(f"\n[🎯 취약점 집중: {weakness_focus}] [문제 {i}]")
                            print(f"❓ {q.get('question', '')}")
                            for option in q.get('options', []):
                                print(f"{option}")
                            print(f"✅ 정답: {q.get('answer', '')}")
                            print(f"💡 해설: {q.get('explanation', '')}")
                            if i < 2 and i < len(questions):
                                input("다음 문제를 보려면 Enter를 누르세요...")
                        
                        if len(questions) > 2:
                            print(f"\n... 외 {len(questions)-2}개 문제가 더 있습니다.")
                    
                    if "file_path" in result:
                        print(f"📁 저장 경로: {result['file_path']}")
                else:
                    print(f"❌ 실패: {result['error']}")
            
            elif choice == "5":
                # PDF 파일 목록 보기 (RAG 엔진 사용)
                pdf_files = agent.rag_engine.get_pdf_files()
                if pdf_files:
                    print(f"\n=== '{agent.rag_engine.data_folder}' 폴더의 PDF 파일 목록 ===")
                    for i, file_path in enumerate(pdf_files, 1):
                        filename = os.path.basename(file_path)
                        file_size = os.path.getsize(file_path) / 1024
                        print(f"{i}. {filename} ({file_size:.1f} KB)")
                else:
                    print(f"'{agent.rag_engine.data_folder}' 폴더에 PDF 파일이 없습니다.")
            
            elif choice == "0":
                print("🧠 LLM 기반 에이전트를 종료합니다.")
                break
            
            else:
                print("잘못된 선택입니다. 0~5 중에서 선택해주세요.")
    
    except Exception as e:
        print(f"에이전트 초기화 실패: {e}")

def test_weakness_analysis():
    """취약점 분석 기능 테스트"""
    try:
        agent = InfoProcessingExamAgent()
        
        # test_sample 폴더에서 분석 파일 선택
        test_sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_sample")
        
        if not os.path.exists(test_sample_dir):
            print(f"❌ test_sample 폴더를 찾을 수 없습니다: {test_sample_dir}")
            return
        
        # 폴더 내 JSON 파일 목록 가져오기
        json_files = [f for f in os.listdir(test_sample_dir) if f.endswith('.json')]
        
        if not json_files:
            print(f"❌ {test_sample_dir} 폴더에 JSON 파일이 없습니다.")
            return
        
        print(f"\n📁 {test_sample_dir} 폴더의 분석 파일 목록:")
        for i, filename in enumerate(json_files, 1):
            file_path = os.path.join(test_sample_dir, filename)
            file_size = os.path.getsize(file_path) / 1024
            print(f"{i}. {filename} ({file_size:.1f} KB)")
        
        # 사용자가 파일 선택
        while True:
            try:
                file_choice = input(f"\n분석할 파일 번호를 선택하세요 (1-{len(json_files)}): ").strip()
                file_index = int(file_choice) - 1
                
                if 0 <= file_index < len(json_files):
                    selected_filename = json_files[file_index]
                    analysis_file_path = os.path.join(test_sample_dir, selected_filename)
                    break
                else:
                    print(f"1-{len(json_files)} 사이의 숫자를 입력해주세요.")
            except ValueError:
                print("유효한 숫자를 입력해주세요.")
        
        print(f"\n📁 선택된 분석 파일: {selected_filename}")
        print(f"📁 파일 경로: {analysis_file_path}")
        
        # 취약점 분석 테스트
        print("🧠 JSON 파일에서 취약점 개념 추출 중...")
        weakness_concepts = agent.weakness_analyzer.extract_weakness_concepts_from_analysis(
            agent.weakness_analyzer.load_analysis_from_file(analysis_file_path)
        )
        
        if weakness_concepts:
            print(f"✅ 취약점 개념 추출 완료!")
            print(f"🧠 추출된 취약점 개념: {weakness_concepts}")
            
            print("\n🔧 추출된 취약점 기반 맞춤 문제 생성 중...")
            result = agent.weakness_quiz_generator.generate_weakness_quiz(
                input_data={
                    "analysis_file_path": analysis_file_path,
                    "target_count": 5
                },
                difficulty="중급"
            )
            
            if "error" not in result:
                print(f"✅ 문제 생성 완료!")
                print(f"📝 생성된 문제 수: {result.get('quiz_count', 0)}")
                print(f"🎯 취약점 개념 수: {result.get('generation_summary', {}).get('analyzed_concepts', 0)}")
                
                # 결과를 weakness 폴더에 저장
                output_file = agent.weakness_quiz_generator.save_weakness_quiz_result(result)
                print(f"💾 결과가 {output_file}에 저장되었습니다.")
            else:
                print(f"❌ 문제 생성 실패: {result['error']}")
        else:
            print("❌ 취약점 개념을 추출할 수 없습니다.")
            
    except Exception as e:
        print(f"❌ 취약점 분석 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
