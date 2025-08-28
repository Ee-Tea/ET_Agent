#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS + Milvus 통합 테스트 파일
실제 Milvus 데이터로 문제 생성 후 RAGAS로 품질 검증
"""

import os
import sys
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# RAGAS 환경 변수 명시적 설정 (OpenAI API용)
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
os.environ.setdefault("OPENAI_LLM_MODEL", "gpt-3.5-turbo")

def test_ragas_installation():
    """RAGAS 설치 상태 확인"""
    try:
        import ragas
        print(f"✅ RAGAS 설치됨: {ragas.__version__}")
        return True
    except ImportError:
        print("❌ RAGAS가 설치되지 않음")
        print("설치 명령어: uv pip install ragas")
        return False

def test_milvus_connection():
    """Milvus 연결 테스트"""
    try:
        from pymilvus import connections, utility
        from langchain_milvus import Milvus
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Milvus 연결 설정
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        collection_name = "info_exam_chunks"  # 실제 존재하는 컬렉션명으로 수정
        
        print(f"🔗 Milvus 연결 시도: {host}:{port}")
        
        # 연결
        if "default" in connections.list_connections():
            connections.disconnect(alias="default")
        connections.connect(alias="default", host=host, port=port)
        
        # 컬렉션 존재 확인
        if not utility.has_collection(collection_name):
            print(f"❌ 컬렉션이 없습니다: {collection_name}")
            # 사용 가능한 컬렉션 목록 출력
            available_collections = utility.list_collections()
            print(f"사용 가능한 컬렉션: {available_collections}")
            return None, None
        
        print(f"✅ Milvus 연결 성공: {collection_name}")
        
        # 컬렉션 정보 출력
        from pymilvus import Collection
        collection = Collection(collection_name)
        print(f"📊 컬렉션 정보:")
        print(f"  - 엔티티 수: {collection.num_entities}")
        print(f"  - 스키마: {collection.schema}")
        
        # 임베딩 모델 초기화
        embeddings_model = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Milvus 벡터스토어 생성
        vectorstore = Milvus(
            embedding_function=embeddings_model,
            collection_name=collection_name,
            connection_args={"host": host, "port": port},
            index_params={"index_type": "AUTOINDEX", "metric_type": "IP"},
            search_params={"metric_type": "IP"},
        )
        
        return vectorstore, embeddings_model
        
    except Exception as e:
        print(f"❌ Milvus 연결 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_simple_search(vectorstore):
    """간단한 검색 테스트"""
    try:
        print("\n🔍 간단한 검색 테스트...")
        
        # 간단한 쿼리로 검색 테스트
        query = "소프트웨어 개발"
        documents = vectorstore.similarity_search(query, k=3)
        
        if documents:
            print(f"✅ 검색 성공! {len(documents)}개 문서 발견")
            for i, doc in enumerate(documents[:2]):  # 처음 2개만 출력
                print(f"  문서 {i+1}: {doc.page_content[:100]}...")
                print(f"  메타데이터: {doc.metadata}")
                print()
        else:
            print("❌ 검색 결과가 없습니다.")
            
        return True
        
    except Exception as e:
        print(f"❌ 검색 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_questions_from_milvus(vectorstore, subject_area: str, num_questions: int = 3):
    """Milvus에서 데이터를 가져와서 문제 생성"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import PromptTemplate
        
        # LLM 초기화 (OpenAI API 사용)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
            return [], ""
        
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.2,
                max_tokens=1024,
                base_url="https://api.openai.com/v1",
                api_key=openai_api_key
            )
            print("✅ OpenAI API LLM 초기화 완료 (GPT-3.5-turbo)")
        except Exception as e:
            print(f"❌ OpenAI API LLM 초기화 실패: {e}")
            return [], ""
        
        print(f"🚀 {subject_area} 과목 문제 생성 시작...")
        
        # Milvus에서 관련 문서 검색
        query = f"{subject_area} 관련 문제"
        documents = vectorstore.similarity_search(query, k=5)
        
        if not documents:
            print(f"❌ {subject_area} 관련 문서를 찾을 수 없습니다.")
            return [], ""
        
        # 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"문서 {i+1}: {doc.page_content}")
            if doc.metadata.get('options'):
                context_parts.append(f"보기: {doc.metadata['options']}")
            if doc.metadata.get('answer'):
                context_parts.append(f"정답: {doc.metadata['answer']}")
            if doc.metadata.get('explanation'):
                context_parts.append(f"해설: {doc.metadata['explanation']}")
        
        context = "\n\n".join(context_parts)
        print(f"📚 컨텍스트 구성 완료: {len(context)}자")
        
        # 문제 생성 프롬프트
        prompt_template = PromptTemplate(
            input_variables=["context", "subject_area", "num_questions"],
            template=(
                "당신은 정보처리기사 출제 전문가입니다. 아래 문서 내용을 바탕으로 {subject_area} 과목의 객관식 문제 {num_questions}개를 생성하세요.\n\n"
                "조건:\n"
                "• 보기에는 번호(1. 2. 3. 4.)를 붙이지 마십시오.\n"
                "• answer에는 정답의 '번호'만 문자열로 적으십시오. 예: \"2\"\n"
                "• 출력은 아래 JSON 형식만 포함하십시오. 다른 텍스트 금지.\n\n"
                "[문서 내용]\n{context}\n\n"
                "[응답 형식]\n"
                "{{\n"
                "  \"questions\": [\n"
                "    {{\n"
                "      \"question\": \"문제 내용을 여기에 작성\",\n"
                "      \"options\": [\"선택지1\", \"선택지2\", \"선택지3\", \"선택지4\"],\n"
                "      \"answer\": \"1\",\n"
                "      \"explanation\": \"정답에 대한 간단한 해설\"\n"
                "    }}\n"
                "  ]\n"
                "}}\n"
            )
        )
        
        prompt = prompt_template.format(
            context=context[:3000],  # 컨텍스트 길이 제한
            subject_area=subject_area,
            num_questions=num_questions
        )
        
        # LLM 호출
        print("🤖 LLM으로 문제 생성 중...")
        response = llm.invoke(prompt)
        response_content = getattr(response, "content", str(response))
        
        # JSON 파싱
        questions = parse_quiz_response(response_content, subject_area)
        print(f"✅ {len(questions)}개 문제 생성 완료")
        
        return questions, context
        
    except Exception as e:
        print(f"❌ 문제 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return [], ""

def parse_quiz_response(response: str, subject_area: str = "") -> List[Dict[str, Any]]:
    """LLM 응답에서 문제 파싱"""
    import re
    
    try:
        # JSON 블록 찾기
        json_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_block_match:
            json_str = json_block_match.group(1)
        else:
            # 일반 JSON 객체 찾기
            json_str_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response.strip(), re.DOTALL)
            if not json_str_match:
                print("❌ JSON 형식을 찾을 수 없습니다.")
                print(f"응답 내용: {response[:200]}...")
                return []
            json_str = json_str_match.group(0)
        
        # JSON 파싱
        data = json.loads(json_str)
        if "questions" not in data or not isinstance(data["questions"], list):
            print("❌ questions 필드가 없거나 리스트가 아닙니다.")
            return []
        
        questions = data.get("questions", [])
        
        # 각 문제에 과목 정보 추가
        for question in questions:
            if "subject" not in question:
                question["subject"] = subject_area
        
        return questions
        
    except Exception as e:
        print(f"❌ 응답 파싱 실패: {e}")
        print(f"응답 내용: {response[:200]}...")
        return []

def validate_questions_with_ragas(questions: List[Dict[str, Any]], context: str):
    """RAGAS를 사용하여 RAG 품질 검증"""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from ragas.llms import llm_factory
        
        if not questions:
            print("❌ 검증할 문제가 없습니다.")
            return None
        
        print(f"\n🔍 RAGAS로 {len(questions)}개 문제 RAG 품질 검증 시작...")
        
        # RAGAS LLM 설정 (OpenAI API 사용)
        try:
            # RAGAS LLM 설정 (OpenAI API 사용)
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
                return None
            
            # RAGAS에서 OpenAI API 사용을 위한 환경 변수 설정
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
            
            llm = llm_factory(
                model="gpt-3.5-turbo",
                base_url="https://api.openai.com/v1"
            )
            print("✅ RAGAS LLM 설정 완료 (OpenAI GPT-3.5-turbo)")
                
        except Exception as llm_error:
            print(f"⚠️ RAGAS LLM 설정 실패: {llm_error}")
            return None
        
        # RAGAS 평가 데이터 구성
        eval_data = {
            "question": [],
            "contexts": [],
            "answer": [],
            "ground_truth": []
        }
        
        for q in questions:
            question_text = q.get("question", "")[:200]
            answer_text = q.get("answer", "") + ": " + q.get("explanation", "")[:100]
            
            eval_data["question"].append(question_text)
            eval_data["contexts"].append([context[:500]])
            eval_data["answer"].append(answer_text)
            eval_data["ground_truth"].append(answer_text)
        
        dataset = Dataset.from_dict(eval_data)
        
        # RAGAS 평가 (RAG 품질 메트릭)
        print("📊 RAGAS RAG 품질 평가 중...")
        
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm
        )
        
        return results
        
    except Exception as e:
        print(f"❌ RAGAS 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_ragas_results(results, questions: List[Dict[str, Any]]):
    """RAGAS 결과 표시 (구조 점수 제거)"""
    if not results:
        return
    
    print("\n📊 RAGAS RAG 품질 평가 결과")
    print("=" * 60)
    
    try:
        # RAGAS 결과 표시
        if hasattr(results, '_scores_dict'):
            scores_dict = results._scores_dict
            
            faithfulness_scores = scores_dict.get('faithfulness', [])
            answer_relevancy_scores = scores_dict.get('answer_relevancy', [])
            context_precision_scores = scores_dict.get('context_precision', [])
            context_recall_scores = scores_dict.get('context_recall', [])
            
            print(f"Faithfulness: {faithfulness_scores}")
            print(f"Answer Relevancy: {answer_relevancy_scores}")
            print(f"Context Precision: {context_precision_scores}")
            print(f"Context Recall: {context_recall_scores}")
            
            # RAGAS 평균 점수
            all_ragas_scores = []
            for metric_scores in scores_dict.values():
                if isinstance(metric_scores, list):
                    valid_scores = [s for s in metric_scores if s is not None and not (isinstance(s, float) and str(s) == 'nan')]
                    all_ragas_scores.extend(valid_scores)
            
            if all_ragas_scores:
                avg_ragas_score = sum(all_ragas_scores) / len(all_ragas_scores)
                print(f"RAGAS 평균 점수: {avg_ragas_score:.4f}")
        
    except Exception as e:
        print(f"❌ RAGAS 결과 표시 중 오류: {e}")
        print(f"원본 결과: {results}")
    
    # 개별 문제 표시 (구조 점수 없이)
    print(f"\n📋 생성된 문제 ({len(questions)}개)")
    print("-" * 60)
    
    for i, question in enumerate(questions, 1):
        question_text = question.get("question", "")
        options = question.get("options", [])
        answer = question.get("answer", "")
        explanation = question.get("explanation", "")
        
        print(f"문제 {i}: {question_text}")
        
        # 보기 표시
        if options:
            for j, option in enumerate(options, 1):
                print(f"        {j}. {option}")
        print(f"정답: {answer}")
        print(f"해설: {explanation}")
        print()

def visualize_ragas_results(all_results: Dict[str, Any]):
    """RAGAS 결과를 시각화"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        
        # 영어 폰트 설정 (한글 폰트 문제 해결)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        print("\n📊 RAGAS Results Visualization Started...")
        
        # 데이터 준비
        subjects = []
        metrics_data = {
            'Faithfulness': [],
            'Answer Relevancy': [],
            'Context Precision': [],
            'Context Recall': []
        }
        
        for subject, result in all_results.items():
            if result["ragas_results"] and hasattr(result["ragas_results"], '_scores_dict'):
                subjects.append(subject)
                scores_dict = result["ragas_results"]._scores_dict
                
                # 각 메트릭의 평균 점수 계산
                for metric_name in metrics_data.keys():
                    metric_key = metric_name.lower().replace(' ', '_')
                    scores = scores_dict.get(metric_key, [])
                    if scores:
                        # nan 값 제거하고 평균 계산
                        valid_scores = [s for s in scores if s is not None and not (isinstance(s, float) and str(s) == 'nan')]
                        if valid_scores:
                            metrics_data[metric_name].append(np.mean(valid_scores))
                        else:
                            metrics_data[metric_name].append(0.0)
                    else:
                        metrics_data[metric_name].append(0.0)
        
        if not subjects:
            print("❌ No data to visualize.")
            return
        
        # 1. 메트릭별 비교 막대 그래프
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RAGAS Quality Assessment Results Visualization', fontsize=16, fontweight='bold')
        
        for i, (metric_name, scores) in enumerate(metrics_data.items()):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            bars = ax.bar(subjects, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title(f'{metric_name} Score', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.0)
            
            # 막대 위에 점수 표시
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # x축 레이블 회전
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('ragas_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("💾 Metrics comparison graph saved: ragas_metrics_comparison.png")
        
        # 2. 과목별 종합 점수 레이더 차트
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # 각 과목의 평균 점수 계산
        subject_scores = []
        for i, subject in enumerate(subjects):
            subject_avg = np.mean([metrics_data[metric][i] for metric in metrics_data.keys()])
            subject_scores.append(subject_avg)
        
        # 레이더 차트 데이터 준비
        angles = np.linspace(0, 2 * np.pi, len(subjects), endpoint=False).tolist()
        angles += angles[:1]  # 첫 번째 점을 마지막에 추가하여 닫힌 도형 만들기
        subject_scores += subject_scores[:1]
        
        ax.plot(angles, subject_scores, 'o-', linewidth=2, label='Subject Average Score')
        ax.fill(angles, subject_scores, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(subjects)
        ax.set_ylim(0, 1.0)
        ax.set_title('Subject-wise RAGAS Overall Score (Radar Chart)', fontweight='bold', pad=20)
        
        # 그리드 추가
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('ragas_subject_radar.png', dpi=300, bbox_inches='tight')
        print("💾 Subject radar chart saved: ragas_subject_radar.png")
        
        # 3. 히트맵
        # 데이터프레임 생성
        df = pd.DataFrame(metrics_data, index=subjects)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df, annot=True, cmap='RdYlGn', vmin=0, vmax=1, 
                   fmt='.3f', cbar_kws={'label': 'Score'})
        plt.title('RAGAS Metrics Heatmap', fontweight='bold', pad=20)
        plt.xlabel('Metrics')
        plt.ylabel('Subjects')
        plt.tight_layout()
        plt.savefig('ragas_heatmap.png', dpi=300, bbox_inches='tight')
        print("💾 Heatmap saved: ragas_heatmap.png")
        
        # 4. 요약 통계 테이블
        print("\n📈 RAGAS Results Summary Statistics:")
        print("=" * 80)
        
        summary_df = df.copy()
        summary_df['Average'] = summary_df.mean(axis=1)
        summary_df['Std Dev'] = summary_df.std(axis=1)
        
        print(summary_df.round(4))
        
        # 전체 평균 점수
        overall_avg = summary_df['Average'].mean()
        print(f"\n🎯 Overall Average Score: {overall_avg:.4f}")
        
        # 메트릭별 평균
        print("\n📊 Metrics Average Score:")
        for metric in metrics_data.keys():
            metric_avg = df[metric].mean()
            print(f"  {metric}: {metric_avg:.4f}")
        
        plt.show()
        print("\n✅ Visualization completed!")
        
    except ImportError as e:
        print(f"❌ Visualization library installation required: {e}")
        print("Install command: uv pip install matplotlib seaborn pandas")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()

def save_results_to_file(questions: List[Dict[str, Any]], ragas_results, context: str, subject_area: str):
    """결과를 JSON 파일로 저장"""
    try:
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ragas_test_{subject_area}_{timestamp}.json"
        
        # RAGAS 결과를 JSON 직렬화 가능한 형태로 변환
        ragas_data = {}
        if ragas_results and hasattr(ragas_results, '_scores_dict'):
            scores_dict = ragas_results._scores_dict
            ragas_data = {
                "faithfulness": scores_dict.get('faithfulness', []),
                "answer_relevancy": scores_dict.get('answer_relevancy', []),
                "context_precision": scores_dict.get('context_precision', []),
                "context_recall": scores_dict.get('context_recall', [])
            }
        
        result_data = {
            "test_info": {
                "subject_area": subject_area,
                "timestamp": timestamp,
                "total_questions": len(questions)
            },
            "context": context,
            "questions": questions,
            "ragas_results": ragas_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 결과 저장 완료: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        return None

def main():
    """메인 실행 함수"""
    print("🧪 RAGAS + Milvus 통합 테스트 시작")
    print("=" * 60)
    
    # 1. RAGAS 설치 확인
    if not test_ragas_installation():
        return
    
    # 2. Milvus 연결 테스트
    vectorstore, embeddings_model = test_milvus_connection()
    if not vectorstore:
        print("❌ Milvus 연결에 실패했습니다.")
        return
    
    # 3. 간단한 검색 테스트
    if not test_simple_search(vectorstore):
        print("❌ 검색 테스트에 실패했습니다.")
        return
    
    # 4. 테스트할 과목들
    test_subjects = ["Software Design", "Database Construction", "Programming Language Utilization"]
    
    all_results = {}
    
    for subject in test_subjects:
        print(f"\n{'='*20} {subject} 과목 테스트 {'='*20}")
        
        # 문제 생성
        questions, context = generate_questions_from_milvus(vectorstore, subject, num_questions=3)
        
        if not questions:
            print(f"❌ {subject} 과목 문제 생성 실패")
            continue
        
        # RAGAS 품질 검증
        ragas_results = validate_questions_with_ragas(questions, context)
        
        # 결과 표시
        display_ragas_results(ragas_results, questions)
        
        # 결과 저장
        filename = save_results_to_file(questions, ragas_results, context, subject)
        
        all_results[subject] = {
            "questions": questions,
            "ragas_results": ragas_results,
            "filename": filename
        }
    
    # 5. 전체 요약
    print(f"\n{'='*20} 전체 테스트 요약 {'='*20}")
    for subject, result in all_results.items():
        if result["ragas_results"]:
            # RAGAS 결과에서 점수 추출
            ragas_result = result["ragas_results"]
            
            # _scores_dict에서 점수 가져오기
            if hasattr(ragas_result, '_scores_dict'):
                scores_dict = ragas_result._scores_dict
                print(f"\n📊 {subject} RAGAS 평가 결과:")
                
                # 각 메트릭의 점수 확인
                faithfulness_scores = scores_dict.get('faithfulness', [])
                answer_relevancy_scores = scores_dict.get('answer_relevancy', [])
                context_precision_scores = scores_dict.get('context_precision', [])
                context_recall_scores = scores_dict.get('context_recall', [])
                
                print(f"Faithfulness: {faithfulness_scores}")
                print(f"Answer Relevancy: {answer_relevancy_scores}")
                print(f"Context Precision: {context_precision_scores}")
                print(f"Context Recall: {context_recall_scores}")
                
                # 평균 점수 계산 (nan 제외)
                all_scores = []
                for metric_scores in scores_dict.values():
                    if isinstance(metric_scores, list):
                        valid_scores = [s for s in metric_scores if s is not None and not (isinstance(s, float) and str(s) == 'nan')]
                        all_scores.extend(valid_scores)
                
                if all_scores:
                    avg_score = sum(all_scores) / len(all_scores)
                    print(f"평균 점수: {avg_score:.4f}")
                    print(f"문제 수: {len(result['questions'])}개")
        else:
            print(f"{subject}: 검증 실패")
    
    # 6. 시각화
    if all_results:
        visualize_ragas_results(all_results)
    
    print("\n✅ RAGAS + Milvus 통합 테스트 완료!")

if __name__ == "__main__":
    main()
