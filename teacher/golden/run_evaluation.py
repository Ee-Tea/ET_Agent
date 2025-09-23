#!/usr/bin/env python3
# uv run python teacher/golden/run_evaluation.py
"""
골든 데이터셋 생성 및 RAG 파이프라인 평가 통합 실행 스크립트
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from generate_golden_dataset import GoldenDatasetGenerator
from evaluate_rag_pipeline import RAGPipelineEvaluator

class EvaluationRunner:
    """평가 실행기"""
    
    def __init__(self):
        self.output_dir = Path(__file__).parent
        self.datasets_dir = self.output_dir / "datasets"
        self.results_dir = self.output_dir / "evaluation_results"
        
        # 디렉토리 생성
        self.datasets_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    async def run_full_evaluation(self, 
                                 testset_size: int = 10,
                                 llm_model: str = "gpt-4o-mini",
                                 embedding_model: str = "text-embedding-3-small"):
        """전체 평가 프로세스 실행"""
        print("🚀 전체 평가 프로세스 시작")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # 1단계: 골든 데이터셋 생성
            print("\n📚 1단계: 골든 데이터셋 생성")
            print("-" * 40)
            
            generator = GoldenDatasetGenerator(
                llm_model=llm_model,
                embedding_model=embedding_model,
                testset_size=testset_size
            )
            
            # 문서 로드
            documents = generator.load_exam_documents()
            parsed_docs = generator.load_parsed_exam_documents()
            all_documents = documents + parsed_docs
            
            if not all_documents:
                print("❌ 로드된 문서가 없습니다.")
                return
            
            # 테스트셋 생성
            dataset = await generator.generate_testset(all_documents)
            
            # 테스트셋 분석 및 저장
            generator.analyze_testset(dataset)
            dataset_file = generator.save_testset(dataset)
            
            print(f"✅ 1단계 완료: {dataset_file}")
            
            # 2단계: RAG 파이프라인 평가
            print("\n🤖 2단계: RAG 파이프라인 평가")
            print("-" * 40)
            
            evaluator = RAGPipelineEvaluator(
                llm_model=llm_model,
                embedding_model=embedding_model
            )
            
            # 데이터셋 로드
            dataset = evaluator.load_golden_dataset(str(dataset_file))
            
            # RAG 응답 생성
            evaluation_data = await evaluator.generate_rag_responses(dataset)
            
            if not evaluation_data:
                print("❌ 생성된 응답이 없습니다.")
                return
            
            # 파이프라인 평가
            result = await evaluator.evaluate_pipeline(evaluation_data)
            
            # 결과 분석 및 저장
            stats = evaluator.analyze_results(result, evaluation_data)
            evaluator.save_results(result, stats, evaluation_data)
            
            # 요약 출력
            evaluator.print_summary(stats)
            
            print(f"✅ 2단계 완료")
            
            # 전체 실행 시간 계산
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n🎉 전체 평가 프로세스 완료!")
            print(f"⏱️ 총 실행 시간: {duration}")
            print(f"📁 결과 위치: {self.results_dir}")
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    async def run_dataset_generation_only(self, 
                                        testset_size: int = 10,
                                        llm_model: str = "gpt-4o-mini",
                                        embedding_model: str = "text-embedding-3-small"):
        """데이터셋 생성만 실행"""
        print("📚 골든 데이터셋 생성만 실행")
        print("=" * 40)
        
        try:
            generator = GoldenDatasetGenerator(
                llm_model=llm_model,
                embedding_model=embedding_model,
                testset_size=testset_size
            )
            
            # 문서 로드
            documents = generator.load_exam_documents()
            parsed_docs = generator.load_parsed_exam_documents()
            all_documents = documents + parsed_docs
            
            if not all_documents:
                print("❌ 로드된 문서가 없습니다.")
                return
            
            # 테스트셋 생성
            dataset = await generator.generate_testset(all_documents)
            
            # 테스트셋 분석 및 저장
            generator.analyze_testset(dataset)
            dataset_file = generator.save_testset(dataset)
            
            print(f"✅ 데이터셋 생성 완료: {dataset_file}")
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    async def run_evaluation_only(self, 
                                llm_model: str = "gpt-4o-mini",
                                embedding_model: str = "text-embedding-3-small"):
        """평가만 실행"""
        print("🤖 RAG 파이프라인 평가만 실행")
        print("=" * 40)
        
        try:
            evaluator = RAGPipelineEvaluator(
                llm_model=llm_model,
                embedding_model=embedding_model
            )
            
            # 가장 최근 데이터셋 파일 찾기
            dataset_files = list(self.datasets_dir.glob("testset_*.json"))
            if not dataset_files:
                print("❌ 데이터셋 파일을 찾을 수 없습니다.")
                print("   먼저 데이터셋 생성을 실행해주세요.")
                return
            
            latest_dataset = max(dataset_files, key=lambda x: x.stat().st_mtime)
            print(f"📚 사용할 데이터셋: {latest_dataset.name}")
            
            # 데이터셋 로드
            dataset = evaluator.load_golden_dataset(str(latest_dataset))
            
            # RAG 응답 생성
            evaluation_data = await evaluator.generate_rag_responses(dataset)
            
            if not evaluation_data:
                print("❌ 생성된 응답이 없습니다.")
                return
            
            # 파이프라인 평가
            result = await evaluator.evaluate_pipeline(evaluation_data)
            
            # 결과 분석 및 저장
            stats = evaluator.analyze_results(result, evaluation_data)
            evaluator.save_results(result, stats, evaluation_data)
            
            # 요약 출력
            evaluator.print_summary(stats)
            
            print(f"✅ 평가 완료")
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="골든 데이터셋 생성 및 RAG 파이프라인 평가")
    
    parser.add_argument(
        "--mode", 
        choices=["full", "dataset", "evaluation"],
        default="full",
        help="실행 모드: full(전체), dataset(데이터셋 생성만), evaluation(평가만)"
    )
    
    parser.add_argument(
        "--testset-size",
        type=int,
        default=10,
        help="생성할 테스트셋 크기 (기본값: 10)"
    )
    
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="사용할 LLM 모델 (기본값: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--embedding-model", 
        default="text-embedding-3-small",
        help="사용할 임베딩 모델 (기본값: text-embedding-3-small)"
    )
    
    args = parser.parse_args()
    
    # 실행기 초기화
    runner = EvaluationRunner()
    
    # 모드에 따른 실행
    if args.mode == "full":
        asyncio.run(runner.run_full_evaluation(
            testset_size=args.testset_size,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model
        ))
    elif args.mode == "dataset":
        asyncio.run(runner.run_dataset_generation_only(
            testset_size=args.testset_size,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model
        ))
    elif args.mode == "evaluation":
        asyncio.run(runner.run_evaluation_only(
            llm_model=args.llm_model,
            embedding_model=args.embedding_model
        ))

if __name__ == "__main__":
    main()
