#!/usr/bin/env python3
"""
RAG 파이프라인 평가 스크립트
생성된 골든 데이터셋을 사용하여 teacher RAG 시스템의 성능을 평가합니다.
"""

import os
import sys
import json
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# Ragas 및 LangChain 관련 import
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
import openai

# Teacher 시스템 관련 import
from teacher.teacher import Teacher
from teacher.agents.retrieve.retrieve_agent import retrieve_agent
from teacher.agents.solution.solution_agent import SolutionAgent

class RAGPipelineEvaluator:
    """RAG 파이프라인 평가기"""
    
    def __init__(self, 
                 llm_model: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-small"):
        """
        Args:
            llm_model: 사용할 LLM 모델명
            embedding_model: 사용할 임베딩 모델명
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # OpenAI 클라이언트 초기화
        self.openai_client = openai.OpenAI()
        
        # LLM 및 임베딩 설정
        self.llm = LangchainLLMWrapper(
            ChatOpenAI(model=llm_model, temperature=0.1)
        )
        self.embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=embedding_model)
        )
        
        # Teacher 시스템 초기화
        self.teacher = Teacher()
        self.retrieve_agent = retrieve_agent
        self.solution_agent = SolutionAgent()
        
        # 출력 디렉토리 설정
        self.output_dir = Path(__file__).parent
        self.results_dir = self.output_dir / "evaluation_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 평가 임계값 설정
        self.thresholds = {
            'faithfulness': 0.90,
            'answer_relevancy': 0.85,
            'context_precision': 0.70,
            'context_recall': 0.70
        }
    
    def load_golden_dataset(self, dataset_path: str) -> EvaluationDataset:
        """골든 데이터셋 로드"""
        print(f"📚 골든 데이터셋 로딩: {dataset_path}")
        
        try:
            # JSON 파일에서 데이터 로드
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # EvaluationDataset 객체로 변환
            dataset = EvaluationDataset.from_pandas(pd.DataFrame(data))
            
            print(f"✅ {len(dataset)}개 샘플 로드 완료")
            return dataset
            
        except Exception as e:
            print(f"❌ 데이터셋 로딩 실패: {e}")
            raise
    
    async def generate_rag_responses(self, dataset: EvaluationDataset) -> List[Dict[str, Any]]:
        """RAG 파이프라인을 통해 응답 생성"""
        print("🤖 RAG 파이프라인으로 응답 생성 중...")
        
        evaluation_data = []
        
        for i, sample in enumerate(dataset):
            print(f"   처리 중: {i+1}/{len(dataset)}")
            
            try:
                question = sample.question
                
                # 1. Retrieve 단계 - 관련 문서 검색
                print(f"     🔍 검색 중: {question[:50]}...")
                contexts = await self._retrieve_contexts(question)
                
                # 2. Generate 단계 - LLM으로 응답 생성
                print(f"     ✍️ 응답 생성 중...")
                response = await self._generate_response(question, contexts)
                
                # 평가용 데이터 구성
                eval_sample = {
                    'question': question,
                    'contexts': contexts,
                    'response': response,
                    'reference': sample.reference,
                    'ground_truths': [sample.reference]  # Ragas 형식에 맞춤
                }
                
                evaluation_data.append(eval_sample)
                
                print(f"     ✅ 완료")
                
            except Exception as e:
                print(f"     ❌ 오류: {e}")
                continue
        
        print(f"✅ {len(evaluation_data)}개 응답 생성 완료")
        return evaluation_data
    
    async def _retrieve_contexts(self, question: str, top_k: int = 5) -> List[str]:
        """질문에 대한 관련 컨텍스트 검색"""
        try:
            # retrieve_agent를 사용하여 관련 문서 검색
            # 실제 구현에서는 retrieve_agent의 인터페이스에 맞게 조정 필요
            contexts = await self.retrieve_agent.retrieve_documents(
                query=question,
                top_k=top_k
            )
            
            # Document 객체에서 텍스트 추출
            if contexts and hasattr(contexts[0], 'page_content'):
                return [ctx.page_content for ctx in contexts]
            elif contexts and isinstance(contexts[0], str):
                return contexts
            else:
                return [str(ctx) for ctx in contexts]
                
        except Exception as e:
            print(f"     ⚠️ 검색 실패: {e}")
            return [f"검색 실패: {e}"]
    
    async def _generate_response(self, question: str, contexts: List[str]) -> str:
        """컨텍스트를 바탕으로 응답 생성"""
        try:
            # 컨텍스트를 하나의 텍스트로 결합
            context_text = "\n\n".join(contexts)
            
            # 프롬프트 구성
            prompt = f"""
다음 컨텍스트를 바탕으로 질문에 답해주세요.

컨텍스트:
{context_text}

질문: {question}

답변:
"""
            
            # LLM으로 응답 생성
            response = self.llm.generate(prompt)
            
            return response
            
        except Exception as e:
            print(f"     ⚠️ 응답 생성 실패: {e}")
            return f"응답 생성 실패: {e}"
    
    async def evaluate_pipeline(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """RAG 파이프라인 평가 실행"""
        print("📊 RAG 파이프라인 평가 실행 중...")
        
        try:
            # EvaluationDataset 객체로 변환
            eval_dataset = EvaluationDataset.from_dict({
                'question': [item['question'] for item in evaluation_data],
                'contexts': [item['contexts'] for item in evaluation_data],
                'response': [item['response'] for item in evaluation_data],
                'ground_truths': [item['ground_truths'] for item in evaluation_data]
            })
            
            # 평가 메트릭 실행
            result = evaluate(
                eval_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy, 
                    context_precision,
                    context_recall
                ],
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            print("✅ 평가 완료")
            return result
            
        except Exception as e:
            print(f"❌ 평가 실패: {e}")
            raise
    
    def analyze_results(self, result, evaluation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """평가 결과 분석"""
        print("📈 평가 결과 분석 중...")
        
        # 결과를 DataFrame으로 변환
        df = result.to_pandas()
        
        # 기본 통계
        stats = {
            'total_samples': len(df),
            'metrics': {
                'faithfulness': {
                    'mean': df['faithfulness'].mean(),
                    'std': df['faithfulness'].std(),
                    'min': df['faithfulness'].min(),
                    'max': df['faithfulness'].max()
                },
                'answer_relevancy': {
                    'mean': df['answer_relevancy'].mean(),
                    'std': df['answer_relevancy'].std(),
                    'min': df['answer_relevancy'].min(),
                    'max': df['answer_relevancy'].max()
                },
                'context_precision': {
                    'mean': df['context_precision'].mean(),
                    'std': df['context_precision'].std(),
                    'min': df['context_precision'].min(),
                    'max': df['context_precision'].max()
                },
                'context_recall': {
                    'mean': df['context_recall'].mean(),
                    'std': df['context_recall'].std(),
                    'min': df['context_recall'].min(),
                    'max': df['context_recall'].max()
                }
            }
        }
        
        # 임계값 기준 통과율 계산
        passed_samples = df[
            (df['faithfulness'] >= self.thresholds['faithfulness']) &
            (df['answer_relevancy'] >= self.thresholds['answer_relevancy']) &
            (df['context_precision'] >= self.thresholds['context_precision']) &
            (df['context_recall'] >= self.thresholds['context_recall'])
        ]
        
        stats['pass_rate'] = len(passed_samples) / len(df)
        stats['passed_samples'] = len(passed_samples)
        
        # 각 메트릭별 통과율
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
            threshold = self.thresholds[metric]
            passed = len(df[df[metric] >= threshold])
            stats['metrics'][metric]['pass_rate'] = passed / len(df)
        
        return stats
    
    def save_results(self, result, stats: Dict[str, Any], evaluation_data: List[Dict[str, Any]]):
        """평가 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 상세 결과 저장 (CSV)
        df = result.to_pandas()
        detailed_file = self.results_dir / f"evaluation_detailed_{timestamp}.csv"
        df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
        
        # 2. 통계 요약 저장 (JSON)
        summary_file = self.results_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 3. 원본 평가 데이터 저장 (JSON)
        raw_file = self.results_dir / f"evaluation_raw_{timestamp}.json"
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 결과 저장 완료:")
        print(f"   - 상세 결과: {detailed_file}")
        print(f"   - 요약 통계: {summary_file}")
        print(f"   - 원본 데이터: {raw_file}")
        
        return detailed_file, summary_file, raw_file
    
    def print_summary(self, stats: Dict[str, Any]):
        """평가 결과 요약 출력"""
        print("\n" + "="*60)
        print("📊 RAG 파이프라인 평가 결과 요약")
        print("="*60)
        
        print(f"📈 전체 통계:")
        print(f"   - 총 샘플 수: {stats['total_samples']}")
        print(f"   - 통과 샘플 수: {stats['passed_samples']}")
        print(f"   - 전체 통과율: {stats['pass_rate']:.2%}")
        
        print(f"\n📊 메트릭별 성능:")
        for metric, values in stats['metrics'].items():
            print(f"   - {metric}:")
            print(f"     * 평균: {values['mean']:.3f}")
            print(f"     * 표준편차: {values['std']:.3f}")
            print(f"     * 범위: {values['min']:.3f} ~ {values['max']:.3f}")
            print(f"     * 통과율: {values['pass_rate']:.2%}")
        
        print(f"\n🎯 임계값 기준:")
        for metric, threshold in self.thresholds.items():
            print(f"   - {metric}: ≥ {threshold}")
        
        # 성능 등급 평가
        overall_score = stats['pass_rate']
        if overall_score >= 0.9:
            grade = "A+ (우수)"
        elif overall_score >= 0.8:
            grade = "A (양호)"
        elif overall_score >= 0.7:
            grade = "B (보통)"
        elif overall_score >= 0.6:
            grade = "C (미흡)"
        else:
            grade = "D (부족)"
        
        print(f"\n🏆 전체 성능 등급: {grade} ({overall_score:.2%})")

async def main():
    """메인 실행 함수"""
    print("🚀 RAG 파이프라인 평가 시작")
    print("=" * 50)
    
    try:
        # 평가기 초기화
        evaluator = RAGPipelineEvaluator(
            llm_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small"
        )
        
        # 골든 데이터셋 로드
        print("\n1️⃣ 골든 데이터셋 로딩")
        datasets_dir = evaluator.output_dir / "datasets"
        
        # 가장 최근 데이터셋 파일 찾기
        dataset_files = list(datasets_dir.glob("testset_*.json"))
        if not dataset_files:
            print("❌ 데이터셋 파일을 찾을 수 없습니다.")
            print("   먼저 generate_golden_dataset.py를 실행해주세요.")
            return
        
        latest_dataset = max(dataset_files, key=lambda x: x.stat().st_mtime)
        dataset = evaluator.load_golden_dataset(str(latest_dataset))
        
        # RAG 응답 생성
        print("\n2️⃣ RAG 응답 생성")
        evaluation_data = await evaluator.generate_rag_responses(dataset)
        
        if not evaluation_data:
            print("❌ 생성된 응답이 없습니다.")
            return
        
        # 파이프라인 평가
        print("\n3️⃣ 파이프라인 평가")
        result = await evaluator.evaluate_pipeline(evaluation_data)
        
        # 결과 분석
        print("\n4️⃣ 결과 분석")
        stats = evaluator.analyze_results(result, evaluation_data)
        
        # 결과 저장
        print("\n5️⃣ 결과 저장")
        evaluator.save_results(result, stats, evaluation_data)
        
        # 요약 출력
        evaluator.print_summary(stats)
        
        print(f"\n✅ RAG 파이프라인 평가 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
