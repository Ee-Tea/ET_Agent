#!/usr/bin/env python3
"""
골든 데이터셋 생성 스크립트
Ragas를 활용하여 teacher RAG 시스템을 위한 고품질 평가 데이터셋을 생성합니다.
"""

import os
import sys
import json
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# Ragas 및 LangChain 관련 import
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import EvaluationDataset

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_core.documents import Document
import openai

class GoldenDatasetGenerator:
    """골든 데이터셋 생성기"""
    
    def __init__(self, 
                 llm_model: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-small",
                 testset_size: int = 10):
        """
        Args:
            llm_model: 사용할 LLM 모델명
            embedding_model: 사용할 임베딩 모델명
            testset_size: 생성할 테스트셋 크기
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.testset_size = testset_size
        
        # OpenAI 클라이언트 초기화
        self.openai_client = openai.OpenAI()
        
        # LLM 및 임베딩 설정
        self.generator_llm = LangchainLLMWrapper(
            ChatOpenAI(model=llm_model, temperature=0.1)
        )
        self.generator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=embedding_model)
        )
        
        # 테스트셋 생성기 초기화
        self.generator = TestsetGenerator(
            llm=self.generator_llm,
            embedding_model=self.generator_embeddings
        )
        
        # 출력 디렉토리 설정
        self.output_dir = Path(__file__).parent
        self.golden_dir = self.output_dir / "datasets"
        self.golden_dir.mkdir(exist_ok=True)
        
    def load_exam_documents(self) -> List[Document]:
        """기존 시험 데이터를 문서로 로드"""
        print("📚 시험 데이터 로딩 중...")
        
        documents = []
        exam_dir = Path(__file__).parent.parent / "exam"
        
        # JSON 파일들 로드
        json_files = [
            "2024년3회_기사필기_전체문제.json",
            "2025년1회_기사필기_전체문제.json", 
            "2025년2회_기사필기_전체문제.json"
        ]
        
        for json_file in json_files:
            file_path = exam_dir / json_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 각 문제를 개별 문서로 변환
                    for i, question_data in enumerate(data.get('questions', [])):
                        # 문제 텍스트 생성
                        question_text = f"""
문제: {question_data.get('question', '')}
선택지:
1. {question_data.get('options', [''])[0] if len(question_data.get('options', [])) > 0 else ''}
2. {question_data.get('options', [''])[1] if len(question_data.get('options', [])) > 1 else ''}
3. {question_data.get('options', [''])[2] if len(question_data.get('options', [])) > 2 else ''}
4. {question_data.get('options', [''])[3] if len(question_data.get('options', [])) > 3 else ''}
정답: {question_data.get('answer', '')}
과목: {question_data.get('subject', '')}
설명: {question_data.get('explanation', '')}
"""
                        
                        doc = Document(
                            page_content=question_text.strip(),
                            metadata={
                                'source': json_file,
                                'question_id': i + 1,
                                'subject': question_data.get('subject', ''),
                                'answer': question_data.get('answer', ''),
                                'exam_title': data.get('exam_title', '')
                            }
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    print(f"⚠️ {json_file} 로딩 실패: {e}")
                    continue
        
        print(f"✅ {len(documents)}개의 시험 문제 문서를 로드했습니다.")
        return documents
    
    def load_parsed_exam_documents(self) -> List[Document]:
        """파싱된 시험 데이터를 문서로 로드"""
        print("📚 파싱된 시험 데이터 로딩 중...")
        
        documents = []
        parsed_dir = Path(__file__).parent.parent / "exam" / "parsed_exam_json"
        
        if not parsed_dir.exists():
            print("⚠️ 파싱된 시험 데이터 디렉토리가 존재하지 않습니다.")
            return documents
        
        for json_file in parsed_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 각 문제를 개별 문서로 변환
                for i, question_data in enumerate(data.get('questions', [])):
                    question_text = f"""
문제: {question_data.get('question', '')}
선택지:
1. {question_data.get('options', [''])[0] if len(question_data.get('options', [])) > 0 else ''}
2. {question_data.get('options', [''])[1] if len(question_data.get('options', [])) > 1 else ''}
3. {question_data.get('options', [''])[2] if len(question_data.get('options', [])) > 2 else ''}
4. {question_data.get('options', [''])[3] if len(question_data.get('options', [])) > 3 else ''}
정답: {question_data.get('answer', '')}
과목: {question_data.get('subject', '')}
설명: {question_data.get('explanation', '')}
"""
                    
                    doc = Document(
                        page_content=question_text.strip(),
                        metadata={
                            'source': json_file.name,
                            'question_id': i + 1,
                            'subject': question_data.get('subject', ''),
                            'answer': question_data.get('answer', ''),
                            'exam_title': data.get('exam_title', '')
                        }
                    )
                    documents.append(doc)
                    
            except Exception as e:
                print(f"⚠️ {json_file.name} 로딩 실패: {e}")
                continue
        
        print(f"✅ {len(documents)}개의 파싱된 시험 문제 문서를 로드했습니다.")
        return documents
    
    async def adapt_prompts_for_korean(self):
        """한국어에 맞게 프롬프트 적응"""
        print("🇰🇷 한국어 프롬프트 적응 중...")
        
        qd = default_query_distribution(self.generator_llm)
        print(f"   - 총 {len(qd)}개의 합성기 발견")
        
        for i, (synth, _) in enumerate(qd):
            try:
                print(f"   - {i+1}/{len(qd)}: {synth.__class__.__name__} 적응 중...")
                prompts = await synth.adapt_prompts("korean", llm=self.generator_llm)
                synth.set_prompts(**prompts)
                print(f"   ✅ {synth.__class__.__name__} 프롬프트 적응 완료")
            except Exception as e:
                print(f"   ⚠️ 프롬프트 적응 실패: {e}")
                continue
        
        print("✅ 모든 프롬프트 적응 완료")
        return qd
    
    async def generate_testset(self, documents: List[Document]) -> EvaluationDataset:
        """테스트셋 생성"""
        print(f"🎯 {self.testset_size}개 크기의 테스트셋 생성 중...")
        print(f"   - 입력 문서 수: {len(documents)}개")
        
        try:
            # 한국어 프롬프트 적응
            print("\n1️⃣ 한국어 프롬프트 적응")
            qd = await self.adapt_prompts_for_korean()
            
            # 테스트셋 생성
            print("\n2️⃣ 테스트셋 생성 시작")
            print("   - 이 과정은 시간이 오래 걸릴 수 있습니다...")
            print("   - OpenAI API 호출로 인한 비용이 발생할 수 있습니다.")
            
            dataset = self.generator.generate_with_langchain_docs(
                documents=documents,
                testset_size=self.testset_size,
                query_distribution=qd
            )
            
            print(f"\n✅ 테스트셋 생성 완료: {len(dataset)}개 샘플")
            return dataset
            
        except Exception as e:
            print(f"❌ 테스트셋 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_testset(self, dataset: EvaluationDataset, filename: str = None):
        """테스트셋을 파일로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"testset_{timestamp}.json"
        
        filepath = self.golden_dir / filename
        
        try:
            # JSON 형태로 저장
            df = dataset.to_pandas()
            df.to_json(filepath, orient='records', force_ascii=False, indent=2)
            
            print(f"💾 테스트셋 저장 완료: {filepath}")
            
            # 통계 정보 출력
            print(f"📊 테스트셋 통계:")
            print(f"   - 총 샘플 수: {len(df)}")
            print(f"   - 컬럼: {list(df.columns)}")
            
            return filepath
            
        except Exception as e:
            print(f"❌ 테스트셋 저장 실패: {e}")
            raise
    
    def analyze_testset(self, dataset: EvaluationDataset):
        """테스트셋 분석"""
        print("📈 테스트셋 분석 중...")
        
        df = dataset.to_pandas()
        
        print(f"\n📊 테스트셋 상세 분석:")
        print(f"   - 총 샘플 수: {len(df)}")
        
        if 'question' in df.columns:
            print(f"   - 평균 질문 길이: {df['question'].str.len().mean():.1f}자")
        
        if 'reference' in df.columns:
            print(f"   - 평균 답변 길이: {df['reference'].str.len().mean():.1f}자")
        
        # 질문 유형 분석 (간단한 키워드 기반)
        if 'question' in df.columns:
            question_types = {
                '정의': df['question'].str.contains('정의|의미|개념|이란|무엇').sum(),
                '방법': df['question'].str.contains('방법|절차|과정|단계').sum(),
                '비교': df['question'].str.contains('비교|차이|다른|틀린').sum(),
                '수치': df['question'].str.contains('몇|개수|크기|길이|시간').sum(),
                '원인': df['question'].str.contains('원인|이유|왜|때문').sum()
            }
            
            print(f"   - 질문 유형 분포:")
            for qtype, count in question_types.items():
                if count > 0:
                    print(f"     * {qtype}: {count}개 ({count/len(df)*100:.1f}%)")

async def main():
    """메인 실행 함수"""
    print("🚀 골든 데이터셋 생성 시작")
    print("=" * 50)
    
    try:
        # 생성기 초기화
        generator = GoldenDatasetGenerator(
            llm_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small", 
            testset_size=10  # 테스트용으로 작은 크기
        )
        
        # 문서 로드
        print("\n1️⃣ 문서 로딩")
        documents = generator.load_exam_documents()
        parsed_docs = generator.load_parsed_exam_documents()
        
        # 모든 문서 합치기
        all_documents = documents + parsed_docs
        
        if not all_documents:
            print("❌ 로드된 문서가 없습니다. 종료합니다.")
            return
        
        print(f"📚 총 {len(all_documents)}개 문서 로드 완료")
        
        # 테스트셋 생성
        print("\n2️⃣ 테스트셋 생성")
        dataset = await generator.generate_testset(all_documents)
        
        # 테스트셋 분석
        print("\n3️⃣ 테스트셋 분석")
        generator.analyze_testset(dataset)
        
        # 테스트셋 저장
        print("\n4️⃣ 테스트셋 저장")
        filepath = generator.save_testset(dataset)
        
        print(f"\n✅ 골든 데이터셋 생성 완료!")
        print(f"📁 저장 위치: {filepath}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
