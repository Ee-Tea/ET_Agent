# -*- coding: utf-8 -*-
"""
PDF → RAGAS 평가 데이터셋 생성기
- PDF 디렉토리에서 문서 로드
- 텍스트 청크 분할 후 LangChain Document 변환
- ragas TestsetGenerator 기반으로 QA 생성
- 최종적으로 question, ground_truth, contexts 필드 저장
"""

import os
import sys
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.persona import Persona
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ===================== 환경설정 =====================
load_dotenv()

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ 오류: OPENAI_API_KEY 환경 변수가 없습니다.")
    sys.exit(1)

TARGET_QUESTIONS = 50   # 생성할 질문 수
PDF_DIR = "./farmer/disaster/pdfs"    # PDF 폴더 경로
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ===================== PDF Loader =====================
def load_pdfs_as_documents(pdf_dir: str):
    """PDF 디렉토리 → Document 리스트"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents = []

    pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_paths:
        print(f"⚠️ PDF 파일 없음: {pdf_dir}")
        return documents

    print(f"📥 PDF {len(pdf_paths)}개 로드 중...")

    for pdf_path in tqdm(pdf_paths, desc="PDF 처리"):
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            for p in pages:
                chunks = splitter.split_text(p.page_content)
                for chunk in chunks:
                    documents.append(Document(page_content=chunk, metadata={"source": os.path.basename(pdf_path)}))
        except Exception as e:
            print(f"⚠️ {pdf_path} 처리 실패: {e}")

    print(f"📚 총 {len(documents)}개 청크 문서 생성됨")
    return documents

# ===================== 메인 =====================
async def main():
    documents = load_pdfs_as_documents(PDF_DIR)
    if not documents:
        print("❌ 문서 없음. 종료")
        return

    # 모델 초기화
    print("⚙️ 모델 초기화...")
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

    # 🔥 OpenAI 임베딩 사용
    generator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-large")
    )

    # Persona & transforms
    personas = [
        Persona(
            name="Farmer",
            role_description="농업 재해에 대한 과거 사례를 참고하여 농업 재해에 대한 질문과 정답을 만드는 농부."
            "질문은 작물의 재해 상황에 대한 대비에 가장 초점을 맞춰줘 예를 들면 토마토를 키우는데 내일 태풍이 온다면 어떻게 대비해야 하는지 이런 질문들로"
            "정답은 문서 내용에만 근거하여 작성해줘"
            "너무 중복된 질문들은 배제해줘"
            "작년 사례 관련 질문들도 몇 개 정도 넣어줘"
        )
    ]
    transforms = [HeadlineSplitter(), NERExtractor()]

    # TestsetGenerator
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas
    )

    # Query 분포
    distribution = [(SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0)]
    for query, _ in distribution:
        prompts = await query.adapt_prompts("korean", llm=generator_llm)
        query.set_prompts(**prompts)

    # 데이터셋 생성
    print(f"🚀 {TARGET_QUESTIONS}개 질문 생성 시작...")
    dataset = generator.generate_with_langchain_docs(
        documents,
        testset_size=TARGET_QUESTIONS,
        transforms=transforms,
        query_distribution=distribution
    )

    eval_dataset = dataset.to_evaluation_dataset()

    # 출력
    results = []
    for i, sample in enumerate(eval_dataset[:TARGET_QUESTIONS]):
        print(f"\n질문 {i+1}")
        print("  Question:", sample.user_input)
        print("  Ground Truth:", sample.reference)
        print("  Contexts:", sample.reference_contexts)
        results.append({
            "question": sample.user_input,
            "ground_truth": sample.reference,
            "contexts": sample.reference_contexts
        })

    # 저장
    df = pd.DataFrame(results)
    df.to_csv("pdf_golden_dataset_openai.csv", index=False, encoding="utf-8-sig")
    with open("pdf_golden_dataset_openai.jsonl", "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n✅ pdf_golden_dataset_openai.csv / .jsonl 저장 완료 ({len(results)}건)")

if __name__ == "__main__":
    asyncio.run(main())
