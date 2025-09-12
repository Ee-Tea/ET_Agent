# -*- coding: utf-8 -*-
"""
Milvus(concepts) 코퍼스 기반 RAGAS 골든셋 생성기 (ragas 0.3.3 호환)
- PDF/OCR/표 파싱 제거
- Milvus 'concepts' 컬렉션을 코퍼스로 활용
- ragas TestsetGenerator로 (question, ground_truth, contexts) 생성
"""

import os
import sys
import re
import json
import pandas as pd
from dotenv import load_dotenv
import asyncio
from datetime import datetime
from typing import List, Iterable

from langchain.schema import Document
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import embedding_factory
from ragas.testset.persona import Persona
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from langchain_openai import ChatOpenAI

# 🔌 Milvus helper (프로젝트 공용 헬퍼)
# - get_milvus_connection_info / search_milvus_documents 사용
from common.milvus_helpers import get_milvus_connection_info, search_milvus_documents

# ===================== 환경설정 =====================
load_dotenv()
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED not OPENAI_API_KEY=REDACTED("❌ OPENAI_API_KEY=REDACTED(1)

# 생성 파라미터
TARGET_QUESTIONS = int(os.getenv("RAGAS_TARGET_Q", "50"))
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "concepts")
PER_TERM_K = int(os.getenv("MILVUS_PER_TERM_K", "30"))   # seed term 당 가져올 문서 수
MAX_DOCS = int(os.getenv("MILVUS_MAX_DOCS", "1000"))      # 전체 상한

# seed terms: 쉼표로 구분하여 환경변수로 주입 가능
# 예) RAGAS_SEED_TERMS="정의,원리,구성요소,장점,단점,비교,사례"
SEED_TERMS = [s.strip() for s in os.getenv("RAGAS_SEED_TERMS", "정의, 원리, 특징, 구성요소, 장단점, 비교, 사례, 적용").split(",") if s.strip()]

def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===================== Milvus → Documents =====================
def milvus_corpus_to_documents(
    milvus_data: dict,
    collection_name: str = DEFAULT_COLLECTION,
    seed_terms: Iterable[str] = SEED_TERMS,
    per_term_k: int = PER_TERM_K,
    max_docs: int = MAX_DOCS,
) -> List[Document]:
    """
    seed_terms 로 Milvus 유사도검색을 여러 번 돌려 코퍼스를 수집한다.
    - 프로젝트 공용 helper: search_milvus_documents(milvus_data, collection_name, query, k) 사용
    - 반환: LangChain Document 리스트(중복 제거)
    """
    if not milvus_data:
        raise RuntimeError("milvus_data 없음. get_milvus_connection_info() 또는 환경변수 설정을 확인하세요.")

    docs: List[Document] = []
    seen = set()

    for term in seed_terms:
        try:
            results = search_milvus_documents(
                milvus_data=milvus_data,
                collection_name=collection_name,
                query=term,
                k=per_term_k
            )
        except Exception as e:
            print(f"❌ Milvus 검색 실패(term='{term}'): {e}")
            continue

        for d in results or []:
            # d 는 Document 로 가정 (retrieve_agent와 동일 패턴)
            # 중복 제거 키: 내용 + (선택) 일부 메타
            key = (clean_text(getattr(d, "page_content", ""))[:200], json.dumps(getattr(d, "metadata", {}), sort_keys=True))
            if key in seen:
                continue
            seen.add(key)

            content = clean_text(getattr(d, "page_content", ""))
            if not content or len(content) < 50:
                continue

            # 메타데이터에 컬렉션/seed 항목 부여
            meta = dict(getattr(d, "metadata", {}) or {})
            meta.setdefault("source", f"milvus:{collection_name}")
            meta.setdefault("seed_term", term)
            docs.append(Document(page_content=content, metadata=meta))

        print(f"✅ term='{term}' → {len(results or [])}개 수집, 누적={len(docs)}")

        if len(docs) >= max_docs:
            print(f"⛔ MAX_DOCS({max_docs}) 도달. 수집 중단.")
            break

    print(f"📚 최종 코퍼스 문서 수: {len(docs)}")
    return docs

# ===================== 페르소나 (concepts 특화) =====================
def get_concept_persona() -> list[Persona]:
    return [
        Persona(
            name="ConceptSeeker",
            role_description=(
                "나는 소프트웨어/컴퓨터공학/데이터/인공지능 등의 '개념'을 체계적으로 학습하려는 학습자다. "
                "핵심 정의, 원리, 특징, 구성요소, 장단점, 유사 개념과의 비교, 적용 사례 등을 구체적으로 알고 싶어 한다. "
                "질문과 정답은 반드시 제공된 컨텍스트 내 사실에 근거해야 하며, "
                "문서의 출처를 묻는 질문이 아니라 개념 그 자체의 내용에 대한 질문만 생성하라."
            )
        )
    ]

# ===================== 골든셋 생성 =====================
def generate_golden_set_from_docs(documents: List[Document]):
    generator_llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        max_tokens=2000,
    ))
    generator_embeddings = embedding_factory("openai", model="text-embedding-3-large")

    personas = get_concept_persona()
    transforms = [HeadlineSplitter(), NERExtractor()]

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas,
    )

    # 안정성: SingleHop 전용
    query = SingleHopSpecificQuerySynthesizer(llm=generator_llm)
    prompts = asyncio.run(query.adapt_prompts("ko", llm=generator_llm))
    query.set_prompts(**prompts)

    dataset = generator.generate_with_langchain_docs(
        documents,
        testset_size=TARGET_QUESTIONS,
        transforms=transforms,
        query_distribution=[(query, 1.0)],
    )

    eval_dataset = dataset.to_evaluation_dataset()

    results = []
    for i, sample in enumerate(eval_dataset[:TARGET_QUESTIONS]):
        print(f"\n질문 {i+1}")
        print("  Question:", sample.user_input)
        print("  Ground Truth:", sample.reference)
        print("  Contexts:", sample.reference_contexts)
        results.append({
            "question": sample.user_input,
            "ground_truth": sample.reference,
            "contexts": sample.reference_contexts,
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"golden_dataset_milvus_{timestamp}.csv"
    jsonl_file = f"golden_dataset_milvus_{timestamp}.jsonl"

    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ {csv_file} / {jsonl_file} 저장 완료")
    print(f"📊 생성된 골든셋 개수: {len(results)}")

# ===================== 실행 진입점 =====================
def run_milvus_mode():
    print("\n🔌 Milvus(concepts) 코퍼스 모드")
    # 연결정보: 프로젝트 공용 함수 사용 (환경변수/설정에서 로드)
    milvus_data = get_milvus_connection_info()
    docs = milvus_corpus_to_documents(
        milvus_data=milvus_data,
        collection_name=DEFAULT_COLLECTION,
        seed_terms=SEED_TERMS,
        per_term_k=PER_TERM_K,
        max_docs=MAX_DOCS,
    )
    if not docs:
        print("❌ Milvus로부터 수집된 문서가 없습니다.")
        return
    generate_golden_set_from_docs(docs)

if __name__ == "__main__":
    print("=" * 50)
    print("🔧 RAGAS 골든셋 생성기 (Milvus 모드 포함)")
    print("=" * 50)
    print("2: Milvus(concepts) 코퍼스에서 생성")
    print("=" * 50)

    # 인터랙티브 없이 곧바로 2번 실행해도 됨
    mode = os.getenv("RAGAS_MODE", "2").strip()
    if mode == "2":
        run_milvus_mode()
    else:
        # 안전하게 디폴트를 Milvus로
        run_milvus_mode()
