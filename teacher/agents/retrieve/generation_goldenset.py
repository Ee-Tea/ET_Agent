# -*- coding: utf-8 -*-
"""
로컬 JSON 코퍼스 기반 RAGAS 골든셋 생성기 (ragas 0.3.3 호환, 한국어 패치 버전)

개요:
- 지정 폴더(하위 포함)의 모든 .json을 스캔하여 LangChain Document로 적재
- RAGAS TestsetGenerator + SingleHopSpecificQuerySynthesizer로 질문/정답/컨텍스트 생성
- 한국어 질의 프롬프트/페르소나/정규화/임베딩을 한글 친화적으로 설정
- 결과를 JSONL/CSV로 저장

환경변수(.env):
- OPENAI_API_KEY=REDACTED API 키 (필수)
- RAGAS_TARGET_Q            : 생성할 질문 개수 (기본 50)
- JSON_MAX_DOCS             : 로딩할 최대 문서 수 (기본 2000)
- JSON_MIN_CHARS            : 문서 최소 문자열 길이(짧은 잡음 필터) (기본 50)
- JSON_CORPUS_DIR           : JSON 코퍼스 루트 경로 (기본 teacher/agents/retrieve/data/json)
- RAGAS_LANG                : 합성 언어 코드(기본 "ko")
"""

import os
import sys
import re
import json
import glob
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any

from langchain.schema import Document

import asyncio
import random

# ===== RAGAS / LangChain 구성 =====
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.testset.persona import Persona
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from langchain_openai import ChatOpenAI

# (임베딩) ragas의 HuggingFaceEmbeddings 래퍼 사용 (다국어 모델 권장)
from ragas.embeddings import HuggingFaceEmbeddings

# ===================== 환경설정 =====================
load_dotenv()
OPENAI_API_KEY=REDACTED("OPENAI_API_KEY=REDACTED not OPENAI_API_KEY=REDACTED("❌ OPENAI_API_KEY=REDACTED를 확인하세요)")
    sys.exit(1)

# 재현성(일부 무작위 샘플링 방지용, LLM은 비결정적일 수 있음)
random.seed(42)

TARGET_QUESTIONS = int(os.getenv("RAGAS_TARGET_Q", "100"))
MAX_DOCS        = int(os.getenv("JSON_MAX_DOCS", "2000"))
MIN_CHARS       = int(os.getenv("JSON_MIN_CHARS", "50"))
JSON_CORPUS_DIR = os.getenv(
    "JSON_CORPUS_DIR",
    os.path.join("teacher", "agents", "retrieve", "data", "json")
)
RAGAS_LANG      = os.getenv("RAGAS_LANG", "ko").strip().lower() or "ko"

# ===================== 유틸 =====================
def clean_text(text: str) -> str:
    """
    공백/개행/제로폭 문자 등을 정규화하여 RAG 파이프라인에서
    쓸데없는 잡음을 줄인다. (한글 포함)
    """
    s = str(text or "")
    # 제로폭 문자/비표준 공백 제거
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    # 다중 공백/개행 -> 단일 공백
    s = re.sub(r"\s+", " ", s)
    return s.strip()

from langchain_text_splitters import RecursiveCharacterTextSplitter

SPLIT_CHARS = 850
SPLIT_OVERLAP = 140

def split_docs(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SPLIT_CHARS,
        chunk_overlap=SPLIT_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for d in documents:
        # 제목(있으면 1줄만) + 본문
        title = (d.metadata.get("item_title") or "").strip()
        prefixed = (f"{title}\n{d.page_content}" if title else d.page_content).strip()
        d2 = Document(page_content=prefixed, metadata=d.metadata)
        chunks.extend(splitter.split_documents([d2]))
    # 너무 짧은 파편 제거
    chunks = [c for c in chunks if len(c.page_content) >= 200]
    return chunks


# ===================== JSON → Document 로더 =====================
def load_json_docs(root_dir: str) -> List[Document]:
    """
    재귀적으로 root_dir 아래의 *.json 파일을 읽어
    items[].content(필수)와 item_title(있으면)을 합쳐 page_content로 만들어 Document 생성.

    지원 형식:
    - 형태 A: {"subject": "...", "total_items": N, "items": [ {...}, ... ]}
    - 형태 B: [ {...}, {...}, ... ]  # 루트가 리스트인 예외 케이스
    """
    paths = glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)
    docs: List[Document] = []
    files, objects = 0, 0

    for path in paths:
        # JSON 파싱
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            files += 1
        except Exception as e:
            print(f"⚠️ JSON 로드 실패: {path} ({e})")
            continue

        top_subject = data.get("subject") if isinstance(data, dict) else None

        # 형태 A: 딕셔너리 + items 리스트
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            for item in data["items"]:
                if not isinstance(item, dict):
                    continue
                objects += 1
                content = clean_text(item.get("content", ""))
                if not content:
                    continue

                title = clean_text(item.get("item_title", ""))
                page_content = f"{title}\n{content}" if title else content
                if len(page_content) < MIN_CHARS:
                    # 지나치게 짧은 텍스트는 노이즈 가능성 높음 → 스킵
                    continue

                meta = {
                    "top_subject": top_subject,
                    "subject": item.get("subject"),
                    "item_id": item.get("item_id"),
                    "item_title": title or None,
                    "chunk_size": item.get("chunk_size"),
                    "source": path,
                }
                docs.append(Document(page_content=page_content, metadata=meta))

        # 형태 B: 루트가 리스트
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                objects += 1
                content = clean_text(item.get("content", ""))
                if not content:
                    continue

                title = clean_text(item.get("item_title", ""))
                page_content = f"{title}\n{content}" if title else content
                if len(page_content) < MIN_CHARS:
                    continue

                meta = {
                    "top_subject": top_subject,
                    "subject": item.get("subject"),
                    "item_id": item.get("item_id") or str(idx),
                    "item_title": title or None,
                    "chunk_size": item.get("chunk_size"),
                    "source": path,
                }
                docs.append(Document(page_content=page_content, metadata=meta))

        # 문서 수 상한 도달 시 조기 종료
        if len(docs) >= MAX_DOCS:
            break

    print(f"📚 파일 스캔 {files}개, 객체 스캔 {objects}개, 문서 {len(docs)}개")
    return docs

# ===================== 페르소나 (개념 학습 특화, 한국어) =====================
def get_concept_persona() -> List[Persona]:
    """
    개념 정리/비교/사례 중심의 질문을 유도하는 학습자 페르소나.
    - 출처 묻기/메타 질문 금지, 개념 내용만 질문하도록 제약
    """
    KOR_GUIDE = (
        "질문과 정답은 반드시 제공된 컨텍스트의 사실에 근거해야 하며, "
        "정답에는 컨텍스트 내 표현(정의·키워드)을 가급적 직접 인용하거나 동치로 재서술하라. "
        "출처/메타질문 금지. 자료 밖 확장·추론 금지. 한 질문은 하나의 핵심 개념만 다뤄라."
    )
    return [
        Persona(
            name="ConceptSeeker",
            role_description=(
                "정보처리기사 등 개념 학습용. 핵심 정의/구성요소/특징/비교/역할/제약 등만 묻는다. "
                + KOR_GUIDE
            ),
        )
        # Persona(
        #     name="ConceptSeeker",
        #     role_description=(
        #         "나는 정보처리기사(또는 유사 개념 학습)와 관련된 '개념'을 체계적으로 학습하려는 학습자다. "
        #         "핵심 정의, 원리, 특징, 구성요소, 장단점, 유사 개념과의 비교, 적용 사례 등을 구체적으로 알고 싶다. "
        #         "질문과 정답은 반드시 제공된 컨텍스트의 사실에 근거해야 하며, "
        #         "문서의 출처나 저작권 등 메타 정보를 묻는 질문은 만들지 말고, "
        #         "오직 개념 이해/적용/비교를 돕는 질문만 생성하라."
        #     ),
        # )
    ]

# ===================== 골든셋 생성 (한국어 패치 포함) =====================
def generate_golden_set_from_docs(documents: List[Document]) -> None:
    """
    - LLM: ChatOpenAI(gpt-4o-mini)
    - 임베딩: HuggingFace multilingual-e5-large (한글/영문 강함)
    - 트랜스폼: NERExtractor (핵심 엔티티 힌트를 이용한 안정적 질문 생성에 도움)
    - Synthesizer: SingleHopSpecificQuerySynthesizer (단일 홉, 구체 질문)
    - Synthesizer 프롬프트: 한국어(RAGAS_LANG=ko) 적용, 실패 시 영어 폴백
    """
    # LLM 래퍼 (RAGAS에서 요구하는 인터페이스 맞추기)
    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,     # 헛소리/창발 최소화
            max_tokens=2000,
        )
    )

    # ✅ 한국어 친화 임베딩 (문서/질문 임베딩 모두에 사용)
    generator_embeddings = HuggingFaceEmbeddings(
        model="intfloat/multilingual-e5-large"
    )

    personas = get_concept_persona()
    transforms = [NERExtractor()]  # 헤드라인 분할은 제거 (JSON 단락 특성 반영)

    # ===== Synthesizer(질의 생성기) 한국어 프롬프트 적용 =====
    query = SingleHopSpecificQuerySynthesizer(llm=generator_llm)

    # 한국어 프롬프트 자동 적용 (ragas가 제공하는 다국어 프롬프트)
    try:
        prompts = asyncio.run(query.adapt_prompts(RAGAS_LANG, llm=generator_llm))
        query.set_prompts(**prompts)
    except Exception as e:
        # 드물게 언어 코드/네트워크 이슈 등으로 실패할 수 있어 안전 폴백
        print(f"⚠️ '{RAGAS_LANG}' 프롬프트 적용 실패, 영어 프롬프트로 대체합니다. ({e})")
        prompts = asyncio.run(query.adapt_prompts("en", llm=generator_llm))
        query.set_prompts(**prompts)

    # ===== TestsetGenerator로 실제 생성 수행 =====
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas,
    )

    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=TARGET_QUESTIONS,
        transforms=transforms,
        # 단일 홉 구체 질문만 비율 1.0로 사용
        query_distribution=[(query, 1.0)],
    )

    # ===== 저장 (UTF-8/UTF-8-SIG) =====
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("teacher", "agents", "retrieve", "goldensets")
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, f"goldenset_{ts}.jsonl")
    csv_path   = os.path.join(out_dir, f"goldenset_{ts}.csv")

    # RAGAS 0.3.3 평가 데이터셋으로 직렬화
    eval_dataset = testset.to_evaluation_dataset()

    # JSONL (한글 포함, 안전 직렬화)
    with open(jsonl_path, "w", encoding="utf-8", newline="") as f:
        for sample in eval_dataset:
            row = {
                "question": sample.user_input,
                "ground_truth": sample.reference,
                "contexts": sample.reference_contexts,
            }
            safe = json.dumps(row, ensure_ascii=False)
            f.write(safe + "\n")

    # CSV (엑셀 호환을 위해 UTF-8-SIG)
    rows = [
        {
            "question": s.user_input,
            "ground_truth": s.reference,
            "contexts": s.reference_contexts,
        }
        for s in eval_dataset
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ 저장 완료")
    print(f" - JSONL: {jsonl_path}")
    print(f" - CSV  : {csv_path}")
    print(f"📊 생성된 골든셋 개수: {len(rows)}")

# ===================== 실행 진입점 =====================
def main():
    print("=" * 50)
    print("🔧 RAGAS 골든셋 생성기 (로컬 JSON 모드, 재귀 스캔, 한국어 패치)")
    print("=" * 50)
    print("\n📂 로컬 JSON 코퍼스 모드")
    print(f"   - 경로(재귀): {JSON_CORPUS_DIR}")
    print(f"   - 언어      : {RAGAS_LANG}")
    print(f"   - 문서 상한 : {MAX_DOCS}, 최소 길이: {MIN_CHARS}, 타깃 문항: {TARGET_QUESTIONS}")

    docs = load_json_docs(JSON_CORPUS_DIR)
    docs = split_docs(docs)
    if not docs:
        print("❌ 로컬 JSON에서 수집된 문서가 없습니다.")
        sys.exit(1)

    generate_golden_set_from_docs(docs)

if __name__ == "__main__":
    main()
