# -*- coding: utf-8 -*-
"""
PDF → 청크 → RAGAS 평가 데이터셋 생성기
- PDF 디렉토리에서 문서 로드
- 텍스트 청크 분할
- OpenAI LLM으로 QA 생성
- SentenceTransformer 임베딩으로 k-NN 기반 contexts 생성
- 최종적으로 JSONL/CSV 저장
"""

import os
import re
import json
import random
import argparse
import numpy as np
import asyncio
import pandas as pd
from glob import glob
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# ===================== LLM 프롬프트 =====================
SYSTEM_PROMPT = (
    "너는 농업/재해 문서를 읽고 그 내용에만 근거하여 '질문-정답'을 만드는 어시스턴트야. "
    "절대 문서에 없는 사실을 추가하지 마. 숫자/단위/지명을 문서 그대로 사용해."
    "너는 농업 재해 전문가야. 농업 재해 전문가의 관점에서 질문과 정답을 만들어줘."
    "질문은 작물의 재해 상황에 대한 대비에 가장 초점을 맞춰줘 예를 들면 토마토를 키우는데 내일 태풍이 온다면 어떻게 대비해야 하는지 이런 질문들로"
    "정답은 문서 내용에만 근거하여 작성해줘"
    "너무 중복된 질문들은 배제해줘"
    "작년 사례 관련 질문들도 몇 개 정도 넣어줘"
)

USER_PROMPT_TMPL = """아래는 단일 문서 청크입니다. 이 텍스트 **내용만** 보고,
1) 한국어 '질문(question)' 1개
2) 해당 질문에 대한 '정답(ground_truth)' 1개

를 JSON으로 만들어줘.

반드시 아래 JSON 스키마를 만족해야 해:
{{
  "question": "질문(한국어, 한 문장)",
  "ground_truth": "정답(문서 내용에만 근거, 한~세 문장)"
}}

[문서 청크]
{chunk}
"""

# ===================== OpenAI 호출 =====================
def call_openai_json(client: OpenAI, model: str, chunk_text: str, temperature: float = 0.2) -> Dict[str, str] | None:
    """OpenAI로 질문/정답 생성 (JSON 출력). 실패 시 None."""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TMPL.format(chunk=chunk_text)}
            ]
        )
        content = resp.choices[0].message.content or ""
        # JSON 추출 (코드블록 감싸도 처리)
        m = re.search(r"\{.*\}", content, flags=re.S)
        jtxt = m.group(0) if m else content
        data = json.loads(jtxt)
        return {"question": data["question"].strip(), "ground_truth": data["ground_truth"].strip()}
    except Exception as e:
        print(f"⚠️ QA 생성 실패: {e}")
        return None

# ===================== Embedding & 유사도 =====================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)

def build_contexts_for_chunk(idx: int, chunks: List[str], vecs: np.ndarray, k: int = 3) -> List[str]:
    """기준 청크 idx와 가장 유사한 k개 문맥을 contexts로 구성"""
    sims = cosine_sim(vecs[[idx]], vecs).ravel()
    order = np.argsort(-sims)[:k]
    return [chunks[j] for j in order]

# ===================== 평가 데이터셋 생성 =====================
def build_eval_dataset(chunks: List[str], num_items: int, openai_model: str, openai_key: str,
                       embed_model: str = "BAAI/bge-m3", k: int = 3) -> List[Dict[str, Any]]:
    """청크 → RAGAS 평가 데이터셋"""
    client = OpenAI(api_key=openai_key)

    # 임베딩
    model = SentenceTransformer(embed_model)
    vecs = model.encode(chunks, batch_size=64, normalize_embeddings=True)

    dataset = []
    seen_q = set()

    for idx in random.sample(range(len(chunks)), min(len(chunks), num_items*2)):
        if len(dataset) >= num_items:
            break

        qa = call_openai_json(client, openai_model, chunks[idx])
        if not qa:
            continue
        if qa["question"] in seen_q:
            continue
        seen_q.add(qa["question"])

        contexts = build_contexts_for_chunk(idx, chunks, np.array(vecs), k=k)

        dataset.append({
            "question": qa["question"],
            "ground_truth": qa["ground_truth"],
            "contexts": contexts
        })

    return dataset

# ===================== PDF → 청크 =====================
def load_pdfs_as_chunks(pdf_dir: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """PDF 디렉토리에서 텍스트를 로드해 청크 리스트로 반환"""
    pdf_paths = sorted(glob(os.path.join(pdf_dir, "*.pdf")))  # *.pdf로 수정
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[str] = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for d in docs:
            for chunk in splitter.split_text(d.page_content):
                chunks.append(chunk)

    print(f"📚 로드 완료: {len(pdf_paths)}개 PDF → 총 {len(chunks)}개 청크")
    return chunks

# ===================== 실행부 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF 기반 RAGAS 평가 데이터셋 생성기")
    parser.add_argument("--pdf-dir", type=str, default="./farmer/disaster/pdfs", help="PDF 파일이 있는 디렉토리")
    parser.add_argument("--num-items", type=int, default=50, help="생성할 QA 쌍 개수")
    parser.add_argument("--out", type=str, default="ragas_eval.jsonl", help="출력 파일 경로")
    # CSV는 항상 저장되므로 제거
    args = parser.parse_args()

    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    chunks = load_pdfs_as_chunks(args.pdf_dir, chunk_size=800, chunk_overlap=100)

    dataset = build_eval_dataset(
        chunks,
        num_items=args.num_items,
        openai_model=OPENAI_MODEL,
        openai_key=OPENAI_KEY,
        embed_model="BAAI/bge-m3",
        k=3
    )

    if dataset:
        # JSONL 저장
        with open(args.out, "w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"✅ JSONL 저장 완료: {args.out} ({len(dataset)}건)")

        # CSV 저장 (항상 저장)
        csv_path = args.out.replace(".jsonl", ".csv")
        pd.DataFrame(dataset).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ CSV 저장 완료: {csv_path} ({len(dataset)}건)")
    else:
        print("❌ 생성된 데이터가 없습니다.")
        
if __name__ == "__main__":
    asyncio.run(main())