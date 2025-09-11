# -*- coding: utf-8 -*-
"""
골든셋 생성기 (PDF → Chunk → Q/A 생성)
- 하이브리드 인덱서(clean_text, chunk_size=900, overlap=150) 기준 사용
- PDF 폴더에서 문서를 읽고 청크를 만든 후, OpenAI LLM으로 질문/답변 생성
- 결과를 JSONL 또는 CSV로 저장 가능
"""

import os
import re
import json
import random
import argparse
import pandas as pd
from glob import glob
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz  # PyMuPDF

# ================= 환경설정 =================
load_dotenv()

PDF_DIR = os.getenv("PDF_DIR", "./farmer/disaster/pdfs")
OUT_JSONL = os.getenv("OUT_JSONL", "golden_set.jsonl")
OUT_CSV = os.getenv("OUT_CSV", "golden_set.csv")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# ================= 유틸 =================
def clean_text(text: str) -> str:
    """텍스트 전처리 (인덱서 기준 동일)"""
    text = re.sub(r"[^\w\s\.\,\(\)\/%\:\-\~가-힣]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

def load_pdfs(pdf_dir: str) -> list[Document]:
    """PDF에서 텍스트를 추출하고 Document 청크로 변환"""
    docs = []
    pdf_paths = sorted(glob(os.path.join(pdf_dir, "*.pdf")))
    splitter = make_splitter()

    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, start=1):
                text = clean_text(page.get_text())
                if not text:
                    continue
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "file_name": os.path.basename(pdf_path),
                            "page": page_num,
                            "type": "pdf_text"
                        }
                    ))
        except Exception as e:
            print(f"⚠️ PDF 처리 실패: {pdf_path} - {e}")
    print(f"📚 총 청크 수: {len(docs)}")
    return docs

# ================= LLM =================
def make_llm():
    if not OPENAI_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY가 필요합니다 (.env 확인)")
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY
    )

# 질문 생성 프롬프트
QUESTION_PROMPT = """아래 텍스트 내용을 기반으로,
농업 재배와 관련된 구체적이고 실용적인 질문을 1개 만들어주세요.
질문은 반드시 한국어여야 하며, 텍스트의 핵심 내용 직접적으로 관련 있어야 합니다.
질문은 작물의 재배, 관리, 수확, 병해충 방제 등 실무적인 내용에 초점을 맞춰주세요.

[텍스트]
{text}
"""

# 답변 생성 프롬프트
ANSWER_PROMPT = """아래 텍스트 내용을 기반으로,
주어진 질문에 정확하고 구체적인 답변을 작성해주세요.
답변은 한국어로 작성하며, 텍스트에 있는 정보만을 사용하여 답변하세요.
답변은 실무진이 바로 활용할 수 있도록 구체적이고 명확하게 작성해주세요.

[질문]
{question}

[텍스트]
{text}
"""

def generate_qa(llm, chunk: Document) -> tuple[str, str, str]:
    """단일 청크에서 Q/A 생성"""
    try:
        q = llm.invoke(QUESTION_PROMPT.format(text=chunk.page_content)).content.strip()
        a = llm.invoke(ANSWER_PROMPT.format(
            question=q,
            text=chunk.page_content
        )).content.strip()
        # 컨텍스트는 청크 내용을 그대로 사용
        context = chunk.page_content
        return q, a, context
    except Exception as e:
        print(f"⚠️ QA 생성 실패: {e}")
        return None, None, None

# ================= 메인 =================
def build_golden_set(num_samples: int, out_jsonl: str, out_csv: str):
    docs = load_pdfs(PDF_DIR)
    if not docs:
        print("❌ PDF에서 추출된 청크가 없습니다.")
        return

    # 무작위 샘플 선택
    sampled_docs = random.sample(docs, min(num_samples, len(docs)))
    llm = make_llm()

    results = []
    for idx, doc in enumerate(sampled_docs, 1):
        print(f"🤖 QA 생성 중... {idx}/{len(sampled_docs)}")
        q, a, context = generate_qa(llm, doc)
        if not q or not a or not context:
            continue
        record = {
            "question": q,
            "ground_truth": a,
            "contexts": [context]
        }
        results.append(record)

    if not results:
        print("❌ 생성된 결과가 없습니다.")
        return

    # JSONL 저장
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ JSONL 저장 완료: {out_jsonl} ({len(results)}건)")

    # CSV 저장 (기존 형식 유지)
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ CSV 저장 완료: {out_csv} ({len(results)}건)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="골든셋 생성기 (PDF → QA)")
    parser.add_argument("--num-items", type=int, default=5, help="생성할 QA 개수")
    parser.add_argument("--out-jsonl", type=str, default=OUT_JSONL, help="저장할 JSONL 파일 경로")
    parser.add_argument("--out-csv", type=str, default=OUT_CSV, help="저장할 CSV 파일 경로")
    args = parser.parse_args()

    build_golden_set(args.num_items, args.out_jsonl, args.out_csv)
