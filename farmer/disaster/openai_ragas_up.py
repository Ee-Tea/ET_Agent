# -*- coding: utf-8 -*-
"""
하이브리드 인덱서 + RAGAS 골든셋 생성기 (SingleHop 전용 안정 버전, ragas 0.3.3 대응)
- PDF 텍스트 + OCR 항상 실행 + 표 파싱 항상 실행
- ragas TestsetGenerator로 골든셋(question, ground_truth, contexts) 생성
- SingleHopSpecificQuerySynthesizer만 사용 (안정성 보장)
"""

import os
import sys
import re
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from dotenv import load_dotenv
import asyncio

import fitz  # PyMuPDF
import easyocr
import pdfplumber
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import embedding_factory
from ragas.testset.persona import Persona
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer

from langchain_openai import ChatOpenAI

# ===================== 환경설정 =====================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY 없음")
    sys.exit(1)

PDF_DIR = "./farmer/disaster/pdfs"
IMAGE_DIR = ""
TARGET_QUESTIONS = 50   # 🔥 생성할 질문 수
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# ===================== 전처리 (OCR + 표 파싱) =====================
ocr_reader = None

def _get_ocr_reader() -> easyocr.Reader:
    global ocr_reader
    if ocr_reader is None:
        print("EasyOCR Reader 로드 중...")
        ocr_reader = easyocr.Reader(["ko", "en"], gpu=True)
    return ocr_reader

def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s\.\,\(\)\/%\:\-\~가-힣]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )

def process_single_pdf(file_path: str) -> list[Document]:
    docs: list[Document] = []
    try:
        # --- 텍스트 + OCR 항상 실행 ---
        doc_fitz = fitz.open(file_path)
        for page_num, page in enumerate(doc_fitz, start=1):
            text = page.get_text()
            try:
                reader = _get_ocr_reader()
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                ocr_results = reader.readtext(img_bytes)
                ocr_text = " ".join([res[1] for res in ocr_results])
                text = text + " " + ocr_text
            except Exception as e:
                print(f"OCR 실패: {file_path} p.{page_num} - {e}")

            cleaned = clean_text(text)
            if cleaned:
                splitter = make_splitter()
                for chunk in splitter.split_text(cleaned):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"file_name": os.path.basename(file_path), "page": page_num, "type": "pdf_text"}
                    ))

        # --- 표 파싱 항상 실행 ---
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                    for table in tables:
                        table_str = "\n".join(
                            ["\t".join([(cell or "") for cell in row]) for row in table if row]
                        )
                        cleaned = clean_text(table_str)
                        if not cleaned:
                            continue
                        splitter = make_splitter()
                        for chunk in splitter.split_text(cleaned):
                            docs.append(Document(
                                page_content=chunk,
                                metadata={"file_name": os.path.basename(file_path), "page": page_num, "type": "pdf_table"}
                            ))
                except Exception as e:
                    print(f"표 파싱 실패: {file_path} p.{page_num} - {e}")

    except Exception as e:
        print(f"PDF 처리 실패 {file_path}: {e}")
    return docs

def process_single_image(file_path: str) -> list[Document]:
    docs: list[Document] = []
    try:
        reader = _get_ocr_reader()
        ocr_results = reader.readtext(file_path)
        text = " ".join([res[1] for res in ocr_results])
        cleaned = clean_text(text)
        if cleaned:
            splitter = make_splitter()
            for chunk in splitter.split_text(cleaned):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"file_name": os.path.basename(file_path), "page": 1, "type": "image_ocr"}
                ))
    except Exception as e:
        print(f"이미지 처리 실패 {file_path}: {e}")
    return docs

def load_documents(pdf_dir: str, image_dir: str = "") -> list[Document]:
    all_docs = []
    pdf_paths = sorted(glob(os.path.join(pdf_dir, "*.pdf")))
    for path in tqdm(pdf_paths, desc="PDF 처리"):
        all_docs.extend(process_single_pdf(path))
    if image_dir:
        img_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            img_paths.extend(glob(os.path.join(image_dir, ext)))
        for path in tqdm(img_paths, desc="이미지 처리"):
            all_docs.extend(process_single_image(path))
    print(f"📚 총 {len(all_docs)}개 Document 생성")
    return all_docs

# ===================== 골든셋 생성 =====================
def generate_golden_set(documents: list[Document]):
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    generator_embeddings = embedding_factory("openai", model="text-embedding-3-large")

    personas = [
        Persona(
            name="Farmer",
            role_description=(
                "나는 한국의 농업에 종사하는 농부다. "
                "특히 기후 변화와 자연재해(태풍, 폭우, 가뭄, 한파, 폭염 등)로 인한 작물 피해에 큰 관심이 있다. "
                "과거의 재해 사례, 정부와 지자체의 대응 매뉴얼, 농촌진흥청 자료 등을 참고하여 "
                "내 작물을 어떻게 보호하고 피해를 최소화할 수 있을지 알고 싶다. "
                "나는 작물별 재해 대응법(예: 벼 침수 피해, 사과 서리 피해, 고추 폭염 피해)이나 "
                "재해 발생 전·후 단계별로 해야 할 조치(사전 예방, 발생 직후 대응, 사후 복구)에 대해 구체적으로 질문한다. "
                "또한, 내가 농사를 짓는 지역(강원도, 전라도 등)에 맞는 특화된 대응 방법을 찾고 싶어 한다. "
                "가능하다면, 표에 정리된 피해 통계나 과거 사례 데이터를 근거로 설명해주길 기대한다. "
                "질문과 정답을 생성할 때는 반드시 여러 PDF 문서를 골고루 참고해서 만들어라. "
                "특정 문서 한두 개에만 집중하지 말고, 가능한 다양한 문서를 근거로 질문-정답을 구성해라."
            )
        )
    ]
    transforms = [HeadlineSplitter(), NERExtractor()]

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas
    )

    # 안정성: SingleHop만 사용
    query = SingleHopSpecificQuerySynthesizer(llm=generator_llm)
    prompts = asyncio.run(query.adapt_prompts("ko", llm=generator_llm))
    query.set_prompts(**prompts)

    dataset = generator.generate_with_langchain_docs(
        documents,
        testset_size=TARGET_QUESTIONS,
        transforms=transforms,
        query_distribution=[(query, 1.0)]
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
            "contexts": sample.reference_contexts
        })

    # 저장
    df = pd.DataFrame(results)
    df.to_csv("golden_dataset_open.csv", index=False, encoding="utf-8-sig")
    with open("golden_dataset_open.jsonl", "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ golden_dataset_open.csv / .jsonl 저장 완료")
    print(f"📊 생성된 골든셋 개수: {len(results)}")

# ===================== 실행 =====================
if __name__ == "__main__":
    docs = load_documents(PDF_DIR, IMAGE_DIR)
    if docs:
        generate_golden_set(docs)
