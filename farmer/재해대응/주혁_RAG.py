# -*- coding: utf-8 -*-
"""
PDF 텍스트 + (필요 시) OCR, 이미지 OCR까지 포함하여
LangChain-FAISS 디렉토리(index.faiss/index.pkl)에 저장하는 하이브리드 인덱서.

- 기본 흐름: 로드 → 정제/분할 → 임베딩 → FAISS 저장/갱신
- PDF는 PyPDFLoader로 텍스트를 우선 사용하고, 페이지 텍스트가 부족하면 PyMuPDF+easyocr로 OCR 보충
- 이미지(jpg/png/jpeg)는 easyocr로 텍스트 추출 후 포함(옵션)

필요 env (.env):
  PDF_DIR=./cropinfo
  IMAGE_DIR=./images               # 선택
  VECTOR_DB_PATH=faiss_pdf_db
  EMBED_MODEL_NAME=jhgan/ko-sroberta-multitask
  CHUNK_SIZE=900
  CHUNK_OVERLAP=150
  USE_OCR=1                        # 1이면 OCR 사용, 0이면 미사용
  OCR_LANGS=ko,en                  # easyocr 언어
  OCR_CONF=0.7                     # easyocr 최소 신뢰도
  OCR_TRIGGER_LEN=80               # PDF 페이지 텍스트 길이가 이 값 미만이면 OCR 보충
  FORCE_OCR=0                      # 1이면 PDF 전 페이지 OCR 강제
"""

import os
import re
from glob import glob
from typing import List, Any

from dotenv import load_dotenv
load_dotenv()

# ===== 설정 =====
PDF_DIR = os.getenv("PDF_DIR", "./pdfs")
IMAGE_DIR = os.getenv("IMAGE_DIR", "").strip()
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "faiss_kma_db")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

USE_OCR = os.getenv("USE_OCR", "1") == "1"
OCR_LANGS = [s.strip() for s in os.getenv("OCR_LANGS", "ko,en").split(",") if s.strip()]
OCR_CONF = float(os.getenv("OCR_CONF", "0.7"))
OCR_TRIGGER_LEN = int(os.getenv("OCR_TRIGGER_LEN", "80"))
FORCE_OCR = os.getenv("FORCE_OCR", "0") == "1"

# ===== LangChain =====
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ===== OCR / PDF 보조 =====
import fitz  # PyMuPDF
import easyocr
from PIL import Image
import numpy as np
import io

# ===== 텍스트 정제/분할 =====
def clean_text(text: str) -> str:
    # 한글/영문/숫자/일부 구두점만 남기고 공백 정리
    text = re.sub(r"[^가-힣a-zA-Z0-9.,()/%:\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

# ===== OCR 유틸 =====
_ocr_reader = None
def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        # GPU 사용 가능하면 자동 활용
        import torch
        _ocr_reader = easyocr.Reader(OCR_LANGS, gpu=torch.cuda.is_available())
    return _ocr_reader

def ocr_image_pil(img: Image.Image) -> str:
    reader = get_ocr_reader()
    arr = np.array(img.convert("RGB"))
    results = reader.readtext(arr, detail=1)
    txts = [t for _, t, conf in results if conf >= OCR_CONF]
    return " ".join(txts).strip()

def ocr_pdf_page(page: fitz.Page, dpi: int = 200) -> str:
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return ocr_image_pil(img)

# ===== PDF 로드(+OCR 보강) =====
def load_pdfs_with_optional_ocr(pdf_dir: str) -> List[Document]:
    paths = sorted(glob(os.path.join(pdf_dir, "*.pdf")))
    if not paths:
        raise FileNotFoundError(f"PDF가 없습니다: {pdf_dir}")

    all_docs: List[Document] = []
    splitter = make_splitter()

    for p in paths:
        print(f"📄 PDF 로드: {os.path.basename(p)}")
        # 1) 우선 PyPDFLoader로 페이지 단위 로드
        pages = []
        try:
            pages = PyPDFLoader(p).load()
        except Exception as e:
            print(f"❗ PyPDFLoader 실패: {p} -> {e}")

        # 2) 페이지별 텍스트 확인 & OCR 보강
        #    - FORCE_OCR이면 무조건 OCR 수행해 텍스트에 덧붙임
        #    - 아니면 텍스트 길이가 OCR_TRIGGER_LEN 미만일 때만 OCR 수행
        try:
            doc_fitz = fitz.open(p)
        except Exception as e:
            print(f"❗ PyMuPDF 열기 실패: {p} -> {e}")
            doc_fitz = None

        processed_docs: List[Document] = []

        for i, d in enumerate(pages):
            base_txt = clean_text(d.page_content or "")
            ocr_txt = ""
            need_ocr = FORCE_OCR or (len(base_txt) < OCR_TRIGGER_LEN)
            if USE_OCR and need_ocr and doc_fitz:
                try:
                    ocr_txt = clean_text(ocr_pdf_page(doc_fitz[i]))
                except Exception as e:
                    print(f"❗ OCR 실패: {p} page={i+1} -> {e}")

            merged = (base_txt + " " + ocr_txt).strip() if ocr_txt else base_txt
            if not merged:
                continue

            # 분할 후 Document로 변환(메타 유지)
            for chunk in splitter.split_text(merged):
                processed_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **(d.metadata or {}),
                            "source": d.metadata.get("source", p),
                            "page": d.metadata.get("page", i + 1),
                            "ocr_applied": bool(ocr_txt),
                            "file_name": os.path.basename(p),
                            "type": "pdf",
                        },
                    )
                )

        print(f"✅ 분할 청크 수: {len(processed_docs)} (OCR 적용 페이지 포함)")
        all_docs.extend(processed_docs)

    print(f"📚 PDF 총 청크 수: {len(all_docs)}")
    return all_docs

# ===== 이미지 OCR 로드(옵션) =====
def load_images_ocr(image_dir: str) -> List[Document]:
    if not image_dir:
        return []
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob(os.path.join(image_dir, ext)))
    paths = sorted(paths)
    if not paths:
        return []

    splitter = make_splitter()
    docs: List[Document] = []

    print(f"🖼️ 이미지 로드: {len(paths)}개 (dir={image_dir})")
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            txt = clean_text(ocr_image_pil(img)) if USE_OCR else ""
            if not txt:
                continue
            for chunk in splitter.split_text(txt):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": p,
                            "file_name": os.path.basename(p),
                            "type": "image",
                            "ocr_applied": True,
                        },
                    )
                )
        except Exception as e:
            print(f"❗ 이미지 처리 실패: {p} -> {e}")

    print(f"✅ 이미지 청크 수: {len(docs)}")
    return docs

# ===== 인덱스 생성/갱신 =====
def build_or_update_faiss(docs: List[Document], db_path: str) -> None:
    if not docs:
        print("⚠️ 인덱싱할 문서가 없습니다. 종료합니다.")
        return

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    if os.path.exists(db_path):
        print(f"📦 기존 인덱스 로드: {db_path}")
        vs = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print("➕ 새 청크 추가 후 인덱스 저장…")
        vs.add_documents(docs)
        vs.save_local(db_path)
    else:
        print("🧱 새 인덱스 생성…")
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(db_path)
    print(f"✅ 저장 완료: {db_path} (index.faiss / index.pkl)")

# ===== 메인 =====
if __name__ == "__main__":
    print("🚀 하이브리드 인덱스 빌드 시작")
    all_docs: List[Document] = []

    # 1) PDF(+OCR 보강)
    pdf_docs = load_pdfs_with_optional_ocr(PDF_DIR)
    all_docs.extend(pdf_docs)

    # 2) 이미지 OCR (옵션)
    if IMAGE_DIR:
        img_docs = load_images_ocr(IMAGE_DIR)
        all_docs.extend(img_docs)

    # 3) 인덱스 빌드/갱신
    build_or_update_faiss(all_docs, VECTOR_DB_PATH)
    print("🎉 인덱스 빌드 완료")
