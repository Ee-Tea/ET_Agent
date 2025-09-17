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
from datetime import datetime

import fitz  # PyMuPDF
import easyocr
import pdfplumber
import numpy as np
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

PDF_DIR = "./farmer/disaster/data"
SINGLE_PDF_PATH = "./farmer/disaster/2025 기상재해 대응기술 가이드북(주요 20작물).pdf"
IMAGE_DIR = ""
TARGET_QUESTIONS = 300  # 🔥 생성할 질문 수
CHUNK_SIZE = 900      # 더 긴 context로 변경
CHUNK_OVERLAP = 150    # overlap도 비례적으로 증가

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
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP, 
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # 더 자연스러운 분할
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
            if cleaned and len(cleaned) > 100:  # 최소 100자 이상만 처리
                splitter = make_splitter()
                for chunk in splitter.split_text(cleaned):
                    if len(chunk) > 200:  # 최소 200자 이상 chunk만 사용
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
                        if cleaned and len(cleaned) > 100:  # 최소 100자 이상만 처리
                            splitter = make_splitter()
                            for chunk in splitter.split_text(cleaned):
                                if len(chunk) > 200:  # 최소 200자 이상 chunk만 사용
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
    
    # 연도별 가중치 설정
    year_weights = {
        "2020": 3.0,  # 2020년 문서는 3배 가중치
        "2022": 2.5,  # 2022년 문서는 2.5배 가중치
        "2023": 2.0,  # 2023년 문서는 2배 가중치
        "2024": 2.0,  # 2024년 문서는 2배 가중치
        "2021": 1.0,  # 2021년 문서는 기본 가중치
    }
    
    for path in tqdm(pdf_paths, desc="PDF 처리"):
        docs = process_single_pdf(path)
        
        # 파일명에서 연도 추출하여 가중치 적용
        filename = os.path.basename(path)
        weight = 1.0
        for year, w in year_weights.items():
            if year in filename:
                weight = w
                break
        
        # 가중치만큼 문서 복제
        for _ in range(int(weight)):
            all_docs.extend(docs)
        
        # 소수점 가중치 처리 (예: 2.5배면 50% 확률로 한 번 더 추가)
        if weight > int(weight) and (weight - int(weight)) > 0.5:
            all_docs.extend(docs)
    
    if image_dir:
        img_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            img_paths.extend(glob(os.path.join(image_dir, ext)))
        for path in tqdm(img_paths, desc="이미지 처리"):
            all_docs.extend(process_single_image(path))
    print(f"📚 총 {len(all_docs)}개 Document 생성")
    return all_docs

# ===================== 페르소나 정의 =====================
def get_personas(mode: str) -> list[Persona]:
    """모드에 따라 다른 페르소나 반환"""
    if mode == "multi_pdf":
        # 여러 PDF용 페르소나 (종합적 관점)
        return [
            Persona(
                name="DiverseFarmer",
                role_description=(
                    "나는 한국의 농업에 종사하는 농부다. "
                    "기후 변화와 자연재해(태풍, 폭우, 가뭄, 한파, 폭염 등)로 인한 작물 피해에 큰 관심이 있다. "
                    "⚠️ 중요: 2021년에만 집중하지 말고, 2020년, 2022년, 2023년, 2024년 등 다양한 연도의 사례를 균형있게 다뤄줘. "
                    "2021년 이상기후 보고서보다는 다른 연도의 자료를 우선적으로 참고해줘. "
                    "또한, 다양한 지역(서울, 부산, 대구, 광주, 대전, 울산, 세종, 경기, 강원, 충북, 충남, 전북, 전남, 경북, 경남, 제주)과 "
                    "다양한 작물(벼, 밀, 옥수수, 콩, 고구마, 감자, 사과, 배, 복숭아, 포도, 딸기, 토마토, 고추, 배추, 무 등)에 대한 "
                    "특화된 피해 사례를 찾고 싶어 한다. "
                    "가능하다면, 표에 정리된 피해 통계나 과거 사례 데이터를 근거로 설명해주길 기대한다. "
                    "특정 문서 한두 개에만 집중하지 말고, 가능한 다양한 문서와 연도를 근거로 질문-정답을 구성해라."
                    "이게 무슨 문서인지에 대한 질문은 만들지 말고 문서 내용에 대한 질문만 만들어줘"
                )
            )
        ]
    else:  # single_pdf
        # 단일 PDF용 페르소나 (기상재해 대응기술 가이드북 특화)
        return [
            Persona(
                name="WeatherDisasterFarmer",
                role_description=(
                    "나는 한국의 농업에 종사하는 농부로, 2025년 기상재해 대응기술 가이드북을 집중적으로 학습하고 있다. "
                    "이 가이드북에 나와있는 20가지 주요 작물의 기상재해 대응 기술에 대해 깊이 알고 싶어한다. "
                    "특히 태풍, 폭우, 가뭄, 한파, 폭염, 서리 등 각종 기상재해에 대한 구체적인 대응 방법과 "
                    "작물별 특성에 맞는 재해 예방 및 복구 기술에 대해 질문한다. "
                    "가이드북에 제시된 단계별 대응 매뉴얼, 시기별 관리 방법, 피해 정도별 조치사항 등을 "
                    "실제 농장 상황에 적용할 수 있도록 구체적이고 실용적인 정보를 원한다. "
                    "질문과 정답은 반드시 이 가이드북의 내용을 기반으로 하되, 실제 농업 현장에서 바로 적용 가능한 수준의 구체적인 답변을 요구한다."
                    "이게 무슨 문서인지에 대한 질문은 만들지 말고 문서 내용에 대한 질문만 만들어줘"
                )
            )
        ]

# ===================== 골든셋 생성 =====================
def generate_golden_set(documents: list[Document], mode: str = "multi_pdf"):
    # 더 강력한 LLM 사용 (질문 품질 향상)
    generator_llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,  # 창의성과 일관성의 균형
        max_tokens=2000   # 더 긴 답변 생성
    ))
    generator_embeddings = embedding_factory("openai", model="text-embedding-3-large")

    personas = get_personas(mode)
    transforms = [HeadlineSplitter(), NERExtractor()]

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=personas
    )

    # 안정성: SingleHop만 사용 (단일 페르소나)
    query = SingleHopSpecificQuerySynthesizer(llm=generator_llm)
    # 한국어 프롬프트 설정 (동기 방식)
    try:
        # adapt_prompts를 동기적으로 호출
        import asyncio
        prompts = asyncio.run(query.adapt_prompts("ko", llm=generator_llm))
        query.set_prompts(**prompts)
    except Exception as e:
        print(f"⚠️ 한국어 프롬프트 설정 실패, 기본 프롬프트 사용: {e}")
        # 기본 프롬프트로 폴백
        query.set_prompts()

    dataset = generator.generate_with_langchain_docs(
        documents,
        testset_size=TARGET_QUESTIONS,
        transforms=transforms,
        query_distribution=[(query, 1.0)]
    )
    
        # 🔥 관련성 높은 context만 추려내는 함수
    def get_top_k_contexts(query: str, documents: list[Document], embedder, k: int = 5) -> list[str]:
        query_vec = embedder.embed_query(query)
        doc_vecs = embedder.embed_documents([doc.page_content for doc in documents])
        scores = np.dot(doc_vecs, query_vec) / (
            np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
        )
        top_idx = np.argsort(scores)[::-1][:k]
        return [documents[i].page_content for i in top_idx]

    eval_dataset = dataset.to_evaluation_dataset()

    results = []
    for i, sample in enumerate(eval_dataset[:TARGET_QUESTIONS]):
        print(f"\n질문 {i+1}")
        print("  Question:", sample.user_input)
        print("  Ground Truth:", sample.reference)
        filtered_contexts = get_top_k_contexts(sample.user_input, documents, generator_embeddings, k=5)
        print("  Contexts:", filtered_contexts)
        results.append({
            "question": sample.user_input,
            "ground_truth": sample.reference,
            "contexts": filtered_contexts  # 필터링된 context 사용
        })

    # 저장 (타임스탬프 추가) - farmer/disaster/data 폴더에 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_multi" if mode == "multi_pdf" else "_single"
    
    # farmer/disaster/data 디렉토리 생성 및 파일 경로 설정
    data_dir = "./farmer/disaster/data"
    os.makedirs(data_dir, exist_ok=True)
    
    csv_file = os.path.join(data_dir, f"golden_dataset_open{suffix}_{timestamp}.csv")
    jsonl_file = os.path.join(data_dir, f"golden_dataset_open{suffix}_{timestamp}.jsonl")
    
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ {csv_file} / {jsonl_file} 저장 완료")
    print(f"📊 생성된 골든셋 개수: {len(results)}")

# ===================== 단일 PDF 처리 =====================
def process_single_pdf_file(pdf_path: str) -> list[Document]:
    """단일 PDF 파일 처리"""
    print(f"📄 단일 PDF 처리: {pdf_path}")
    docs = process_single_pdf(pdf_path)
    print(f"📚 {len(docs)}개 Document 생성")
    return docs

# ===================== 실행 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("🔧 RAGAS 골든셋 생성기")
    print("=" * 50)
    print("0: 전체 PDFs 처리 (disaster/pdfs 폴더)")
    print("1: 단일 PDF 처리 (기상재해 대응기술 가이드북)")
    print("=" * 50)
    
    while True:
        try:
            mode_input = input("모드를 선택하세요 (0 또는 1): ").strip()
            if mode_input == "0":
                print("\n📁 전체 PDFs 처리 모드")
                docs = load_documents(PDF_DIR, IMAGE_DIR)
                if docs:
                    generate_golden_set(docs, "multi_pdf")
                break
            elif mode_input == "1":
                print("\n📄 단일 PDF 처리 모드")
                if os.path.exists(SINGLE_PDF_PATH):
                    docs = process_single_pdf_file(SINGLE_PDF_PATH)
                    if docs:
                        generate_golden_set(docs, "single_pdf")
                else:
                    print(f"❌ 파일을 찾을 수 없습니다: {SINGLE_PDF_PATH}")
                break
            else:
                print("❌ 0 또는 1을 입력해주세요.")
        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            break
