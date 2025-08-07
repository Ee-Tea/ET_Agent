# ===== hybrid_rag_embed.py =====
import os
import io
import re
import pickle
from glob import glob
from PIL import Image
import numpy as np
import torch
import faiss
import fitz  # PyMuPDF
import easyocr
from sentence_transformers import SentenceTransformer

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 사용 디바이스:", device)

# ✅ 모델 로드
text_model = SentenceTransformer("jhgan/ko-sroberta-multitask", device=str(device))
ocr_reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())

# ✅ 벡터 정규화 함수
def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# ✅ 텍스트 정제 및 분할
def clean_text(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9.,()\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_chunk(chunk, min_len=30, max_len=1200):
    chunk = chunk.strip()
    if not (min_len <= len(chunk) <= max_len):
        return False
    if re.fullmatch(r"[^\uac00-\ud7a3a-zA-Z]+", chunk):
        return False
    if len(set(chunk)) <= 3:
        return False
    return True

def deduplicate_chunks(chunks):
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        key = chunk.strip()
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    return unique_chunks

def chunk_text(text, max_len=1000, overlap=200):
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunk = text[start:end]
        if is_valid_chunk(chunk):
            chunks.append(chunk.strip())
        if end == len(text):
            break
        start += max_len - overlap
    return deduplicate_chunks(chunks)

# ✅ 임베딩 생성 (OCR 텍스트만 사용)
def create_text_embeddings(text_chunks):
    try:
        text_embs = text_model.encode(text_chunks, batch_size=8, convert_to_numpy=True)
        return [(chunk, l2_normalize(emb)) for chunk, emb in zip(text_chunks, text_embs)]
    except Exception as e:
        print(f"❌ 텍스트 임베딩 오류: {e}")
        return []

# ✅ PDF 및 이미지 처리

def process_pdf_page(page, rag_docs, rag_vecs, source_file, page_num):
    try:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        ocr_results = ocr_reader.readtext(np.array(img), detail=1)
        ocr_text = " ".join(text for _, text, conf in ocr_results if conf >= 0.7)
        page_text = page.get_text()
        combined_text = f"{page_text} {ocr_text}"
        text_chunks = chunk_text(combined_text)
        for chunk, emb in create_text_embeddings(text_chunks):
            rag_docs.append({
                "type": "ocr_text",
                "source": source_file,
                "file": f"{os.path.basename(source_file)}_page{page_num + 1}.png",
                "text": chunk,
                "embedding_model": "ko-sroberta-multitask"
            })
            rag_vecs.append(emb)
    except Exception as e:
        print(f"❌ PDF 페이지 처리 오류: {e}")

def process_image_file(path, rag_docs, rag_vecs):
    try:
        img = Image.open(path).convert("RGB")
        ocr_results = ocr_reader.readtext(np.array(img), detail=1)
        ocr_text = " ".join(text for _, text, conf in ocr_results if conf >= 0.7)
        text_chunks = chunk_text(ocr_text)
        for chunk, emb in create_text_embeddings(text_chunks):
            rag_docs.append({
                "type": "ocr_text",
                "source": path,
                "file": os.path.basename(path),
                "text": chunk,
                "embedding_model": "ko-sroberta-multitask"
            })
            rag_vecs.append(emb)
    except Exception as e:
        print(f"❌ 이미지 파일 처리 오류: {e}")

def process_file(path, rag_docs, rag_vecs):
    if path.lower().endswith(".pdf"):
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            process_pdf_page(page, rag_docs, rag_vecs, source_file=path, page_num=i)
    else:
        process_image_file(path, rag_docs, rag_vecs)

if __name__ == "__main__":
    rag_docs, rag_vecs = [], []

    # 🔍 처리할 파일 목록 수집
    files = glob("pdfs/*.pdf") + glob("images/*.png") + glob("images/*.jpg") + glob("images/*.jpeg")
    print(f"📁 총 처리할 파일 수: {len(files)}개")

    for idx, path in enumerate(files):
        print(f"\n📄 ({idx+1}/{len(files)}) 파일 처리 중: {path}")

        before_chunks = len(rag_docs)
        process_file(path, rag_docs, rag_vecs)
        after_chunks = len(rag_docs)

        print(f"   ➕ 생성된 청크 수: {after_chunks - before_chunks}")
        print(f"   ✅ 누적 청크 수: {after_chunks}")

    # 🧠 FAISS 벡터 인덱스 생성
    print("\n📦 FAISS 인덱스 생성 중...")
    matrix = np.array(rag_vecs).astype("float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    print(f"✅ 인덱스에 벡터 {len(rag_vecs)}개 추가 완료")

    # 💾 저장
    faiss.write_index(index, "multimodal_rag.index")
    with open("multimodal_rag.pkl", "wb") as f:
        pickle.dump(rag_docs, f)

    print("\n🎉 저장 완료:")
    print(f"   - 인덱스 파일: multimodal_rag.index")
    print(f"   - 문서 메타데이터: multimodal_rag.pkl")
    print(f"   - 총 저장 청크 수: {len(rag_docs)}")
