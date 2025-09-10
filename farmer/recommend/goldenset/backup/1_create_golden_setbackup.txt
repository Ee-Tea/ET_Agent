# 1_create_golden_set.py (v6.0: 최종 로직 수정 버전)

import os
import re
import logging
import random
from datetime import datetime
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import fitz
import torch
from langchain_community.cache import InMemoryCache
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import langchain

# ==================== 설정: 생성할 질문 개수 및 PDF 경로 ====================
NUM_QUESTIONS_TO_GENERATE = 10  # <-- 생성할 질문 개수를 여기서 수정하세요.
PDF_INPUT_DIR = os.getenv("PDF_INPUT_DIR", r"C:\Rookies_project\pdf")
# =======================================================================

# ==================== 설정 (공통) ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("create_golden_set")
langchain.llm_cache = InMemoryCache()
load_dotenv(find_dotenv())

# API 키 및 모델 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY가 .env 파일에 필요합니다.")

# 전역 객체
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask"), model_kwargs={"device": device})
llm_question = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.7, api_key=OPENAI_API_KEY)
llm_gt = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)

# ==================== 프롬프트 (새로운 로직에 맞게 수정) ====================
# 1. 주어진 컨텍스트(정답의 재료)에서 질문을 생성하는 프롬프트
QUESTION_SYSTEM_PROMPT = """
당신은 주어진 [컨텍스트] 내용을 바탕으로, 자연스러운 작물 추천에 해당하는 질문을 생성하는 AI입니다.
[컨텍스트]는 어떤 질문에 대한 '정답'입니다. 이 '정답'을 보고, 사용자가 어떤 질문을 했을지 역으로 추론하여 질문을 한 줄로 만드세요.

**규칙:**
1.  **질문 의도 명확화**: 질문은 **오직 작물 추천**을 요청하는 내용이어야 합니다. 재배 기술, 수확량, 병충해 등 기타 정보에 대한 질문은 절대 생성하지 마세요.
2.  **상황 부여**: '고랭지', '해안가', '초보자' 등 질문자가 처한 상황을 상상하여 자연스러운 질문을 만드세요.
3.  **형식 준수**: '추천'이라는 단어를 포함한 한글 의문문 한 문장만, 다른 기호 없이 깔끔하게 출력합니다.
4.  **중복 방지**: 이전에 생성된 질문과 중복되지 않도록 주의하세요.
6.  **품명,품종 언급 금지** : 질문에 특정 작물 품명 언급 금지(예: "배나무 품명 추천" X, "배" O)
7. ** 작물명 언급 금지** : [컨텍스트]에 직접 언급된 작물 이름(예: '여주,토마토,배,망고 등')을 포함하지 마세요.

**[좋은 질문 예시]**
- 고랭지 지역에서 키울만한 작물 추천 해주세요. 
- 해안가 근처에서 살고 있는데 키울 수 있는 작물 추천 해주세요.
- 비가 많이 오는 지역에서 키울만한 작물 추천 해주세요.
- 작은 텃밭이 있는데 키울만한 작물 추천 해주세요.
- 농사 경험이 전혀 없는 초보자인데 어떤 작물을 키우면 좋을지 추천 부탁드립니다.
"""
question_prompt = ChatPromptTemplate.from_messages([
    ("system", QUESTION_SYSTEM_PROMPT),
    ("user", "[컨텍스트]:\n{chunk_text}\n\n[추론된 질문]:")
])

# 2. 주어진 컨텍스트와 질문으로 모범 답안을 요약/정리하는 프롬프트
GT_SYSTEM_PROMPT = """
당신은 주어진 [컨텍스트]를 사용하여, 주어진 [질문]에 대한 '작물 추천 모범 답안'을 생성하는 AI입니다.

**규칙:**
1.  **요약 및 정리**: [컨텍스트] 내용 중에서 [질문]에 대한 답변이 될 부분만 골라 5~8 문장으로 간결하게 요약하고 정리합니다. (컨텍스트-답변 일치도 100% 목표)
2.  **근거 기반**: [컨텍스트]에 없는 내용은 절대 추가하지 마세요.(할루시네이션 방지)**
3. **작물 추천 특화**: 작물 추천에 관련된 정보로 답변을 구성하세요.
4. **형식 준수**: 마크다운, 불릿포인트, 번호매기기 등은 사용하지 말고, 일반 문장으로만 작성하세요. (특수문자, 기호 사용 금지)
5. **문장 간결화**: 문장이 너무 길면 적절히 나누어 가독성을 높이세요.
6. **중복 제거**: 동일한 내용이 반복되지 않도록 주의하세요.
7. **존댓말 사용**: 답변은 항상 존댓말로 작성하세요.
8. **불필요한 서론 제거**: "제공된 정보에 따르면", "컨텍스트를 바탕으로" 등 불필요한 서론은 제거하세요.
"""
gt_prompt = ChatPromptTemplate.from_messages([
    ("system", GT_SYSTEM_PROMPT),
    ("user", "[질문]: {question}\n\n[컨텍스트]:\n{contexts}\n\n[모범 답안]:")
])

# ==================== 핵심 함수 ====================
def extract_chunks_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    try:
        doc = fitz.open(pdf_path)
        if doc.is_encrypted: return []
        full_text = [page.get_text("text", clip=fitz.Rect(0, page.rect.height * 0.1, page.rect.width, page.rect.height * 0.9)) for page in doc]
        text = re.sub(r'\s+', ' ', "\n".join(full_text)).strip()
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) < 1200: buf += s + " "
            else:
                if len(buf) > 800: chunks.append({"source": os.path.basename(pdf_path), "text": buf.strip()})
                buf = s + " "
        if buf and len(buf.strip()) > 800: chunks.append({"source": os.path.basename(pdf_path), "text": buf.strip()})
    except Exception as e:
        logger.warning(f"PDF 처리 실패: {pdf_path} - {e}")
    return chunks

def collect_pdf_chunks(input_dir: str) -> List[Dict[str, str]]:
    pdfs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    all_chunks = []
    for p in pdfs: all_chunks.extend(extract_chunks_from_pdf(p))
    return all_chunks

# ==================== 메인 실행 로직 (새로운 로직 적용) ====================
def main():
    # random.seed(42) # 매번 다른 질문을 원하면 주석 처리
    logger.info("PDF에서 청크 수집을 시작합니다...")
    chunks = collect_pdf_chunks(PDF_INPUT_DIR)
    if not chunks:
        logger.error("유효한 청크를 찾지 못했습니다.")
        return

    num_to_generate = NUM_QUESTIONS_TO_GENERATE
    logger.info(f"{len(chunks)}개의 청크에서 {num_to_generate}개를 샘플링합니다.")
    sample_chunks = chunks if num_to_generate >= len(chunks) else random.sample(chunks, num_to_generate)
    
    golden_items = []
    for i, chunk_info in enumerate(sample_chunks, 1):
        context_text = chunk_info["text"]
        
        # 1. 컨텍스트(정답 재료)에서 질문을 생성
        logger.info(f"[{i}/{num_to_generate}] 컨텍스트 기반 질문 생성 중...")
        question = llm_question.invoke(question_prompt.format_prompt(chunk_text=context_text)).content.strip()
        logger.info(f"   -> 생성된 질문: {question[:50]}...")

        # 2. 동일한 컨텍스트와 생성된 질문으로 모범 답안 생성
        logger.info(f"[{i}/{num_to_generate}] 모범 답안 생성 중...")
        ground_truth = llm_gt.invoke(gt_prompt.format_prompt(question=question, contexts=context_text)).content.strip()
        
        # 3. 고품질 데이터만 최종 저장
        if question and ground_truth:
            logger.info(f"   -> 생성 완료: 질문, 모범 답안")
            golden_items.append({
                "question": question,
                "ground_truth": ground_truth,
                "gt_contexts": str([context_text]), # 원본 청크를 컨텍스트로 저장
                "source_pdf": chunk_info["source"]
            })
        else:
            logger.warning(f"-> 데이터 생성 실패로 건너뜀.")

    if not golden_items:
        logger.error("생성된 골든셋이 없습니다. 프로그램을 종료합니다.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"1_golden_set_{timestamp}.csv"
    
    df = pd.DataFrame(golden_items)
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    logger.info(f"총 {len(golden_items)}개의 **고품질 골든셋**을 생성하여 '{output_filename}' 파일로 저장했습니다.")

if __name__ == "__main__":
    main()