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
당신은 주어진 [컨텍스트]의 내용을 바탕으로 사용자의 질문을 역으로 추론하는 AI입니다. [컨텍스트]는 어떤 질문에 대한 '모범 질문'입니다. 이 '모범 질문'을 보고, 사용자가 어떤 질문을 했을지 상상하여 자연스러운 한 문장으로 질문을 만드세요.

**규칙:**
1.  **최우선 규칙: 작물 추천에 집중**: 질문은 오직 **특정 조건이나 환경에 맞는 작물 추천**을 요청하는 내용이어야 합니다. 재배 기술, 수확량, 병충해, 영양 성분 등 기타 정보에 대한 질문은 절대 생성하지 마세요.
2.  **컨텍스트 기반 질문**: 질문은 [컨텍스트]에 있는 정보로만 답변할 수 있어야 합니다. 질문이 답변에 대한 완벽한 '정답'을 유도해야 합니다.
3.  **자연스러운 상황 부여**: 질문자가 처한 상황을 상상하여 '경사지', '비닐피복', '해안가', '초보자' 등 구체적인 조건이나 환경을 질문에 포함하세요.
4.  **특정 작물명 언급 금지**: 질문에 '포도', '키위', '마늘'과 같이 단일 특정 작물명,'작물 나무명', '~재배하기 좋은 지역' 등 직접적으로 언급하지 마세요. 
대신, '경사지', '해안가', '비가림 하우스'와 같은 **재배 환경**을 중심으로 질문을 구성해야 합니다.
5.  **형식 준수**: 질문은 반드시 '~주세요' 또는 '~부탁드립니다'로 끝나는 한글 의문문이어야 합니다. 다른 불필요한 문장이나 기호는 일절 포함하지 마세요.

**[모범 질문]**
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
당신은 주어진 [컨텍스트]만을 근거로 [질문]에 대한 '작물 추천 모범 답안'을 생성하는 AI입니다.

**규칙(필수):**
1) [출처 제한] 답변의 모든 정보는 100% [컨텍스트]에 있는 내용만 사용하세요. 없는 정보는 절대 추가/추측하지 마세요.
2) [작물명 정규화] 품종명·숫자코드·외래어 표기는 제외하고, 일반 작물명만 사용하세요.
   예: 캠벨얼리/101-14/3309/거봉 → 포도, 홍로 → 사과
3) [중복 제거] 같은 작물은 한 번만 언급하세요.
4) [형식 강제] 각 문장은 반드시 “<작물명>을/를 추천드립니다.”로 시작하고, 바로 이어서 [컨텍스트]에 근거한 한 줄 이유를 덧붙이세요.
   예: 포도를 추천드립니다. 캠벨얼리는 초보자도 키우기 쉽습니다.
   예: 사과를 추천드립니다. 홍로는 병충해에 강합니다.
5) [금지] 불릿/번호/마크다운, 기호 금지. 작물명이 아닌 효능·성분명 단독 표기 금지.
6) [톤] 항상 존댓말로, 간결한 한 문장씩 나열하세요.

**검증 체크리스트(출력 직전 자기검사):**
- 모든 문장이 “추천드립니다”를 포함했습니까?
- 작물명은 [컨텍스트]에 직접 등장했고 일반명으로 정규화되었습니까?
- 이유 문구의 모든 근거가 [컨텍스트]에 있습니까?
- 불릿/번호/마크다운 기호가 없습니까?

**[생성 과정 예시]**
- 질문: 가뭄이 심한 지역에서 키울 수 있는 작물 추천 부탁드립니다.
- 컨텍스트: ['적은 면적일때는 퇴비, 볏짚, 산야초 등으로 덮어 토양 수 분을 최대한 보존한다. 가뭄이 심할 경우 물대기가 가능한 줄뿌림 재배지는 물 을 흘려 대고 휴립광산파 재배지는 배수구에만 물을 대준 후 즉시 빼주어 습해 가 없도록 한다. 관수 당시 생육 단계 관수 여부 간장 (cm) 립수 (개) 지엽 엽색도 (4/30) 천립중 (g) 1L중 (g) 불임개체 발생률(%) 수량 (kg/10a) 수량 지수 수잉기 무관수 관수 69 82 27 35 54.7 55.9 34.5 37.3 661 704 19 10 172 252 100 146 출수기 무관수 관수 51 57 16 18 58.5 59.9 44.7 45.3 720 727 0 0 233 288 100 123 * 파종 방법 : 평면세조파(4m 간격 관배수용 고랑 설치), 관수 방법 : 고랑 관수, 시험 품종 : 큰알보리 1호(수잉기), 삼도보리(출수기) 라. 쓰러짐(도복) 피해 보리에서의 도복은 일반적으로 출수기를 전후하여 비, 바람에 의해 일어나기 시 작하는데 쓰러지면 광합성이 저하되고 잎, 줄기 등의 상처로 인하여 호흡이 증대 되며 종실로의 물질축적이 감퇴되어 수량과 품질이 저하된다. 또한 도복이 심해 이삭이 땅에 닿으면 종실이 부패하거나 수발아되어 품질에 치명적 손상을 입히 게 된다. 수확 시 기계수확 작업도 불편하여 작업 시간이 연장되고 종실의 손실 이 많아지며 등숙 기간도 길어져 후작물과의 작부 체계에도 불리해진다. (1) 도복해 발생기작 보리의 키가 커지고 이삭이 무거워지며 줄기와 뿌리가 약해지면 잘 쓰러진다. 출 수 후에는 잎과 줄기의 광합성 산물 및 저장 물질이 대부분 이삭으로 이동하므로 줄기의 조직이 크게 약해진다. 따라서 출수 후 20~30일 경이 도복에 가장 약하 다. 보리의 도복에 의한 수량 감소 정도는 출수 후 10일경(유숙기)에 도복되었을 때 가장 크며, 감소 정도는 심한 경우 40~50%에 달한다. 도복 시기 도복 정도에 따른 수량 감수율(%) 반도복 전도복 전좌절 출수직전 4 8 10 출수기 10 13 15 유숙기 10 15 15 호숙기 4 6 8 황숙기 - 2 2 * \x07반도복 : 45˚정도 쓰러진 상태, 전도복 90˚정도 쓰러진 상태, 전좌절 : 하부절간이 꺾인 상태로 완전 도복된 상태, 위 성적은 같은 방향으로 인위적 도복을 유발시킨 상태의 것이므로 실제 비, 바람에 의한 도복 시에는 피해가 더욱 커질 수 있음.']
- 모범 답안: 보리는 가뭄에 강한 작물로 추천드립니다. 보리는 물대기가 가능한 줄뿌림 재배 방식으로 재배할 수 있어 가뭄 상황에서도 효과적으로 성장할 수 있습니다.

**[질문]:** {question}
**[컨텍스트]:** {contexts}
**[모범 답안]:**
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