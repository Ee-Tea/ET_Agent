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
NUM_QUESTIONS_TO_GENERATE = 50 # <-- 생성할 질문 개수를 여기서 수정하세요.
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
당신은 주어진 [컨텍스트]를 바탕으로, 사용자가 실제 LLM 채팅에서 했을 법한 '자연스러운 작물 추천 답변'을 역추론하여 생성하는 AI입니다.
[컨텍스트]는 어떤 질문에 대한 '모범 답변'입니다. 그에 어울리는 자연스러운 '사용자 질문'을 만드세요.

[핵심 규칙]
1) 반드시 '추천'이라는 단어가 포함된, 특정 환경/조건에 맞는 '작물 추천'을 요청하는 질문이어야 합니다.
2) 질문은 현실적인 상황(예: 해안가, 고랭지, 여름/겨울, 초보자, 옥상 정원, 작은 텃밭, 남부/북부 지방, 건조/다우 지역 등)을 배경으로 하세요.
3) 특정 작물명(예: 포도, 사과 등), 묘목/품종/씨앗 구매, 가격,병해충 등 '작물에 해당하는 내용'은 절대 포함하지 마세요.
4) 재배기술·병해충·영양성분·약효·시장성·가격 등 '추천 자체'와 직접 관련 없는 주제는 포함하지 마세요.
5) 메타 표현 금지: '작물명', '작물 묘목', '작물과 관련된 내용', '~작물 키우기 좋은 지역' 같은 템플릿 문구를 그대로 쓰지 마세요.
6) 질문은 [컨텍스트] 정보만으로 자연스럽게 답변될 수 있어야 합니다.(컨텍스트 밖의 정보 요구 금지).
7) 문체/형식: 한 문장, 문장 끝은 '~주세요' 또는 '~부탁드립니다'로 마무리합니다.
8) '올리브', '아보카도', '망고', '과일명', '채소명', '과수 작물 묘목' 등 질문에 포함하지 마세요.
9) 질문은 어색한 부분 없이 자연스러워야 합니다.

[출력 형식]
- {num_questions}개의 서로 다른 질문을 생성하세요.
- 추가 설명·번호·불릿 없이, 각 질문을 구성하세요.
- 사용자에게 실제로 도움이 될 만한, 구체적이고 현실적인 질문이어야 합니다.
- 모든 줄은 위 규칙을 만족해야 합니다.

[검증 체크리스트]
- 각 질문에 '추천'이 포함되어 있습니까?
- 특정 환경/조건(예: 해안가, 고랭지, 여름/겨울, 초보자, 옥상 정원, 작은 텃밭, 남부/북부 지방, 건조/다우 지역 등)을 배경으로 했습니까?
- 특정 작물명이나 '묘목/품종/씨앗/가격' 같은 금지어가 없습니까?
- 재배기술/병해충/영양/약효/시장성/가격 등의 금지 주제가 없습니까?
- 한 문장으로 끝나며 '~주세요' 또는 '~부탁드립니다'로 마무리했습니까?
- '올리브', '아보카도', '망고', '과일', '채소', '작물 묘목' 등 질문내용에 포함이 되있나요?

[컨텍스트]
{chunk_text}

[출력]
"""
question_prompt = ChatPromptTemplate.from_messages([
    ("system", QUESTION_SYSTEM_PROMPT),
    ("user", "[컨텍스트]:\n{chunk_text}\n\n[추론된 질문]:")
])

# 2. 주어진 컨텍스트와 질문으로 모범 답안을 요약/정리하는 프롬프트
GT_SYSTEM_PROMPT = """
당신은 주어진 [질문]에 대해 [컨텍스트]만을 근거로 '질문과 직접 연관된 작물 추천 답변'을 생성하는 AI입니다.
[컨텍스트]에는 로컬 문서와 웹검색 요약 등 유사 데이터가 포함될 수 있으며, 답변은 반드시 [컨텍스트]에 있는 사실만 사용해야 합니다.

[핵심 규칙]
1) [컨텍스트]의 사실만 사용하세요. 추측/상식 추가 금지.
2) [작물명 정규화] 품종명·숫자코드·외래어 표기는 제거하고 일반 작물명만 사용하세요.
   예: 캠벨얼리/101-14/3309/거봉 → 포도, 홍로/JM.1 → 사과, 큰알보리1호 → 보리, 감귤 → 귤, 참깨 → 깨
3) [중복 제거] 같은 작물은 한 번만 언급하세요.
4) [여러 작물 허용] 질문에 맞으면 1~3개 작물을 제시하되, 각 작물마다 [컨텍스트] 근거를 요약한 이유를 붙이세요.
5) [출력 형식] 각 줄은 다음 형식으로만 작성하세요.
   "<작물명>을/를 추천드립니다. <이유(컨텍스트 근거 요약)>"
6) 불릿/번호/마크다운/여러 문단 금지, 각 작물마다 정확히 한 줄.
7) 거절/무추천 금지: '추천이 어렵다' 등의 문장을 출력하지 말고, [컨텍스트]에서 가장 근거가 강한 후보를 보수적으로 1개 이상 제시하세요.
8) 답변 생성은 3~5문장으로 작성하세요.
9) 질문과 관련된 작물을 추천하고, 해당하는 추천 이유를 질문의 의도에 맞게 구체적으로 작성하세요.

[선정 기준(내부 가이드, 출력 금지)]
- 질문의 환경/조건과 일치하는 내성(가뭄/습해/염분/저온/고온/그늘 등)·적응성·초보자 친화성 기술이 있는 작물 우선.
- 명시적·강한 근거 > 약한 근거 > 일반적 기술 순으로 우선순위.
- 근거가 유사하면 1~2개로 축약.

[검증 체크리스트]
- 모든 줄이 "추천드립니다" 형식이며, 이유가 [컨텍스트]에 실재합니까?
- 작물명 일반화·중복 제거가 되었습니까?
- 추측/외부지식 보강이 없습니까?
- 불릿/번호/여러 문단이 없습니까?

[질문]
{question}

[컨텍스트]
{contexts}

[모범 답안]
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
        question = llm_question.invoke(
        question_prompt.format_prompt(chunk_text=context_text, num_questions=1)  # ← 여기!
        ).content.strip()
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