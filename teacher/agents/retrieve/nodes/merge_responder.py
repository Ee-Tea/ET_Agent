import os
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# LLM 모델 설정을 환경변수에서 가져오기
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# Grok이 OpenAI 호환 API를 제공한다고 가정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
  raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 또는 환경변수에 키를 설정하세요.")

def merge_context(wiki, ddg, milvus="") -> str:
    """
    여러 개의 context chunk를 하나의 문자열로 병합합니다.
    """
    # MilvusDB 결과가 있는지 확인
    milvus_section = ""
    if milvus and milvus.strip():
        milvus_section = f"""

    [MilvusDB 벡터 검색 결과]
    {milvus}"""
    
    merge_prompt = f"""
    다음 검색 결과들을 읽고 중복되는 내용을 제거한 후, 간결하게 통합하여 설명해 주세요.

    [Wikipedia 결과]
    {wiki}

    [DuckDuckGo 결과]
    {ddg}{milvus_section}

    ==> 중복을 제거한 요약 결과:
    """
    
    llm = ChatOpenAI(
        api_key=openai_api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        model=OPENAI_LLM_MODEL,
        temperature=0.5,
        max_tokens=LLM_MAX_TOKENS
    )
    
    response = llm.invoke([HumanMessage(content=merge_prompt)])
    return response.content.strip()

def generate_answer(prompt: str, context: str) -> str:
    # context = "\n\n".join(context_chunks[:10])  # context 최대 5개만 사용
    full_prompt = f"""당신은 중학교 교사입니다. 다음의 데이터를 바탕으로 사용자의 질문에 답변해야 합니다.

    {context}

    위 참고자료를 바탕으로 사용자의 질문에 친절하고 이해하기 쉽게 답변해주세요.
    질문: {prompt}
    답변:"""
    
    llm = ChatOpenAI(
        api_key=openai_api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        model=OPENAI_LLM_MODEL,
        temperature=0.5,
        max_tokens=LLM_MAX_TOKENS
    )
    
    response = llm.invoke([HumanMessage(content=full_prompt)])
    return response.content.strip()