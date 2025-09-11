#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_golden_with_llm_cultivation.py

- Milvus(+LangChain)에서 문서를 검색해 OpenAI LLM으로 직접 Q/A 생성
- RAGAS ❌
- 결과를 JSONL 파일로 저장 (golden_cultivation.jsonl 형식)
"""

import os, re, json, random, argparse, hashlib
from typing import List, Dict, Any
from dotenv import load_dotenv

# =============== ENV ===============
load_dotenv()
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agri_disaster_docs")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise SystemExit("❌ OPENAI_API_KEY가 필요합니다.")

# =============== Milvus (검색) ===============
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

lc_embedding = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = Milvus(
    embedding_function=lc_embedding,
    collection_name=COLLECTION_NAME,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 12, "fetch_k": 50, "lambda_mult": 0.5})

# =============== OpenAI LLM ===============
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)

GENERAL_QUERIES = ["태풍 피해", "홍수 피해", "가뭄 피해", "폭염 피해", "한파 피해", "재해 대응", "피해 복구", "재해 예방", "기상재해", "자연재해", "농업 피해", "시설물 피해", "작물 피해", "재해 대비", "응급 대응", "농작물 피해", "농경지 피해", "농업시설 피해", "기상특보", "재해경보", "피해상황", "복구방안", "대응지침"]
CROPS = ["감자","벼","토마토","딸기","고추","옥수수","상추","양파","마늘","배추","사과","배","복숭아","참외","수박","호박","블루베리","잎들깨","여주","유기농쌀"]

def gather_langchain_docs(n_items: int) -> List[Document]:
    topics = GENERAL_QUERIES + CROPS
    search_limit = min(n_items * 30, 500)
    docs: List[Document] = []
    seen = set()

    for topic in random.sample(topics * 5, len(topics) * 5):
        results = retriever.invoke(topic)
        for d in results:
            txt = (getattr(d, "page_content", "") or "").strip()
            if not txt:
                continue
            cleaned = re.sub(r"\s{2,}", " ", txt)
            if len(cleaned) > 2000:
                cleaned = cleaned[:2000]
            key = hashlib.sha256(cleaned[:500].encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)

            title = d.metadata.get("title") or cleaned[:30]
            metadata = {**d.metadata, "title": title}
            docs.append(Document(page_content=cleaned, metadata=metadata))
        if len(docs) >= search_limit:
            break
    return docs

# =============== 재해 관련 노트 추가 ===============
DISASTER_PAT = re.compile(r"(태풍|홍수|가뭄|폭염|한파|재해|피해|복구|대응|기상|경보|특보)")
def add_disaster_note_if_any(row: Dict[str, Any]) -> Dict[str, Any]:
    if DISASTER_PAT.search(" ".join(row.get("contexts", []))):
        row["disaster_note"] = "컨텍스트에 재해대응 관련 정보가 포함되어 있어 참고가 가능합니다."
    else:
        row["disaster_note"] = ""
    return row

# =============== LLM QA 생성기 ===============
def generate_qa_from_doc(doc: Document) -> Dict[str, Any]:
    prompt = f"""
당신은 농업 재해대응 전문가입니다. 아래 문서를 바탕으로 재해대응과 관련된 하나의 질문과 답변을 생성하세요.
- 질문은 재해 피해, 대응 방법, 복구 방안 등 재해대응과 관련된 내용으로 작성
- 답변은 문서에 근거해 간결하고 실용적으로 작성
- JSON 형식으로만 출력: {{ "question": "...", "answer": "..." }}

문서 내용:
{doc.page_content}
"""
    resp = llm.invoke(prompt)
    try:
        qa = json.loads(resp.content)
    except Exception:
        qa = {"question": f"{doc.metadata.get('title','이 문서')}에 대해 설명해 주세요.", 
              "answer": doc.page_content[:200]}
    return {
        "question": qa.get("question", "").strip(),
        "ground_truth": qa.get("answer", "").strip(),
        "contexts": [doc.page_content]
    }

# =============== main ===============
def main(num_items: int, out_path: str):
    docs = gather_langchain_docs(num_items)
    if not docs:
        print("❌ 문서를 수집하지 못했습니다.")
        return

    print(f"📄 수집된 문서 수: {len(docs)}")
    print(f"📄 첫 번째 문서 미리보기: {docs[0].page_content[:200]}...")

    rows = []
    for d in docs[:num_items]:
        qa = generate_qa_from_doc(d)
        qa = add_disaster_note_if_any(qa)
        rows.append(qa)

    # ✅ JSONL 저장 (golden_cultivation.jsonl 형식)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            # JSONL 형식으로 저장 (각 줄이 하나의 JSON 객체)
            json_obj = {
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "contexts": r["contexts"]  # 배열 형태로 유지
            }
            # disaster_note가 있으면 추가
            if r.get("disaster_note"):
                json_obj["disaster_note"] = r["disaster_note"]
            
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    print(f"✅ JSONL 저장 완료: {out_path} (생성 {len(rows)}건)")
    print("  - 필드: question, ground_truth, contexts (+ disaster_note 선택)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OpenAI LLM 기반 직접 Q/A 생성기 (JSONL 저장)")
    ap.add_argument("--num-items", type=int, default=2)
    ap.add_argument("--out", default="golden_disaster.jsonl")
    args = ap.parse_args()
    main(args.num_items, args.out)
