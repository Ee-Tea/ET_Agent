# -*- coding: utf-8 -*-
"""
정보처리기사 'generator 에이전트' 컨텍스트 기반 RAGAS 골든셋 생성기
- RAGAS TestsetGenerator로 샘플 수/분포 확보
- question: "문제 생성 요청" 프롬프트(문제 수 = 1~N(≤100) 랜덤)
- ground_truth: [{question, options[4]}, ...]  # 정답/해설 없음
- contexts: generator의 create_context_from_documents 결과 '전량'을 리스트 1원소로 저장

환경변수(.env)
- OPENAI_API_KEY
- OPENAI_BASE_URL (선택)
- RAGAS_TARGET_Q              : 샘플 수(기본 50)
- RAGAS_LANG                  : ko (기본)
- MILVUS_SUBJECTS             : 콤마구분
- MILVUS_TOPK_CONCEPTS        : 기본 20
- MILVUS_TOPK_PROBLEMS        : 기본 30
- OUT_DIR                     : 기본 teacher/agents/TestGenerator/goldensets
- EXAM_JSON_DIR               : 로컬 폴백
- MAX_Q_PER_REQUEST           : 한 샘플에서 생성할 최대 문제 수(기본 20, 상한 100)
- GLOBAL_RANGE_PROB           : 과목지정 대신 전체범위로 요청할 확률(0~1, 기본 0.5)

pip install ragas datasets langchain_text_splitters langchain-openai langchain-huggingface pymilvus pandas
"""

import os, re, json, glob, random, sys
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# RAGAS
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.testset.persona import Persona
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.embeddings import HuggingFaceEmbeddings

# generator 에이전트 동일 헬퍼
from common.milvus_helpers import (
    search_milvus_documents_by_subject,
    create_context_from_documents,
)
try:
    from common.milvus_helpers import get_milvus_connection_info
except Exception:
    get_milvus_connection_info = None

# ---------------- 설정 ----------------
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY가 없습니다. .env를 확인하세요."); sys.exit(1)

TARGET_Q             = int(os.getenv("RAGAS_TARGET_Q", "50"))
RAGAS_LANG           = (os.getenv("RAGAS_LANG", "ko") or "ko").strip().lower()
TOPK_CONCEPTS        = int(os.getenv("MILVUS_TOPK_CONCEPTS", "20"))
TOPK_PROBLEMS        = int(os.getenv("MILVUS_TOPK_PROBLEMS", "30"))
OUT_DIR              = os.getenv("OUT_DIR", os.path.join("teacher","agents","TestGenerator","goldensets"))
EXAM_JSON_DIR        = os.getenv("EXAM_JSON_DIR", os.path.join("teacher","exam","parsed_exam_json"))
MAX_Q_PER_REQUEST    = min(100, int(os.getenv("MAX_Q_PER_REQUEST", "20")))  # 안전 상한 (기본 20, 최대 100)
GLOBAL_RANGE_PROB    = max(0.0, min(1.0, float(os.getenv("GLOBAL_RANGE_PROB", "0.5"))))

DEFAULT_SUBJECTS = ["소프트웨어설계","소프트웨어개발","데이터베이스구축","프로그래밍언어활용","정보시스템구축관리"]
SUBJECTS = [s.strip() for s in os.getenv("MILVUS_SUBJECTS","").split(",") if s.strip()] or DEFAULT_SUBJECTS

random.seed(42)

# ---------------- 유틸 ----------------
_ZWS = "\u200b\u200c\u200d\ufeff"
def clean_text(s: str) -> str:
    s = str(s or "").replace(_ZWS, "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=120, separators=["\n\n","\n"," ",""])
    out=[]
    for d in docs:
        if len(d.page_content) > 1800: out.extend(splitter.split_documents([d]))
        else: out.append(d)
    return [x for x in out if len(x.page_content) >= 60]

# ---- 로컬 폴백 로더 ----
def load_exam_docs(root_dir: str) -> List[Document]:
    paths = glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)
    docs: List[Document] = []
    def make_doc(item: Dict[str,Any], default_subject: str):
        q=item.get("question") or item.get("item_title") or ""
        opts=item.get("options") or []
        ans=item.get("answer"); exp=item.get("explanation") or item.get("content") or ""
        subj=item.get("subject") or default_subject or "미지정"
        ans_txt=None
        if isinstance(ans,str) and ans.strip().isdigit():
            idx=int(ans.strip())-1
            if isinstance(opts,list) and 0<=idx<len(opts): ans_txt=str(opts[idx])
        if not ans_txt: ans_txt=str(ans) if ans is not None else ""
        q=clean_text(q); opts=[clean_text(o) for o in (opts if isinstance(opts,list) else [])]
        exp=clean_text(exp); subj=clean_text(subj)
        if not q: return None
        lines=[f"[과목] {subj}", f"{q}"]
        if opts: lines.append("; ".join(opts))
        if ans_txt: lines.append(f"정답: {ans_txt}")
        if exp: lines.append(f"해설: {exp}")
        pc="\n".join(lines).strip()
        if len(pc)<40: return None
        return Document(page_content=pc, metadata={"subject":subj,"source":"local_json"})
    for p in paths:
        try:
            data=json.load(open(p,"r",encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ JSON 로드 실패: {p} ({e})"); continue
        if isinstance(data,dict):
            subj=data.get("subject"); items=data.get("items")
            if isinstance(items,list):
                for it in items:
                    if isinstance(it,dict):
                        d=make_doc(it,subj); 
                        if d: docs.append(d)
            else:
                d=make_doc(data,subj); 
                if d: docs.append(d)
        elif isinstance(data,list):
            for it in data:
                if isinstance(it,dict):
                    d=make_doc(it,it.get("subject")); 
                    if d: docs.append(d)
    print(f"📚 로컬 exam JSON 문서: {len(docs)}개")
    return docs

# ---- Milvus 연결 ----
def resolve_milvus_data() -> Dict[str,Any]:
    if get_milvus_connection_info is not None:
        try:
            md=get_milvus_connection_info()
            if md and md.get("connection_status",False):
                print("✅ Milvus 연결(get_milvus_connection_info) 성공"); return md
            else:
                print("⚠️ get_milvus_connection_info 유효하지 않음 → 직접 연결 시도")
        except Exception as e:
            print("⚠️ get_milvus_connection_info 실패:",e)
    try:
        from pymilvus import connections
        uri=os.getenv("MILVUS_URI","http://localhost:19530")
        user=os.getenv("MILVUS_USERNAME"); pwd=os.getenv("MILVUS_PASSWORD"); token=os.getenv("MILVUS_TOKEN")
        kwargs={"uri":uri}
        if token: kwargs["token"]=token
        else:
            if user: kwargs["user"]=user
            if pwd: kwargs["password"]=pwd
        connections.connect(alias="default", **kwargs)
        print(f"✅ Milvus 직접 연결 성공: {uri}")
        return {"connection_status":True,"alias":"default"}
    except Exception as e:
        print("❌ Milvus 직접 연결 실패:",e)
        return {"connection_status":False}

# ---- 컨텍스트 수집 ----
@dataclass
class SubjectBundle:
    subject: str|None
    docs: List[Document]
    merged_context_full: str  # 전량

def fetch_subject_bundle(subject: str, milvus_data: Dict[str,Any]) -> SubjectBundle:
    concept_docs = search_milvus_documents_by_subject(milvus_data,"concepts",subject,TOPK_CONCEPTS)
    problem_docs = search_milvus_documents_by_subject(milvus_data,"problems",subject,TOPK_PROBLEMS)
    docs = concept_docs + problem_docs
    merged = create_context_from_documents(docs) if docs else ""
    return SubjectBundle(subject=subject, docs=docs, merged_context_full=merged)

# ---- 사용자 프롬프트(골든셋 question) ----
from typing import Tuple

def build_user_prompt(subject: str | None) -> Tuple[str, int]:
    # 1~min(100, MAX_Q_PER_REQUEST)에서 랜덤 선택
    k = random.randint(1, max(1, min(100, MAX_Q_PER_REQUEST)))
    # 과목이 있어도 일정 확률로 전체 범위를 사용
    use_global = (subject is None) or (random.random() < GLOBAL_RANGE_PROB)

    if use_global:
        text = (
            f"정보처리기사 전체 범위에서 객관식 {k}문제 만들어줘. "
            f"각 문항은 보기 4개만 제공해."
        )
    else:
        # subject가 유효하지 않으면 랜덤 과목 선택
        subj = subject if (subject in SUBJECTS) else random.choice(SUBJECTS)
        text = (
            f"정보처리기사 {subj} 과목 객관식 {k}문제 만들어줘. "
            f"각 문항은 보기 4개만 제공해."
        )
    return text, k

# ---- ground_truth 생성용 프롬프트 ----
def build_generation_prompt(context_full: str, subject: str|None, k: int) -> str:
    subject_part = f"{subject} 과목" if subject else "전체 범위"
    return f"""당신은 정보처리기사 출제 전문가입니다. 아래 컨텍스트를 근거로 {subject_part}에서 객관식 문제 {k}개를 생성하세요.

반드시 다음 형식을 지키세요:
1) 출력은 오직 JSON 하나(배열)만.
2) JSON 스키마: 
[
  {{"question":"문항 본문","options":["보기1","보기2","보기3","보기4"]}},
  ...
]  # 길이 = {k}
3) 각 options는 정확히 4개, 중복/유사 반복 금지.
4) 정답과 해설은 포함하지 마세요(키 자체를 만들지 마세요).
5) 모든 문항은 컨텍스트의 사실에 기반.

[컨텍스트 시작]
{context_full}
[컨텍스트 끝]
"""

def parse_mcq_list_json(text: str, k: int) -> List[Dict[str,Any]]:
    content=text.strip()
    m=re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if m: content=m.group(1).strip()
    # 배열 블록 추출
    m2=re.search(r"\[\s*[\s\S]*\s*\]", content)
    if m2: content=m2.group(0)
    data=json.loads(content)
    if not isinstance(data, list):
        raise ValueError("배열 JSON이 아님")
    out=[]
    for item in data:
        q=clean_text((item or {}).get("question"))
        opts=[clean_text(o) for o in ((item or {}).get("options") or []) if o]
        if not q or len(opts)!=4:
            continue
        # 정답/해설 키 제거(혹시 들어오면)
        item_norm={"question":q,"options":opts}
        out.append(item_norm)
        if len(out)>=k:
            break
    if len(out)<max(1,k//2):  # 최소 절반은 확보(너무 부족하면 실패 처리)
        raise ValueError(f"생성 문항 수 부족: {len(out)}<{k}")
    return out

# ---------------- 메인 ----------------
def main():
    print("=== RAGAS 기반 골든셋 생성 (랜덤 문제수 + 보기만 + 컨텍스트 포함) ===")
    print(f"TARGET_Q = {TARGET_Q}")
    print(f"SUBJECTS = {SUBJECTS}")
    print(f"TOPK: concepts={TOPK_CONCEPTS}, problems={TOPK_PROBLEMS}")
    print(f"MAX_Q_PER_REQUEST = {MAX_Q_PER_REQUEST} (상한 100)")

    # LLM for RAGAS + 생성용
    base_llm = ChatOpenAI(model=os.getenv("RAGAS_OPENAI_MODEL","gpt-4o-mini"),
                          temperature=0.2, max_tokens=3000,
                          base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    llm_wrapper = LangchainLLMWrapper(base_llm)
    emb = HuggingFaceEmbeddings(model="intfloat/multilingual-e5-large")

    # RAGAS synthesizer
    synth = SingleHopSpecificQuerySynthesizer(llm=llm_wrapper)
    import asyncio
    try:
        prompts = asyncio.run(synth.adapt_prompts(RAGAS_LANG, llm=llm_wrapper))
        synth.set_prompts(**prompts)
    except Exception:
        pass

    # 컨텍스트 준비
    milvus_data = resolve_milvus_data()
    bundles: List[SubjectBundle] = []
    if milvus_data.get("connection_status",False):
        for subj in SUBJECTS:
            b = fetch_subject_bundle(subj, milvus_data)
            if b.docs:
                bundles.append(b)
                print(f"📚 {subj}: 문서 {len(b.docs)}개 / 컨텍스트 길이={len(b.merged_context_full)}")
            else:
                print(f"⚠️ {subj}: 문서 없음")
    else:
        print("⚠️ Milvus 미연결 → 로컬 JSON 폴백")
        local_docs = load_exam_docs(EXAM_JSON_DIR)
        if not local_docs:
            print("❌ 컨텍스트 없음"); sys.exit(1)
        merged = create_context_from_documents(local_docs)
        bundles.append(SubjectBundle(subject=None, docs=local_docs, merged_context_full=merged))

    if not bundles:
        print("❌ 컨텍스트 번들을 구성하지 못했습니다."); sys.exit(1)

    # RAGAS에 전달할 문서 구성(분포 확보용)
    ragas_docs: List[Document] = []
    for b in bundles:
        ragas_docs.append(Document(page_content=b.merged_context_full, metadata={"subject": b.subject or "전체"}))
        ragas_docs.extend(split_docs(b.docs))

    # RAGAS Testset 생성(샘플 수 확보; 실제 ground_truth/contexts는 아래에서 재구성)
    generator = TestsetGenerator(
        llm=llm_wrapper,
        embedding_model=emb,
        persona_list=[Persona(name="ExamGeneratorRequestor", role_description="사용자 관점의 요청문을 생성")]
    )
    testset = generator.generate_with_langchain_docs(
        documents=ragas_docs,
        testset_size=TARGET_Q,
        transforms=[NERExtractor()],
        query_distribution=[(synth, 1.0)],
    )

    # RAGAS 결과 수 만큼, 요구 스키마로 후처리
    rows=[]
    bi=0
    for _ in testset.to_evaluation_dataset():
        b = bundles[bi % len(bundles)]; bi+=1

        # 사용자 요청 프롬프트 + 랜덤 문제 수
        user_prompt_text, k = build_user_prompt(b.subject)

        # 보기만 포함하는 문제 리스트 생성 (컨텍스트 전량 사용)
        gen_prompt = build_generation_prompt(b.merged_context_full, b.subject, k)
        try:
            resp = base_llm.invoke(gen_prompt)
            gt_list = parse_mcq_list_json(getattr(resp,"content",str(resp)), k)
        except Exception as e:
            print(f"⚠️ 문제 리스트 생성 실패({b.subject}): {e}")
            continue

        # === RAGAS 평가 포맷 ===
        rows.append({
            "question": user_prompt_text,        # 요청문
            "ground_truth": gt_list,             # [{question, options[4]}, ...]
            "contexts": [b.merged_context_full], # 전량 1개(리스트)
            "subject": b.subject or "전체",
            "requested_n": k,
            "generated_n": len(gt_list),
        })

    if not rows:
        print("❌ 생성 실패"); sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUT_DIR, f"goldenset_generator_ragas_{ts}.jsonl")
    csv_path   = os.path.join(OUT_DIR, f"goldenset_generator_ragas_{ts}.csv")

    with open(jsonl_path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # CSV는 ground_truth/contexts를 문자열화해 함께 저장
    import pandas as pd
    df = pd.DataFrame([{
        "question": r["question"],
        "ground_truth": json.dumps(r["ground_truth"], ensure_ascii=False),
        "contexts": json.dumps(r["contexts"], ensure_ascii=False),
        "contexts_len": len(r["contexts"][0]),
        "subject": r["subject"],
        "requested_n": r["requested_n"],
        "generated_n": r["generated_n"],
    } for r in rows])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ 저장 완료\n - JSONL: {jsonl_path}\n - CSV  : {csv_path}\n📊 샘플 수: {len(rows)}")

if __name__ == "__main__":
    main()
