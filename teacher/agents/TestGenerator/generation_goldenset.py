# -*- coding: utf-8 -*-
"""
정보처리기사 '문제 생성 에이전트'용 RAGAS 골든셋 생성기 (Context-first + Persona + LLM GT)
- 컨텍스트는 기존 경로/형식 유지: CONCEPT_MASTER_FILE(마스터) + DEFAULT_EXAM_DIR(공식 문제)
- 질문(question): 에이전트 호출용 요청문 고정 템플릿
- 정답/해설 없는 Ground Truth: 컨텍스트 기반 LLM 생성(문제 질문+보기4개만)
"""

import os, re, json, glob, random, uuid, hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 임베딩(HF)
from langchain_huggingface import HuggingFaceEmbeddings
# LLM (에이전트와 동일 인터페이스/ENV)
from langchain_openai import ChatOpenAI
# RAGAS 스타일 요소
from ragas.llms import LangchainLLMWrapper
from ragas.testset.persona import Persona
# LangChain Document (일부 유틸에 사용 가능)
from langchain.schema import Document

load_dotenv()

# -------------------- 경로/환경 --------------------
DEFAULT_EXAM_DIR = os.path.join("teacher", "exam", "parsed_exam_json")
DEFAULT_CONCEPT_DIR = os.path.join("teacher", "concepts")
DEFAULT_OUT_DIR = os.path.join("teacher", "agents", "TestGenerator", "goldensets")

EXAM_DIR = os.getenv("EXAM_DIR", DEFAULT_EXAM_DIR)
CONCEPT_DIR = os.getenv("CONCEPT_DIR", DEFAULT_CONCEPT_DIR)
CONCEPT_MASTER_FILE = os.getenv("CONCEPT_MASTER_FILE", "전체정처기_개념.json")
OUT_DIR = os.getenv("OUT_DIR", DEFAULT_OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_SAMPLES = 50
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)

# ----- 컨텍스트/검색 파라미터 -----
CTX_CONCEPT_CHUNK_CHARS = int(os.getenv("CTX_CONCEPT_CHUNK_CHARS", "1800"))   # 개별 청크 목표 길이
CTX_CONCEPT_TOPK       = int(os.getenv("CTX_CONCEPT_TOPK", "0"))             # 0 => 자동(총 길이 예산 기반)
CTX_CONCEPT_TOTAL_CHARS= int(os.getenv("CTX_CONCEPT_TOTAL_CHARS", "3200"))   # 자동 모드 총 길이 예산
CTX_PROBLEM_CTXK       = 2  

# 임베딩 모델
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

# LLM 설정 (에이전트와 동일 ENV 키 사용)
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.15"))  # 약간의 다양성
LLM_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "1500"))
LLM_TIMEOUT      = int(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES  = int(os.getenv("LLM_MAX_RETRIES", "3"))
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", None)  # 필요 시 사용
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

# 과목 표기
GENERATOR_SUBJECTS_DEFAULT = [
    "소프트웨어 설계", "소프트웨어 개발", "데이터베이스 구축",
    "프로그래밍 언어 활용", "정보시스템 구축 관리",
]
SUBJECT_ALIAS = {
    "소프트웨어설계": "소프트웨어 설계",
    "소프트웨어개발": "소프트웨어 개발",
    "데이터베이스구축": "데이터베이스 구축",
    "프로그래밍언어활용": "프로그래밍 언어 활용",
    "정보시스템구축관리": "정보시스템 구축 관리",
}
SUBJECTS = [s.strip() for s in os.getenv("SUBJECTS", ",".join(GENERATOR_SUBJECTS_DEFAULT)).split(",") if s.strip()]
ALL_SCOPE_TOKEN = "전체 범위"
if ALL_SCOPE_TOKEN not in SUBJECTS:
    SUBJECTS.append(ALL_SCOPE_TOKEN)

# -------------------- 유틸 --------------------
def log(msg: str): print(f"[goldenset] {msg}")

def clean_text(s: Any) -> str:
    if s is None: return ""
    t = str(s)
    t = t.replace("\u200b","").replace("\u200c","").replace("\u200d","").replace("\ufeff","")
    t = t.replace("\r\n","\n")
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()

def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")

def subject_match(s: str, target: str) -> bool:
    if not s or not target: return False
    ns, nt = norm(s), norm(target)
    return nt in ns or ns in nt

def list_json_files(root: str) -> List[str]:
    if not root or not os.path.isdir(root): return []
    files = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
    files += glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)
    return sorted(set(files))

def _normalize_gt(obj: dict) -> Optional[dict]:
    """
    LLM이 준 결과를 엄격히 {question: str, options: [str*4]} 로 강제.
    - 여분 키 제거
    - 옵션 4개 미만/초과, 비문자열, 빈칸 → 실패 처리
    - 옵션 앞의 번호/기호 제거(예: '1) ', '- ', '① ' 등)
    """
    if not isinstance(obj, dict):
        return None

    q = clean_text(obj.get("question", ""))
    opts_raw = obj.get("options", [])
    if not isinstance(q, str) or not isinstance(opts_raw, list):
        return None

    def _strip_bullet(s: str) -> str:
        s = clean_text(s)
        # 번호/기호 패턴 제거
        s = re.sub(r'^\s*(?:\d+\)|\(\d+\)|[①-⑨]|[-*•●▶▷◇])\s*', '', s)
        return s.strip()

    opts = [_strip_bullet(o) for o in opts_raw if isinstance(o, str) and clean_text(o)]
    # 정확히 4개만 허용
    if len(opts) != 4:
        if len(opts) > 4:
            opts = opts[:4]
        else:
            return None

    # 옵션 중복 제거(의미 동일시 위험 시 스킵)
    if len(set(opts)) < 4:
        return None

    if not q:
        return None

    return {"question": q, "options": opts}


# -------------------- 개념 로더 (마스터 파일 우선) --------------------
def _normalize_name(s: str) -> str:
    base = os.path.splitext(os.path.basename(s))[0]
    base = re.sub(r"[\s_\-]+", "", base)
    base = re.sub(r"[()\[\]{}]", "", base)
    return base

def _find_master_concept_path(concept_dir: str, master_name: str) -> Optional[str]:
    exact = os.path.join(concept_dir, master_name)
    if os.path.isfile(exact): return exact
    want = _normalize_name(master_name)
    cands = [fp for fp in glob.glob(os.path.join(concept_dir, "*.json"))]
    for fp in cands:
        if _normalize_name(fp) == want:
            return fp
    tokens = [t for t in re.split(r"[\s_\-]+", os.path.splitext(master_name)[0]) if t]
    for fp in cands:
        key = _normalize_name(fp)
        if all(_normalize_name(t) in key for t in tokens):
            return fp
    # 업로드 폴더 폴백
    mnt = os.path.join("/mnt/data", master_name)
    if os.path.isfile(mnt):
        return mnt
    return None

def load_concept_corpus(concept_dir: str) -> List[Dict[str, str]]:
    """
    기대 구조(마스터):
    {
      "데이터베이스 구축": [
        {"text":"...", "metadata":{"section_title":"..."}}, ...
      ],
      ...
    }
    -> {"content": text, "subject": section_title or top_key, "top_subject": top_key}
    """
    out: List[Dict[str, str]] = []

    mpath = _find_master_concept_path(concept_dir, CONCEPT_MASTER_FILE)
    if mpath:
        log(f"개념 마스터 파일 사용: {mpath}")
        try:
            data = json.load(open(mpath, "r", encoding="utf-8-sig"))
            if isinstance(data, dict):
                for top_key, arr in data.items():
                    if not isinstance(arr, list): continue
                    for it in arr:
                        if not isinstance(it, dict): continue
                        txt = it.get("text") or it.get("content") or ""
                        if not txt: continue
                        meta = it.get("metadata") or {}
                        sect = meta.get("section_title") or top_key
                        txt = clean_text(txt)
                        if txt:
                            out.append({"content": txt, "subject": sect, "top_subject": top_key})
        except Exception as e:
            log(f"⚠️ 마스터 파일 파싱 실패: {e}")

    if out: return out

    log("⚠️ 마스터 미탐/미파싱 → 일반 JSON 폴백")
    for fp in list_json_files(concept_dir):
        try:
            data = json.load(open(fp, "r", encoding="utf-8-sig"))
        except Exception as e:
            log(f"⚠️ 개념 JSON 로드 실패: {fp} ({e})"); continue

        if isinstance(data, dict) and not data.get("items") and "content" not in data:
            for top_key, arr in data.items():
                if not isinstance(arr, list): continue
                for it in arr:
                    if isinstance(it, dict):
                        txt = it.get("text") or it.get("content") or ""
                        meta = it.get("metadata") or {}
                        sect = meta.get("section_title") or top_key
                        txt = clean_text(txt)
                        if txt:
                            out.append({"content": txt, "subject": sect, "top_subject": top_key})
            continue

        if isinstance(data, dict) and isinstance(data.get("items"), list):
            subj = data.get("subject", "")
            for it in data["items"]:
                if isinstance(it, dict) and ("content" in it or "text" in it):
                    _txt = clean_text(it.get("content") or it.get("text"))
                    if _txt:
                        s = it.get("subject", subj)
                        out.append({"content": _txt, "subject": s, "top_subject": s})
            if "content" in data:
                _txt = clean_text(data.get("content"))
                if _txt:
                    s = data.get("subject", subj)
                    out.append({"content": _txt, "subject": s, "top_subject": s})
            continue

        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict) and ("content" in it or "text" in it):
                    _txt = clean_text(it.get("content") or it.get("text"))
                    if _txt:
                        s = it.get("subject","")
                        out.append({"content": _txt, "subject": s, "top_subject": s})
            continue

        if isinstance(data, dict) and ("content" in data or "text" in data):
            _txt = clean_text(data.get("content") or data.get("text"))
            if _txt:
                s = data.get("subject","")
                out.append({"content": _txt, "subject": s, "top_subject": s})

    return [d for d in out if d["content"].strip()]

# -------------------- 문제 로더 (컨텍스트용) --------------------
def load_problem_bank(exam_dir: str) -> List[Dict[str, Any]]:
    bank: List[Dict[str, Any]] = []
    for fp in list_json_files(exam_dir):
        try:
            data = json.load(open(fp, "r", encoding="utf-8-sig"))
        except Exception as e:
            log(f"⚠️ 문제 JSON 로드 실패: {fp} ({e})"); continue

        bundles = data if isinstance(data, list) else [data]
        for b in bundles:
            qs = b.get("questions", []) if isinstance(b, dict) else []
            for q in qs:
                if not isinstance(q, dict): continue
                qtext = clean_text(q.get("question", ""))
                opts = [clean_text(o) for o in (q.get("options") or [])]
                subj = clean_text(q.get("subject", ""))
                if not qtext or len(opts) < 4: continue
                bank.append({
                    "qid": str(uuid.uuid4()),
                    "question": qtext, "options": opts[:4],
                    "subject": subj
                })
    return bank

# -------------------- 임베딩/검색 --------------------
def build_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

_SENT_SPLIT = re.compile(r'(?<=[\.!?])\s+(?=[가-힣A-Za-z0-9\[\(])')
_LIST_BREAK = re.compile(r'\n(?=(?:- |\* |• |● |▶|▷|◇|①|②|③|\(\d+\)|\d+\.) )')

def _split_for_chunks(text: str, max_chars: int) -> List[str]:
    t = text.replace("\r\n", "\n")
    parts = _LIST_BREAK.split(t)
    out = []
    for p in parts:
        p = p.strip()
        if not p: continue
        if len(p) <= max_chars:
            out.append(p)
        else:
            sents = _SENT_SPLIT.split(p)
            buf, cur = [], 0
            for s in sents:
                s = s.strip()
                if not s: continue
                add = ((" " if buf else "") + s)
                if cur + len(add) > max_chars and buf:
                    out.append("".join(buf))
                    buf, cur = [s], len(s)
                else:
                    buf.append(add if buf else s)
                    cur += len(add)
            if buf: out.append("".join(buf))
    return out

def select_concept_contexts(subject: Optional[str], concepts: List[Dict[str,str]]) -> List[str]:
    # 과목 필터(그대로)
    if subject in GENERATOR_SUBJECTS_DEFAULT:
        pretty = SUBJECT_ALIAS.get(subject, subject)
        pool = [c for c in concepts
                if subject_match(c.get("top_subject",""), pretty)
                or subject_match(c.get("subject",""), pretty)]
        if not pool:
            log(f"⚠️ 개념 매칭 0건 → 전체에서 폴백 (subject={subject})")
            pool = concepts[:]
    else:
        pool = concepts[:]

    # 후보 청크 생성
    blob = "\n\n".join([c["content"] for c in pool]).strip()
    if not blob:
        return []
    cand_chunks = _split_for_chunks(blob, CTX_CONCEPT_CHUNK_CHARS)
    cand_chunks = [ch.strip() for ch in cand_chunks if ch.strip()]
    if not cand_chunks:
        return []

    # (A) 고정 TopK 모드: CTX_CONCEPT_TOPK > 0이면 기존 동작 유지
    if CTX_CONCEPT_TOPK and CTX_CONCEPT_TOPK > 0:
        chunks_sorted = sorted(cand_chunks, key=lambda x: len(x), reverse=True)
        picked = chunks_sorted[:CTX_CONCEPT_TOPK]
        return ["### 개념/요약\n" + ch for ch in picked]

    # (B) 자동 모드: 총 길이 예산 기반
    #   - 너무 긴 청크만 반복되는 걸 방지하기 위해 상위 12개를 뽑아 살짝 셔플 후 greedy 적재
    #   - 매 실행마다 약간 다른 조합을 위해 지터 ±200
    budget = max(800, CTX_CONCEPT_TOTAL_CHARS + random.randint(-200, 200))
    base = sorted(cand_chunks, key=lambda x: len(x), reverse=True)[:12]
    random.shuffle(base)

    picked, cur = [], 0
    for ch in base:
        L = len(ch)
        # 첫 청크는 무조건 채택, 이후엔 예산 초과 시 중단
        if not picked or cur + L <= budget:
            picked.append(ch)
            cur += L
        # 예산을 이미 넘었다면 중단
        if cur >= budget:
            break

    # 안전장치: 최소 1개
    if not picked:
        picked = [base[0]]

    return ["### 개념/요약\n" + ch for ch in picked]


def embed_passages(texts: List[str], embedder: HuggingFaceEmbeddings) -> np.ndarray:
    if not texts: return np.zeros((0,1), dtype=np.float32)
    vecs = embedder.embed_documents([f"passage: {t}" for t in texts])
    return np.array(vecs, dtype=np.float32)

def embed_query(text: str, embedder: HuggingFaceEmbeddings) -> np.ndarray:
    v = embedder.embed_query(f"query: {text}")
    return np.array(v, dtype=np.float32)

def cosine_sim(qvec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if mat.shape[0] == 0: return np.array([])
    qn = qvec / (np.linalg.norm(qvec) + 1e-12)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return mn @ qn

def argsort_topk(sims: np.ndarray, k: int) -> List[int]:
    if sims.size == 0: return []
    k = min(k, sims.size)
    idx = np.argpartition(-sims, kth=k-1)[:k]
    return idx[np.argsort(-sims[idx])].tolist()

def format_problem_ctx(p: Dict[str, Any]) -> str:
    opts = "\n".join([f"{i+1}) {o}" for i, o in enumerate(p.get("options") or [])])
    return ("### 공식 문제 (컨텍스트)\n"
            f"과목: {p.get('subject','')}\n"
            f"문제: {p.get('question','')}\n{opts}").strip()

def _normalize_gt(obj: dict) -> Optional[dict]:
    """
    ground_truth를 {question: str, options: [str, str, str, str]} 로 강제.
    - 여분 키 제거
    - 옵션 4개 정확히, 중복 금지
    - 앞번호/불릿 제거
    """
    if not isinstance(obj, dict):
        return None
    q = clean_text(obj.get("question", ""))
    opts_raw = obj.get("options", [])
    if not isinstance(q, str) or not isinstance(opts_raw, list):
        return None
    def _strip_bullet(s: str) -> str:
        s = clean_text(s)
        s = re.sub(r'^\s*(?:\d+\)|\(\d+\)|[①-⑨]|[-*•●▶▷◇])\s*', '', s)
        return s.strip()
    opts = [_strip_bullet(o) for o in opts_raw if isinstance(o, str) and clean_text(o)]
    if len(opts) != 4:
        return None
    if len(set(opts)) < 4:
        return None
    if not q:
        return None
    return {"question": q, "options": opts}


# -------------------- 페르소나/LLM 준비 (RAGAS 스타일) --------------------
def build_llm_for_generation():
    kwargs = dict(
        model=OPENAI_LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
        api_key=OPENAI_API_KEY,
    )
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    core = ChatOpenAI(**kwargs)
    wrapper = LangchainLLMWrapper(core)
    return wrapper, core

def get_personas() -> List[Persona]:
    base_rule = (
        "모든 생성은 제공된 컨텍스트의 사실에만 근거한다. "
        "하나의 핵심 개념만 다루며, 메타/출처/범위 밖 질문은 금지한다. "
        "보기는 상호 배타적·간결하며 4개로 제한한다."
    )
    return [
        Persona(
            name="ConceptSeekerKR",
            role_description=(
                "정보처리기사 개념 학습자 관점. 정의/구성요소/특징/비교/제약/적용조건 중심으로 질문 생성. " + base_rule
            ),
        ),
        Persona(
            name="ExamSetterKR",
            role_description=(
                "정보처리기사 출제자 관점. 단일 사실/개념에 근거한 구체적인 문항을 만든다. " + base_rule
            ),
        ),
    ]

# 컨텍스트 → LLM용 프롬프트
GEN_PROMPT_TMPL = (
    "당신은 정보처리기사 출제 전문가이자 한국어 문항 작성자입니다.\n"
    "아래 컨텍스트만을 근거로, {subject_area} 과목의 객관식 1문제를 생성하세요.\n"
    "필수 규칙:\n"
    "1) 출력은 JSON 하나만: {{\"question\":\"문제\", \"options\":[\"보기1\",\"보기2\",\"보기3\",\"보기4\"]}}\n"
    "2) 보기에는 번호/기호를 붙이지 마세요. 텍스트만.\n"
    "3) 정답과 해설은 절대 포함하지 마세요.\n"
    "4) 한 문제는 하나의 핵심 개념만 묻고, 보기 4개는 상호배타적이어야 합니다.\n"
    "5) 컨텍스트 밖 지식 금지.\n\n"
    "[컨텍스트]\n{context}\n"
)

def generate_question_json(llm_core: ChatOpenAI, subject_area: str, contexts: List[str]) -> Optional[Dict[str, Any]]:
    ctx = "\n\n".join([c.replace("### 개념/요약\n","") for c in contexts])[:4000]
    prompt = GEN_PROMPT_TMPL.format(subject_area=subject_area, context=ctx)
    resp = llm_core.invoke(prompt)
    content = getattr(resp, "content", str(resp)).strip()
    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None

    return _normalize_gt(data)


# -------------------- 프롬프트/샘플 구성 --------------------
def pick_subject_for_prompt() -> str:
    pool = [s for s in SUBJECTS if s != ALL_SCOPE_TOKEN] or GENERATOR_SUBJECTS_DEFAULT
    return random.choice(pool)

def build_user_prompt(subject: Optional[str]) -> str:
    subj = subject if subject in GENERATOR_SUBJECTS_DEFAULT else "전체 범위"
    return (f"정보처리기사 {subj} 과목 1문제 만들어줘. "
            f"보기 총 4개여야 하고 그중 정답은 1개여야 해. "
            f"문제 질문과 보기 4개만 만들어야 하고 정답과 해설은 만들지마.")

# -------------------- 메인 --------------------
def main():
    log("경로 확인")
    log(f"EXAM_DIR    = {EXAM_DIR} (exists: {os.path.isdir(EXAM_DIR)})")
    log(f"CONCEPT_DIR = {CONCEPT_DIR} (exists: {os.path.isdir(CONCEPT_DIR)})")
    log(f"CONCEPT_MASTER_FILE = {CONCEPT_MASTER_FILE}")
    log(f"OUT_DIR     = {OUT_DIR}")
    log(f"TARGET_SAMPLES={TARGET_SAMPLES}")

    # 1) 로드
    concept_items = load_concept_corpus(CONCEPT_DIR)
    problem_bank  = load_problem_bank(EXAM_DIR)
    log(f"개념 텍스트 조각: {len(concept_items)}")
    log(f"문제 풀(보기 4개 이상): {len(problem_bank)}")

    # 2) 문제 인덱스(공식 문제 컨텍스트 유사도용)
    embedder = build_embedder()
    problem_texts = [clean_text(p["question"] + " " + " ".join(p.get("options") or [])) for p in problem_bank]
    problem_mat   = embed_passages(problem_texts, embedder)

    # 3) LLM & 페르소나
    llm_wrapper, llm_core = build_llm_for_generation()
    personas = get_personas()  # 페르소나 가이드는 프롬프트에 이미 반영됨

    # 4) 샘플 생성 (context-first + LLM GT without answers)
    rows: List[Dict[str, Any]] = []
    seen = set()

    for i in range(TARGET_SAMPLES):
        subject = pick_subject_for_prompt()
        user_prompt = build_user_prompt(subject)

        # 4-1) 개념 컨텍스트 K
        concept_contexts = select_concept_contexts(subject, concept_items)
        if not concept_contexts:
            log("⚠️ 개념 컨텍스트 0건 → 스킵"); continue

        # 4-2) 공식 문제 컨텍스트 2개 (유사도 상위 20 중 무작위 선택)
        concept_query = "\n\n".join([c.replace("### 개념/요약\n","") for c in concept_contexts])[:4000]
        qvec = embed_query(concept_query, embedder)
        sims = cosine_sim(qvec, problem_mat)
        if sims.size:
            idx = np.argpartition(-sims, kth=min(20, sims.size)-1)[:min(20, sims.size)]
            idx = idx[np.argsort(-sims[idx])]
            cand = list(idx)
        else:
            cand = []
        random.shuffle(cand)
        chosen = cand[:min(CTX_PROBLEM_CTXK, len(cand))]
        problem_contexts = [format_problem_ctx(problem_bank[j]) for j in chosen]

        # 4-3) contexts 합치기(개념 K + 문제 2)
        contexts_field = concept_contexts + problem_contexts

        # 4-4) LLM으로 GT(정답/해설 없는 문제 JSON 생성)
        gt = generate_question_json(llm_core, subject, contexts_field)
        if not gt:
            log("⚠️ LLM GT 생성 실패 → 스킵"); continue

        # 4-5) 중복 억제(질문+첫 컨텍스트 해시)
        c0 = (contexts_field[0] if contexts_field else "")
        fp = hashlib.sha1((gt["question"] + "||" + c0).encode("utf-8")).hexdigest()
        if fp in seen:
            log("↷ 중복 감지 → 스킵"); continue
        seen.add(fp)

        rows.append({
            "question": user_prompt,
            "ground_truth": [gt],                  # {'question', 'options[4]'} only
            "contexts": contexts_field if contexts_field else [""]
        })

        if (i + 1) % 10 == 0:
            log(f"진행: {i+1}/{TARGET_SAMPLES}")

    # 5) 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(OUT_DIR, f"generator_golden_ragas_style_{ts}.jsonl")
    csv_path   = os.path.join(OUT_DIR, f"generator_golden_ragas_style_{ts}.csv")

    with open(jsonl_path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_rows = []
    for r in rows:
        q = r["question"]
        n = len(r["ground_truth"])
        c0 = (r["contexts"][0] if (r.get("contexts") and r["contexts"][0]) else "")
        cprev = (c0[:300] + "…") if len(c0) > 300 else c0
        csv_rows.append({"question": q, "gt_len": n, "contexts_preview": cprev})
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    log(f"✅ 완료: {len(rows)} 샘플 저장")
    log(f"JSONL → {jsonl_path}")
    log(f"CSV   → {csv_path}")

if __name__ == "__main__":
    main()
