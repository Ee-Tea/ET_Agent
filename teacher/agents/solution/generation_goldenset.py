# -*- coding: utf-8 -*-
"""
정보처리기사 문제풀이 에이전트용 골든셋 생성기

입력:
- EXAMS_PATH: 골든셋의 '문항 소스'가 되는 문제 JSON (또는 디렉터리)
  포맷: {"exam_title":..., "questions":[{"question","options"[],"answer","explanation","subject"}...]}
- SIMILAR_PROBLEMS_PATH: '유사 문제'로 참조할 문제 JSON(파일 또는 디렉터리)
  (위와 동일 포맷, EXAMS_PATH와 동일 파일을 넣어도 됨. 자기 자신은 검색에서 제외)
- CONCEPTS_DIR: 유사 개념 JSON들이 있는 디렉터리
  포맷 A: {"subject":..., "items":[{"item_title","content","subject"}...]}
  포맷 B: [{"item_title","content","subject"}, ...]

동작:
- 각 문항(question+options 텍스트) 임베딩 → 개념 코퍼스에서 top-2, 유사문제에서 top-2 검색
- contexts = [개념2, 문제2] (총 4개)  ※ 문제 자기자신 제외
- ground_truth = 평문("정답:~\n풀이:~\n과목:~")

출력:
- OUT_DIR/ipa_golden_{YYYYmmdd_HHMMSS}.jsonl
- OUT_DIR/ipa_golden_{YYYYmmdd_HHMMSS}.csv  (엑셀 호환 UTF-8-SIG)

필요:
- pip install -U "langchain-huggingface" "langchain-openai" "langchain-text-splitters" "pandas" "numpy" "python-dotenv"
- 임베딩: intfloat/multilingual-e5-large (자동 다운로드)
- OPENAI_API_KEY=REDACTED 풀이 생성에 필요; 없으면 폴백 생성)
"""

import os
import re
import glob
import json
import uuid
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# 임베딩
from langchain_huggingface import HuggingFaceEmbeddings
# LLM (풀이 생성용)
from langchain_openai import ChatOpenAI

# 텍스트 전처리/분할(개념 문서용)
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- 디버그/옵션 ----
load_dotenv()
DEBUG = os.getenv("DEBUG", "1") == "1"
USE_AGENT_PROMPT = os.getenv("USE_AGENT_PROMPT", "1") == "1"
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
PROMPT_CTX_MAX_CHARS = int(os.getenv("PROMPT_CTX_MAX_CHARS", "900"))

def log(msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

def log_head(title: str):
    line = "─" * 60
    print(f"\n{line}\n{title}\n{line}")

def log_list(name: str, arr, n=3):
    try:
        total = len(arr)
    except Exception:
        total = "?"
    head = arr[:n] if isinstance(arr, list) else []
    log(f"{name}: 총 {total}개")
    for i, it in enumerate(head):
        log(f"  - {i+1}: {str(it)[:120]}{'...' if len(str(it))>120 else ''}")

# ===================== 경로 설정 =====================
# 이 파일 위치: teacher/agents/solution/generation_goldenset.py
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # -> teacher/

# 기본 경로(프로젝트 루트 기준)
DEFAULT_EXAMS_PATH            = os.path.join(PROJECT_ROOT, "exam", "test_parsed_exam_json")        # ← 소스(오피셜 정답 JSON 폴더)
DEFAULT_SIMILAR_PROBLEMS_PATH = os.path.join(PROJECT_ROOT, "exam", "parsed_exam_json")             # ← context(유사문제)
DEFAULT_CONCEPTS_DIR          = os.path.join(PROJECT_ROOT, "agents", "retrieve", "data", "json")   # ← context(유사개념)
DEFAULT_OUT_DIR               = os.path.join(PROJECT_ROOT, "agents", "solution", "goldensets")     # ← 결과 저장

EXAMS_PATH            = os.getenv("EXAMS_PATH", DEFAULT_EXAMS_PATH)                  # 폴더
SIMILAR_PROBLEMS_PATH = os.getenv("SIMILAR_PROBLEMS_PATH", DEFAULT_SIMILAR_PROBLEMS_PATH)  # 폴더
CONCEPTS_DIR          = os.getenv("CONCEPTS_DIR", DEFAULT_CONCEPTS_DIR)              # 폴더
OUT_DIR               = os.getenv("OUT_DIR", DEFAULT_OUT_DIR)                        # 폴더
os.makedirs(OUT_DIR, exist_ok=True)

MIN_CHARS  = int(os.getenv("MIN_CHARS", "50"))
LIMIT_ENV  = os.getenv("LIMIT", None)
LIMIT: Optional[int] = int(LIMIT_ENV) if (LIMIT_ENV and LIMIT_ENV.isdigit()) else None

USE_OFFICIAL_SOLUTION = os.getenv("USE_OFFICIAL_SOLUTION", "1") == "1"  # 공식풀이가 있으면 우선 사용
MERGE_OFFICIAL_AND_GEN = os.getenv("MERGE_OFFICIAL_AND_GEN", "0") == "1"  # 공식풀이+생성풀이 병합

random.seed(42)

log_head("경로 확인")
log(f"HERE           = {HERE}")
log(f"PROJECT_ROOT   = {PROJECT_ROOT}")
log(f"EXAMS_PATH     = {EXAMS_PATH}    (exists: {os.path.isdir(EXAMS_PATH)})")
log(f"SIMILAR_PATH   = {SIMILAR_PROBLEMS_PATH} (exists: {os.path.isdir(SIMILAR_PROBLEMS_PATH)})")
log(f"CONCEPTS_DIR   = {CONCEPTS_DIR}  (exists: {os.path.isdir(CONCEPTS_DIR)})")
log(f"OUT_DIR        = {OUT_DIR}       (exists: {os.path.isdir(OUT_DIR)})")

# ===================== 유틸 =====================
def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def as_list_path_or_dir(path_or_dir: str, pattern="*.json") -> List[str]:
    if not path_or_dir:
        log("⚠️ as_list_path_or_dir: 입력 경로가 비었습니다.")
        return []
    if os.path.isdir(path_or_dir):
        files = sorted(glob.glob(os.path.join(path_or_dir, "**", pattern), recursive=True))
        log(f"📁 디렉터리 스캔: {path_or_dir} → {len(files)}개 매칭")
        if DEBUG: log_list("  예시 파일", files, n=5)
        return files
    if os.path.isfile(path_or_dir):
        log(f"📄 단일 파일 사용: {path_or_dir}")
        return [path_or_dir]
    log(f"❌ 경로가 존재하지 않습니다: {path_or_dir}")
    return []

# --- 컨텍스트 중복 제거 유틸 ---
def _normalize_ctx_key(text: str) -> str:
    """
    컨텍스트 비교용 키로 정규화:
    - [개념]/[유사문제] 헤더 제거
    - 공백/개행 축약
    """
    if not text:
        return ""
    t = str(text)
    # 헤더 제거
    t = re.sub(r"^\s*\[(개념|유사문제)\]\s*", "", t, flags=re.I)
    # 공백/개행 축약
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _dedup_keep_order(blocks: List[str], max_n: int) -> List[str]:
    """정규화 키로 중복 제거(순서 유지) 후 최대 max_n개까지 반환"""
    seen = set()
    out = []
    for b in blocks:
        k = _normalize_ctx_key(b)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(b)
        if len(out) >= max_n:
            break
    return out

# ===================== 개념 로딩/분할 =====================
SPLIT_CHARS = 850
SPLIT_OVERLAP = 140

def load_concept_docs(root_dir: str) -> List[Document]:
    log_head("개념 JSON 로딩")
    if not root_dir or not os.path.isdir(root_dir):
        return []
    docs: List[Document] = []
    for path in glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ 개념 JSON 로드 실패: {path} ({e})")
            continue

        def _push(item: Dict[str, Any], idx: int):
            content = clean_text(item.get("content", ""))
            if not content:
                return
            title = clean_text(item.get("item_title", ""))
            page = (f"{title}\n{content}" if title else content).strip()
            if len(page) < MIN_CHARS:
                return
            meta = {
                "kind": "concept",
                "subject": item.get("subject"),
                "item_title": title or None,
                "source": path,
                "idx": idx,
            }
            docs.append(Document(page_content=page, metadata=meta))

        if isinstance(data, dict) and isinstance(data.get("items"), list):
            for idx, it in enumerate(data["items"]):
                if isinstance(it, dict):
                    _push(it, idx)
        elif isinstance(data, list):
            for idx, it in enumerate(data):
                if isinstance(it, dict):
                    _push(it, idx)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SPLIT_CHARS, chunk_overlap=SPLIT_OVERLAP, separators=["\n\n", "\n", " ", ""]
    )
    log(f"개념 원문 문서 수(분할 전): {len(docs)}")
    chunks: List[Document] = []
    for d in docs:
        chunks.extend(splitter.split_documents([d]))
    chunks = [c for c in chunks if len(c.page_content) >= 200]
    log(f"개념 청크 수(분할 후, len>=200): {len(chunks)}")
    return chunks

# ===================== 유사문제 로딩 =====================
def load_problem_bank(paths: List[str]) -> List[Dict[str, Any]]:
    """
    문제은행: 검색용으로 flatten
    """
    bank: List[Dict[str, Any]] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ 문제 JSON 로드 실패: {path} ({e})")
            continue

        exam_title = data.get("exam_title") if isinstance(data, dict) else None
        questions = data.get("questions", []) if isinstance(data, dict) else []
        if isinstance(questions, list):
            for q in questions:
                if not isinstance(q, dict): 
                    continue
                qtext = clean_text(q.get("question", ""))
                opts  = [clean_text(o) for o in (q.get("options") or [])]
                if not qtext:
                    continue
                item = {
                    "qid": str(uuid.uuid4()),
                    "question": qtext,
                    "options": opts,
                    "answer": str(q.get("answer", "")),
                    "explanation": clean_text(q.get("explanation", "")),
                    "subject": clean_text(q.get("subject", "")),
                    "source_exam": exam_title,
                    "source_path": path,
                }
                bank.append(item)
    return bank

# ===================== 임베딩/검색 =====================
def build_embeddings(texts: List[str], model_name: str = "intfloat/multilingual-e5-large") -> Tuple[np.ndarray, HuggingFaceEmbeddings]:
    """문서 임베딩 (e5: passage 프리픽스) + 임베더 반환"""
    log_head(f"임베딩 계산 시작 (문서 {len(texts)}개)")
    t0 = time.time()
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    prefixed = [f"passage: {t}" for t in texts]
    vecs = embedder.embed_documents(prefixed)  # List[List[float]]
    mat = np.array(vecs, dtype=np.float32)
    dt = time.time() - t0
    log(f"임베딩 완료: shape={mat.shape}, 소요={dt:.2f}s, 모델={model_name}")
    return mat, embedder

def embed_query(text: str, embedder: HuggingFaceEmbeddings) -> np.ndarray:
    """질의 임베딩 (e5: query 프리픽스)"""
    q = f"query: {text}"
    v = embedder.embed_query(q)
    arr = np.array(v, dtype=np.float32)
    log(f"질의 임베딩 완료: dim={arr.shape[0]}, 텍스트 길이={len(text)}")
    return arr

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return b_norm @ a_norm

def search_top_k(qvec: np.ndarray, mat: np.ndarray, k: int = 2, exclude_idx: Optional[int] = None) -> List[int]:
    sims = cosine_sim(qvec, mat)  # (N,)
    if exclude_idx is not None and 0 <= exclude_idx < len(sims):
        sims[exclude_idx] = -1e9
    idx = np.argpartition(-sims, kth=min(k, len(sims)-1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist()

# ===================== 에이전트 프롬프트(정답 고정) =====================
def _split_blocks(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [b.strip() for b in re.split(r"\n{2,}", text) if b and b.strip()]
    return parts

def _format_ctx_for_prompt(final_blocks: List[Tuple[str, str]], max_chars: int = 900) -> str:
    lines, used = [], 0
    for i, (kind, block) in enumerate(final_blocks, 1):
        src = "유사문제" if kind == "problem" else "개념"
        header = f"[CTX {i} | {src}]"
        body = block.strip()
        remain = max(0, max_chars - used)
        if remain <= 0:
            break
        if len(body) > remain:
            body = body[:remain]
        lines.append(f"{header}\n{body}")
        used += len(body)
    return "\n\n".join(lines)

def _strip_md(txt: str) -> str:
    t = re.sub(r"^```.*?```", "", txt, flags=re.S)
    t = re.sub(r"[*_`>]+", "", t)
    return t.strip()

def generate_explanation_fixed_answer(answer_idx: str,
                                      question_txt: str,
                                      options: List[str],
                                      concept_ctxs: List[str],
                                      problem_ctxs: List[str]) -> Tuple[str, str]:
    """
    정답 번호(answer_idx)는 확정.
    컨텍스트만 근거로 '왜 그 답이 맞는지' 풀이를 생성.
    반환: (explanation, maybe_subject)  # subject는 공란일 수 있음
    """
    # 블록 구성: 유사문제 2 + 개념 2
    p_text = "\n\n".join(problem_ctxs or [])
    c_text = "\n\n".join(concept_ctxs or [])
    p_blocks = _split_blocks(p_text)[:2]
    c_blocks = _split_blocks(c_text)[:2]
    final_blocks: List[Tuple[str, str]] = [("problem", b) for b in p_blocks] + [("concept", b) for b in c_blocks]
    ctx_structured = _format_ctx_for_prompt(final_blocks, max_chars=PROMPT_CTX_MAX_CHARS)

    # 보기
    opts_lines = "\n".join(f"{i+1}) {o}" for i, o in enumerate(options or []))
    sel_txt = ""
    try:
        idx_int = int(answer_idx) - 1
        if 0 <= idx_int < len(options):
            sel_txt = options[idx_int]
    except:
        pass

    prompt = f"""
너는 정보처리기사 해설 작성자다.
정답 번호는 이미 확정되었다: {answer_idx}번. 정답 번호를 바꾸지 말고,
'왜 {answer_idx}번이 정답인지'를 아래 컨텍스트 블록만을 근거로 3~6문장 한국어 평문 풀이로 작성하라.
오직 컨텍스트에 명시된 사실만 사용하고, 새로운 지식/추론을 추가하지 말 것.
가능하면 근거 키워드를 원문 표현과 동치로 재서술하라. 마크다운 금지.

[문제]
{question_txt}

[보기]
{opts_lines}

[정답 보기 텍스트]
{sel_txt}

[컨텍스트 블록들]
{ctx_structured}

출력 형식:
풀이: ...
과목(선택): ...
""".strip()

    try:
        llm = ChatOpenAI(model=AGENT_MODEL, temperature=0.3, max_tokens=600)
        res = llm.invoke(prompt)
        raw = (getattr(res, "content", None) or str(res)).strip()
        clean = _strip_md(raw)
        # 풀이 추출
        sol_m = re.search(r"풀이\s*[:：]\s*(.+?)(?:\n\s*과목|$)", clean, flags=re.S)
        sub_m = re.search(r"과목\s*[:：]\s*([^\n]+)", clean)
        explanation = (sol_m.group(1).strip() if sol_m else clean).strip()
        subject = (sub_m.group(1).strip() if sub_m else "").strip()
        return explanation[:1000], subject
    except Exception as e:
        log(f"⚠️ LLM 풀이 생성 실패(정답 고정): {e}")
        # 폴백: 컨텍스트 한두 문장 인용
        base_ctx = (concept_ctxs[:1] + problem_ctxs[:1])
        snippet = (base_ctx[0] if base_ctx else "")
        snippet = " ".join(snippet.splitlines()[:3])[:400]
        fallback = f"{answer_idx}번이 정답인 근거는 컨텍스트에 제시된 내용 때문이다. {snippet}"
        return fallback, ""

# ===================== 본 생성 로직 =====================
def main():
    log_head("프로그램 시작")
    print("📘 정보처리기사 문제풀이 에이전트 — 골든셋 생성기")
    print("=" * 60)
    print(f"EXAMS_PATH            : {EXAMS_PATH}")
    print(f"SIMILAR_PROBLEMS_PATH : {SIMILAR_PROBLEMS_PATH}")
    print(f"CONCEPTS_DIR          : {CONCEPTS_DIR}")
    print(f"OUT_DIR               : {OUT_DIR}")
    print(f"MIN_CHARS             : {MIN_CHARS}")
    print(f"LIMIT                 : {LIMIT}")
    print(f"USE_AGENT_PROMPT      : {USE_AGENT_PROMPT} (model={AGENT_MODEL})")

    # 1) 개념 로딩
    concept_docs = load_concept_docs(CONCEPTS_DIR)
    log(f"📚 개념 청크 수: {len(concept_docs)}")

    # 2) 유사문제 로딩 (검색 풀)
    similar_problem_files = as_list_path_or_dir(SIMILAR_PROBLEMS_PATH)
    similar_bank = load_problem_bank(similar_problem_files)
    log(f"🧩 유사문제 풀 수: {len(similar_bank)}")
    if DEBUG and len(similar_bank) > 0:
        log(f"🧩 유사문제 예시: Q='{similar_bank[0]['question'][:80]}...'")

    # 3) 문항 소스 로딩 (골든셋의 대상: 오피셜 정답 JSON 폴더)
    exam_files = as_list_path_or_dir(EXAMS_PATH)
    if not exam_files:
        log("❌ EXAMS_PATH에 유효한 파일/디렉터리가 없습니다.")
        return
    exam_source = load_problem_bank(exam_files)
    if not exam_source:
        log("❌ 문항 소스(JSON)에서 문항을 읽지 못했습니다.")
        return
    if LIMIT:
        exam_source = exam_source[:LIMIT]
    log(f"📝 대상 문항 수: {len(exam_source)}")
    if DEBUG and len(exam_source) > 0:
        log(f"📝 대상 문항 예시: Q='{exam_source[0]['question'][:80]}...'")

    # 4) 인덱스 구축 (개념/유사문제)
    log_head("인덱스 구축 (텍스트 수집)")
    concept_texts = [d.page_content for d in concept_docs]
    problem_texts = [
        clean_text(p["question"] + " " + " ".join(p.get("options") or []))
        for p in similar_bank
    ]
    log(f"개념텍스트={len(concept_texts)}, 문제텍스트={len(problem_texts)}")

    # 임베더 초기화
    model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    try:
        log_head("임베더 초기화")
        embedder = HuggingFaceEmbeddings(model_name=model_name)
        log(f"임베더 준비 완료: {model_name}")
    except Exception as e:
        log(f"❌ 임베더 초기화 실패: {e}")
        raise

    # 행렬 계산
    try:
        if concept_texts:
            concept_vecs = embedder.embed_documents([f"passage: {t}" for t in concept_texts])
            concept_mat = np.array(concept_vecs, dtype=np.float32)
        else:
            concept_mat = np.zeros((0, 1), dtype=np.float32)
        log(f"개념 임베딩 shape={concept_mat.shape}")
    except Exception as e:
        log(f"❌ 개념 임베딩 실패: {e}")
        raise

    try:
        if problem_texts:
            problem_vecs = embedder.embed_documents([f"passage: {t}" for t in problem_texts])
            problem_mat = np.array(problem_vecs, dtype=np.float32)
        else:
            problem_mat = np.zeros((0, 1), dtype=np.float32)
        log(f"문제 임베딩 shape={problem_mat.shape}")
    except Exception as e:
        log(f"❌ 문제 임베딩 실패: {e}")
        raise

    # 5) 검색 + 골든셋 구성
    log_head("골든셋 구성 시작")
    rows = []
    for i, item in enumerate(exam_source, 1):
        if i % 10 == 1:
            log(f"진행: {i}/{len(exam_source)}")

        q_text = clean_text(item["question"])
        opts   = [clean_text(o) for o in (item.get("options") or [])]
        query  = (q_text + "\n" + "\n".join([f"- {o}" for o in opts])).strip()

        # 쿼리 임베딩
        try:
            qvec = embed_query(query, embedder)
        except Exception as e:
            log(f"❌ 질의 임베딩 실패 (i={i}): {e}")
            continue

        # 개념 top-2
        # 개념 top-2 (여유분에서 중복 제거)
        concept_ctxs: List[str] = []
        if len(concept_docs) > 0 and concept_mat.shape[0] > 0:
            try:
                # 여유분을 넉넉히 뽑고(최대 12), 그중 유니크 2개만 사용
                cand_idx = search_top_k(qvec, concept_mat, k=min(12, concept_mat.shape[0]))
                if DEBUG: log(f"[{i}] 개념 후보 인덱스(여유분): {cand_idx}")
                cand_blocks: List[str] = []
                for j in cand_idx:
                    d = concept_docs[j]
                    title = d.metadata.get("item_title") or ""
                    ctx = f"[개념] {title}\n{d.page_content}" if title else f"[개념]\n{d.page_content}"
                    cand_blocks.append(ctx)
                concept_ctxs = _dedup_keep_order(cand_blocks, max_n=2)
            except Exception as e:
                log(f"⚠️ 개념 검색 실패 (i={i}): {e}")

        # 유사문제 top-2 (자기 자신 제외 + 여유분에서 중복 제거)
        problem_ctxs: List[str] = []
        if len(similar_bank) > 0 and problem_mat.shape[0] > 0:
            try:
                # 동일 문제 텍스트 제외 인덱스
                exclude_idx = None
                base = clean_text(item["question"] + " " + " ".join(item.get("options") or []))
                for idx, ptxt in enumerate(problem_texts):
                    if ptxt == base:
                        exclude_idx = idx
                        break

                cand_idx = search_top_k(qvec, problem_mat, k=min(12, problem_mat.shape[0]), exclude_idx=exclude_idx)
                if DEBUG: log(f"[{i}] 유사문제 후보 인덱스(여유분): {cand_idx}")
                cand_blocks: List[str] = []
                for j in cand_idx:
                    p = similar_bank[j]
                    ctx_q = p["question"]
                    ctx_opts = "\n".join([f"{k+1}) {o}" for k, o in enumerate(p.get("options") or [])])
                    ctx_ans = p.get("answer", "")
                    ctx_exp = p.get("explanation", "")
                    ctx_sub = p.get("subject", "")
                    block = (
                        "[유사문제]\n"
                        f"과목: {ctx_sub}\n"
                        f"문제: {ctx_q}\n"
                        f"{ctx_opts}\n"
                        f"정답: {ctx_ans}\n"
                        f"풀이: {ctx_exp}"
                    ).strip()
                    cand_blocks.append(block)

                problem_ctxs = _dedup_keep_order(cand_blocks, max_n=2)
            except Exception as e:
                log(f"⚠️ 유사문제 검색 실패 (i={i}): {e}")

        # ---- 정답/과목(오피셜)
        official_answer  = str(item.get("answer", "")).strip()
        official_subject = (item.get("subject", "") or "").strip()
        official_solution = (item.get("explanation", "") or "").strip()

        explanation = ""
        maybe_subject = ""

        if USE_OFFICIAL_SOLUTION and official_solution:
            # 1) 공식 풀이 우선 사용
            explanation = official_solution.strip()
            # 과목은 오피셜이 비면 나중에 LLM로 보강할 수도 있으므로 maybe_subject는 비워둠
            log(f"[{i}] 공식 풀이 사용(길이: {len(explanation)}자)")
        else:
            # 2) 공식 풀이가 없으면 컨텍스트 기반으로 생성
            if USE_AGENT_PROMPT:
                explanation, maybe_subject = generate_explanation_fixed_answer(
                    answer_idx=official_answer,
                    question_txt=q_text,
                    options=opts,
                    concept_ctxs=concept_ctxs[:2],
                    problem_ctxs=problem_ctxs[:2],
                )
                log(f"[{i}] 생성 풀이 사용(길이: {len(explanation)}자)")
            else:
                base_ctx = (concept_ctxs[:1] + problem_ctxs[:1])
                snippet = (base_ctx[0] if base_ctx else "")
                snippet = " ".join(snippet.splitlines()[:3])[:400]
                explanation, maybe_subject = (
                    f"{official_answer}번이 정답인 근거는 컨텍스트에 제시된 내용 때문이다. {snippet}",
                    ""
                )
                log(f"[{i}] 폴백 풀이 사용(길이: {len(explanation)}자)")

        # (선택) 병합 모드: 공식풀이 + 생성풀이를 함께 담고 싶다면
        if MERGE_OFFICIAL_AND_GEN and official_solution and USE_AGENT_PROMPT:
            # 위에서 공식 풀이를 우선 사용한 경우, 생성 풀이를 추가로 붙임
            if explanation.strip() != official_solution.strip():
                merged = f"{official_solution.strip()}\n(컨텍스트 근거) {explanation.strip()}"
                explanation = merged[:1200]  # 너무 길어지지 않도록 컷
                log(f"[{i}] 공식+생성 풀이 병합")

        # 과목 결정: 오피셜 우선, 없으면 LLM 힌트 사용
        final_subject = official_subject if official_subject else maybe_subject

        # ground_truth = 평문
        ground_truth_text = f"정답: {official_answer}\n풀이: {explanation}\n과목: {final_subject}".strip()


        # question 문자열(선택지 포함)
        display_q = q_text
        if opts:
            numbered = "\n".join([f"{k+1}) {o}" for k, o in enumerate(opts)])
            display_q = f"{q_text}\n{numbered}"

        # contexts = concept_ctxs[:2] + problem_ctxs[:2]
        contexts = _dedup_keep_order(concept_ctxs[:2] + problem_ctxs[:2], max_n=4)

        rows.append({
            "question": display_q,
            "ground_truth": ground_truth_text,
            "contexts": contexts,
        })

    # 저장
    log_head("저장 단계")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(OUT_DIR, f"ipa_golden_{ts}.jsonl")
    csv_path   = os.path.join(OUT_DIR, f"ipa_golden_{ts}.csv")
    log(f"JSONL → {jsonl_path}")
    log(f"CSV   → {csv_path}")

    with open(jsonl_path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_rows = [
        {"question": r["question"], "ground_truth": r["ground_truth"], "contexts": " | ".join(r["contexts"])}
        for r in rows
    ]
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    log(f"✅ 저장 완료. 샘플 수={len(rows)}")

if __name__ == "__main__":
    main()
