# -*- coding: utf-8 -*-
"""
정보처리기사 '문제 생성 에이전트'용 RAGAS 골든셋 생성기

형식 (RAGAS JSONL 1줄/샘플):
{
  "question": "<사용자 문제 생성 요청 프롬프트>",
  "ground_truth": [
      {"question": "...", "options": ["...", "...", "...", "..."]},
      ...
  ],
  "contexts": ["<컨텍스트 전량(또는 컷) 문자열 1덩어리>"]
}

설명
- question: generator에게 보낼 "문제 생성 요청" 프롬프트 (과목 지정/전체 범위 랜덤, 개수 1~≤100 랜덤)
- ground_truth: 평가 타겟 형태에 맞춰 '보기만' 제공(정답/해설 없음)
- contexts: 지정 폴더의 JSON들을 읽어 만든 컨텍스트(전량 1원소). 너무 크면 MAX_CTX_CHARS로 컷

환경변수(.env 가능)
- EXAM_DIR=teacher/exam/parsed_exam_json
- CONCEPT_DIR=teacher/agents/retrieve/data/json
- OUT_DIR=teacher/agents/TestGenerator/goldensets   (없으면 생성)
- TARGET_SAMPLES=50          # 만들 샘플 수
- MAX_Q_PER_REQUEST=100      # 한 요청 최대 문항수
- SUBJECTS=콤마구분 과목명   # 기본 5과목 + '전체 범위' 자동 포함
- MIN_CTX_CHARS=50           # 너무 짧은 조각은 스킵
- MAX_CTX_CHARS=0            # 0이면 전량, >0이면 그 길이만큼 잘라서 contexts[0]에 저장
- SEED=42

필요: pip install pandas python-dotenv
"""

import os
import re
import json
import glob
import random
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union
from dotenv import load_dotenv

load_dotenv()

# -------------------- 경로/환경 --------------------
DEFAULT_EXAM_DIR = os.path.join("teacher", "exam", "parsed_exam_json")
DEFAULT_CONCEPT_DIR = os.path.join("teacher", "agents", "retrieve", "data", "json")
DEFAULT_OUT_DIR = os.path.join("teacher", "agents", "TestGenerator", "goldensets")

EXAM_DIR = os.getenv("EXAM_DIR", DEFAULT_EXAM_DIR)
CONCEPT_DIR = os.getenv("CONCEPT_DIR", DEFAULT_CONCEPT_DIR)
OUT_DIR = os.getenv("OUT_DIR", DEFAULT_OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_SAMPLES = int(os.getenv("TARGET_SAMPLES", "20"))
MAX_Q_PER_REQUEST = max(1, int(os.getenv("MAX_Q_PER_REQUEST", "5")))
MIN_CTX_CHARS = int(os.getenv("MIN_CTX_CHARS", "50"))
MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "0"))  # 0 => 전량
SEED = int(os.getenv("SEED", "42"))
CTX_PROBLEM_COUNT = int(os.getenv("CTX_PROBLEM_COUNT", "2"))     # 컨텍스트에 넣을 '공식 문제' 개수
CTX_CONCEPT_MAX_CHARS = int(os.getenv("CTX_CONCEPT_MAX_CHARS", "4000"))  # 컨셉트 텍스트 컷 (0이면 전량)

# generator.py의 5과목 명칭을 그대로 사용 + '전체 범위' 가능
GENERATOR_SUBJECTS_DEFAULT = [
    "소프트웨어설계",
    "소프트웨어개발",
    "데이터베이스구축",
    "프로그래밍언어활용",
    "정보시스템구축관리",
]
SUBJECTS = [
    s.strip() for s in os.getenv("SUBJECTS", ",".join(GENERATOR_SUBJECTS_DEFAULT)).split(",")
    if s.strip()
]
# '전체 범위'도 랜덤 대상에 포함
ALL_SCOPE_TOKEN = "전체 범위"
if ALL_SCOPE_TOKEN not in SUBJECTS:
    SUBJECTS.append(ALL_SCOPE_TOKEN)

random.seed(SEED)

# -------------------- 유틸 --------------------
def log(msg: str):
    print(f"[goldenset] {msg}")

def clean_text(s: Any) -> str:
    if s is None:
        return ""
    t = str(s)
    t = t.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def list_json_files(root: str) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    return sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))

# -------------------- 컨텍스트 구축 --------------------

def load_concept_corpus(concept_dir: str) -> List[Dict[str, str]]:
    """
    개념/요약 JSON에서 'content'와 'subject'를 추출
    - 포맷 A: {"subject": ..., "items":[{"item_title","content","subject"}, ...]}
    - 포맷 B: [{"item_title","content","subject"}, ...]
    - 포맷 C: {"content": ..., "subject": ...}
    """
    out: List[Dict[str, str]] = []

    def push_content(txt: Any, subject: str = ""):
        txt = clean_text(txt)
        subj = clean_text(subject)
        if len(txt) >= MIN_CTX_CHARS:
            out.append({"content": txt, "subject": subj})

    for fp in list_json_files(concept_dir):
        try:
            data = json.load(open(fp, "r", encoding="utf-8"))
        except Exception as e:
            log(f"⚠️ 개념 JSON 로드 실패: {fp} ({e})")
            continue

        # 포맷 A: dict with "items"
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            subj = data.get("subject", "")
            for it in data["items"]:
                if isinstance(it, dict) and "content" in it:
                    push_content(it["content"], it.get("subject", subj))
            # 혹시 최상위에도 content/subject가 있으면 수집
            if "content" in data:
                push_content(data["content"], data.get("subject", subj))

        # 포맷 B: list of dicts
        elif isinstance(data, list):
            for it in data:
                if isinstance(it, dict) and "content" in it:
                    push_content(it["content"], it.get("subject", ""))

        # 포맷 C: 단일 dict
        elif isinstance(data, dict) and "content" in data:
            push_content(data["content"], data.get("subject", ""))

    # 공백 content 제거
    return [item for item in out if item["content"].strip()]


def load_problem_bank(exam_dir: str) -> List[Dict[str, Any]]:
    """
    문제 JSON: {"exam_title":..., "questions":[{question, options[], answer, explanation, subject}]}
    - ground_truth로 쓸 보기-only 샘플을 뽑기 위해 로드
    """
    bank: List[Dict[str, Any]] = []
    for fp in list_json_files(exam_dir):
        try:
            data = json.load(open(fp, "r", encoding="utf-8"))
        except Exception as e:
            log(f"⚠️ 문제 JSON 로드 실패: {fp} ({e})")
            continue
        qs = data.get("questions", []) if isinstance(data, dict) else []
        for q in qs:
            if not isinstance(q, dict): 
                continue
            qtext = clean_text(q.get("question", ""))
            opts = [clean_text(o) for o in (q.get("options") or [])]
            subj = clean_text(q.get("subject", ""))
            if not qtext or len(opts) < 4:
                continue
            # 보기 4개로 정규화
            opts = opts[:4]
            bank.append({
                "question": qtext,
                "options": opts,
                "subject": subj
            })
    return bank

def build_context_for_sample(
    subject: Optional[str],
    concepts: List[Union[str, Dict[str, Any]]],
    problems: List[Dict[str, Any]],
) -> List[str]:
    """
    컨텍스트 구성:
    - (A) 개념/요약: 전량 합친 뒤 CTX_CONCEPT_MAX_CHARS로 컷
    - (B) 공식 문제: subject 기준으로 필터 후 최대 CTX_PROBLEM_COUNT개 샘플(문항+보기4개)
    반환: contexts 필드에 바로 넣을 리스트[str]
    """
    # ----- (A) 개념/요약 블록 만들기 -----
    concept_lines: List[str] = []
    for c in concepts or []:
        if isinstance(c, str):
            txt = clean_text(c)
            if len(txt) >= MIN_CTX_CHARS:
                concept_lines.append(txt)
        elif isinstance(c, dict):
            content = clean_text(c.get("content", ""))
            subj = clean_text(c.get("subject", ""))
            if len(content) >= MIN_CTX_CHARS:
                concept_lines.append(f"[{subj}]\n{content}" if subj else content)

    concept_blob = "\n\n".join(concept_lines).strip()
    if CTX_CONCEPT_MAX_CHARS and CTX_CONCEPT_MAX_CHARS > 0 and len(concept_blob) > CTX_CONCEPT_MAX_CHARS:
        concept_blob = concept_blob[:CTX_CONCEPT_MAX_CHARS]

    parts: List[str] = []
    if concept_blob:
        parts.append("### 개념/요약\n" + concept_blob)

    # ----- (B) 공식 문제 2개(기본) 선별 -----
    pool = problems
    if subject in GENERATOR_SUBJECTS_DEFAULT:
        def _match(s: str) -> bool:
            s = (s or "").replace(" ", "")
            t = subject.replace(" ", "")
            return t in s or s in t
        filt = [q for q in problems if _match(q.get("subject", ""))]
        if filt:
            pool = filt

    if pool:
        k_ctx = min(CTX_PROBLEM_COUNT, len(pool))
        picks = random.sample(pool, k_ctx) if len(pool) > k_ctx else pool
        lines = []
        for p in picks:
            q = clean_text(p.get("question", ""))
            opts = [clean_text(o) for o in (p.get("options") or [])][:4]
            if not q or len(opts) < 4:
                continue
            olines = "\n".join([f"- {o}" for o in opts])
            lines.append(f"문제: {q}\n{olines}")
        if lines:
            parts.append("### 공식 문제 (컨텍스트)\n" + "\n\n".join(lines))

    context_blob = "\n\n".join(parts).strip()
    return [context_blob if context_blob else ""]



# -------------------- 프롬프트/샘플러 --------------------
def pick_subject_for_prompt() -> str:
    """5과목 + '전체 범위' 중에서 랜덤."""
    return random.choice(SUBJECTS)


def build_user_prompt(subject: Optional[str]) -> Tuple[str, int]:
    """
    1~min(100, MAX_Q_PER_REQUEST)에서 랜덤 k,
    - subject가 5과목 중 하나면 과목 지시
    - subject가 '전체 범위'면 전 범위 지시
    - 그 외에도 동작 (기본 전 범위)
    """
    k = random.randint(1, max(1, min(10, MAX_Q_PER_REQUEST)))
    if subject in GENERATOR_SUBJECTS_DEFAULT:
        text = (
            f"정보처리기사 {subject} 과목 객관식 {k}문제 만들어줘. "
            f"각 문항은 보기 4개만 제공하고 정답과 해설은 주지 마."
        )
    elif subject == ALL_SCOPE_TOKEN:
        text = (
            f"정보처리기사 전체 범위에서 객관식 {k}문제 만들어줘. "
            f"각 문항은 보기 4개만 제공하고 정답과 해설은 주지 마."
        )
    else:
        # 안전 폴백: 전체 범위
        text = (
            f"정보처리기사 전체 범위에서 객관식 {k}문제 만들어줘. "
            f"각 문항은 보기 4개만 제공하고 정답과 해설은 주지 마."
        )
    return text, k

def sample_ground_truth(bank: List[Dict[str, Any]], subject: Optional[str], k: int) -> List[Dict[str, Any]]:
    """
    ground_truth는 보기-only로 k개 샘플.
    - 과목 지정이면 subject 필터(포함 매칭)
    - '전체 범위'면 전체에서 무작위
    """
    pool = bank
    if subject in GENERATOR_SUBJECTS_DEFAULT:
        def _match(s: str) -> bool:
            # generator.py의 과목 alias까지 엄밀히 쓰진 않되, 포함 매칭으로 관대하게
            s = (s or "").replace(" ", "")
            t = subject.replace(" ", "")
            return t in s or s in t
        pool = [q for q in bank if _match(q.get("subject", ""))] or bank

    if len(pool) <= k:
        picks = pool[:]  # 부족하면 전량
    else:
        picks = random.sample(pool, k)

    # 보기-only로 리셰이프 (정답/해설 제외)
    out = []
    for q in picks:
        opts = (q.get("options") or [])[:4]
        if len(opts) < 4:
            continue
        out.append({
            "question": q.get("question", ""),
            "options": opts
        })
    return out

# -------------------- 메인 --------------------
def main():
    log("경로 확인")
    log(f"EXAM_DIR    = {EXAM_DIR} (exists: {os.path.isdir(EXAM_DIR)})")
    log(f"CONCEPT_DIR = {CONCEPT_DIR} (exists: {os.path.isdir(CONCEPT_DIR)})")
    log(f"OUT_DIR     = {OUT_DIR}")
    log(f"TARGET_SAMPLES={TARGET_SAMPLES}, MAX_Q_PER_REQUEST={MAX_Q_PER_REQUEST}, MAX_CTX_CHARS={MAX_CTX_CHARS}")

    # 1) 로드
    concept_texts = load_concept_corpus(CONCEPT_DIR)  # 문자열 리스트
    problem_bank = load_problem_bank(EXAM_DIR)        # 문제 dict 리스트 (보기-only 추출용)
    log(f"개념 텍스트 조각: {len(concept_texts)}")
    log(f"문제 풀(보기 4개 이상): {len(problem_bank)}")

    # 2) 컨텍스트 전량(필요 시 컷) 1덩어리
    #    - generator는 과목별 검색도 하지만, 본 골든셋은 '동일 컨텍스트'를 전달해도 RAGAS의 비교에는 충분
    #    - 과목별로 문맥이 너무 방대하면 MAX_CTX_CHARS로 제어
    rows: List[Dict[str, Any]] = []
    for i in range(TARGET_SAMPLES):
        subject = pick_subject_for_prompt()
        prompt, k = build_user_prompt(subject)

        # ground_truth: 기존 로직 그대로 유지 (k개, 보기-only)
        gt = sample_ground_truth(problem_bank, subject, k)
        if not gt:
            continue

        # ✅ 컨텍스트: 공식 문제 2개 + 개념(컷) — subject 기반으로 매번 생성
        contexts_field = build_context_for_sample(subject, concept_texts, problem_bank)

        rows.append({
            "question": prompt,
            "ground_truth": gt,
            "contexts": contexts_field
        })
        if (i + 1) % 10 == 0:
            log(f"진행: {i+1}/{TARGET_SAMPLES}")

    # 3) 샘플 생성
    rows: List[Dict[str, Any]] = []
    for i in range(TARGET_SAMPLES):
        subject = pick_subject_for_prompt()
        prompt, k = build_user_prompt(subject)
        gt = sample_ground_truth(problem_bank, subject, k)
        if not gt:
            continue
        rows.append({
            "question": prompt,
            "ground_truth": gt,
            "contexts": contexts_field
        })
        if (i + 1) % 10 == 0:
            log(f"진행: {i+1}/{TARGET_SAMPLES}")

    # 4) 저장 (JSONL + CSV 미리보기)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(OUT_DIR, f"generator_golden_{ts}.jsonl")
    csv_path = os.path.join(OUT_DIR, f"generator_golden_{ts}.csv")

    with open(jsonl_path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # CSV는 사람이 보기 좋게 일부 요약
    csv_rows = []
    for r in rows:
        q = r["question"]
        n = len(r["ground_truth"])
        ctx_preview = (r["contexts"][0][:300] + "…") if r["contexts"] and len(r["contexts"][0]) > 300 else (r["contexts"][0] if r["contexts"] else "")
        csv_rows.append({
            "question": q,
            "gt_len": n,
            "contexts_preview": ctx_preview
        })
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    log(f"✅ 완료: {len(rows)} 샘플 저장")
    log(f"JSONL → {jsonl_path}")
    log(f"CSV   → {csv_path}")

if __name__ == "__main__":
    main()
