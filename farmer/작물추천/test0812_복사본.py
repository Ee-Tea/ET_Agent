# =====================
# Crop_Recommedations4.py (주석 강화 버전)
# - 초보자도 전체 흐름을 이해할 수 있도록 "거의 모든 줄"에 주석을 달았습니다.
# - 원본 코드의 동작을 해치지 않도록, 문자열 내부(프롬프트 등)엔 주석을 넣지 않았습니다.
# - 실행 환경: Python 3.10+, Windows 경로 예시 포함
# =====================

from dotenv import load_dotenv  # .env 파일에서 환경변수를 읽어오는 유틸 함수
import os, pandas as pd, numpy as np, re, json, hashlib, warnings  # 표준/서드파티 라이브러리 모음 (OS, 데이터 처리, 정규식, JSON, 해시, 경고 제어)
from typing import TypedDict, Optional, List, Dict, Tuple  # 타입 힌트(정적 분석용)
from numpy.linalg import norm  # 벡터의 노름 계산(코사인 유사도 계산에 사용)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트를 청크로 자르는 도구
from langchain_community.embeddings import HuggingFaceEmbeddings  # 허깅페이스 임베딩 모델 래퍼
from langchain_community.vectorstores import FAISS  # 벡터스토어로 FAISS 사용
from langchain_openai import ChatOpenAI  # OpenAI 호환형 LLM 클라이언트(여기선 Groq API 호환 엔드포인트 사용)
from langchain.prompts import PromptTemplate  # 프롬프트 템플릿 관리
from langgraph.graph import StateGraph, END  # LangGraph: 상태 기반 그래프 정의/종료 심볼
from langchain_core.documents import Document  # 문서 객체(텍스트+메타데이터)
import pdfplumber  # PDF 텍스트 추출 라이브러리
import xml.etree.ElementTree as ET  # XML 파서
from collections import defaultdict  # 기본값을 갖는 dict(그룹핑 시 편리)

# ========== 2. 상태 및 글로벌 ==========
load_dotenv()  # .env 파일 읽어 환경변수 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 환경변수에서 OpenAI API 키(여기선 Groq 엔드포인트에 전달)
EVAL_MODE = False   # 평가모드 플래그(평가 시 내부 로그 출력 여부를 결정)

class RAGState(TypedDict, total=False):  # LangGraph 상태로 주고받을 데이터의 스키마(필드 선택적)
    question: str  # 사용자 질문: 문자열(str)
    golden_answer: Optional[str]  # 평가용 정답 문장(골든): 문자열(str) 또는 None
    dfs: Optional[List[pd.DataFrame]]  # 로드한 파일들의 데이터프레임 목록: 리스트[pd.DataFrame] 또는 None
    docs: Optional[List]  # 분할된 Document 리스트: 리스트(List) 또는 None
    vectorstore: Optional['FAISS']  # 임베딩이 저장된 벡터스토어(FAISS): FAISS 인스턴스 또는 None
    context: Optional[str]  # 검색된 컨텍스트(LLM에 전달할 텍스트): 문자열(str) 또는 None
    answer: Optional[str]  # 최종 생성 답변: 문자열(str) 또는 None
    folder_path: Optional[str]  # 데이터 폴더 경로(옵션): 문자열(경로) 또는 None
    df: Optional[pd.DataFrame]  # 전체 합친 데이터프레임(옵션): pandas.DataFrame 또는 None
    context_dataframe: Optional[pd.DataFrame]  # 검색된 청크들의 상세 테이블: pandas.DataFrame 또는 None
    allowed_crops: Optional[List[str]]  # 컨텍스트에서 발견된 작물명 리스트(중복 제거): 리스트[str] 또는 None
    allowed_crops_str: Optional[str]  # 위 리스트를 문자열로 합친 값: 문자열(str) 또는 None
    retrieved_docs: Optional[List[Document]]  # 선택된 컨텍스트 문서들: 리스트[Document] 또는 None
    context_files: Optional[List[str]]  # 컨텍스트가 나온 파일 이름 목록: 리스트[str] 또는 None


def log(*args, **kwargs):  # 평가 모드일 때만 print 하는 헬퍼 함수
    if EVAL_MODE:  # EVAL_MODE True일 때만
        print(*args, **kwargs)  # 전달된 메시지를 출력


# ========== 3. 유틸 함수 ==========

def cosine_similarity(vec1, vec2) -> float:  # 두 벡터의 코사인 유사도 계산
    vec1 = np.array(vec1, dtype=float)  # 입력을 float 배열로 변환
    vec2 = np.array(vec2, dtype=float)  # 입력을 float 배열로 변환
    n1, n2 = norm(vec1), norm(vec2)  # 각 벡터의 L2 노름 계산
    return float(np.dot(vec1, vec2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0  # 둘 중 하나라도 0벡터면 0.0 반환


def _fix_number_ranges(text: str) -> str:  # 숫자 범위 표기를 깔끔하게 보정(예: '10 20' -> '10~20')
    text = re.sub(r'(?<![~\-–—])(\d{2,})\s+(\d{2,})(?![\d~\-–—])', r'\1~\2', text)  # 공백 사이 숫자 보정
    text = re.sub(r'(?<![\d~\-–—])(\d{2,})(\d{2,})(?![\d~\-–—])', r'\1~\2', text)  # 붙어있는 숫자 보정
    return text  # 보정된 문자열 반환


def clean_korean_answer(answer: str) -> str:  # 한글 답변을 깨끗하게 정리하는 함수(불필요 문자 제거 등)
    allowed = r'[^가-힣A-Za-z0-9\s\.,\?!:%/\(\)\-\–\—~\u223C\u301C\uFF5E°℃℉·µμ]'  # 허용 문자 정규식
    text = re.sub(allowed, '', str(answer))  # 허용되지 않는 문자 제거
    text = re.sub(r'([.,?!])\1+', r'\1', text)  # 반복된 구두점 하나로 축소
    text = text.replace('\r', ' ').replace('\n', ' ')  # 줄바꿈/캐리지리턴 공백으로 치환
    text = re.sub(r'\s+', ' ', text).strip()  # 다중 공백 정리
    text = _fix_number_ranges(text)  # 숫자 범위 표기 보정
    return text if len(text) >= 5 else "주어진 정보로는 답변할 수 없습니다."  # 너무 짧으면 기본 문구 반환


def normalize_text(s: str) -> str:  # 텍스트 표준화(공백/대소문자 정리)
    if s is None:  # None 방어
        return ""
    s = str(s)  # 문자열로 변환
    s = re.sub(r'\s+', ' ', s)  # 다중 공백을 하나로
    s = s.strip().lower()  # 앞뒤 공백 제거 + 소문자화
    return s  # 표준화된 텍스트 반환


def sha256_hexdigest(s: str) -> str:  # 입력 문자열의 SHA-256 해시(16진) 계산
    return hashlib.sha256(s.encode("utf-8")).hexdigest()  # 보편적 중복 체크/키 생성에 사용


def safe_cell_map(df: pd.DataFrame, func) -> pd.DataFrame:  # DataFrame 모든 셀에 안전하게 함수 적용
    if hasattr(pd.DataFrame, "map"):  # pandas 버전에 따라 map 지원 여부 확인(가상)
        return df.map(func)  # 지원되면 map 사용(빠름)
    else:
        with warnings.catch_warnings():  # 경고 무시 컨텍스트
            warnings.simplefilter("ignore", category=FutureWarning)  # 미래 경고 무시
            return df.applymap(func)  # 셀 단위 적용


# ========== 4. 파일 불러오기 ==========

def file_load_node(state: RAGState) -> RAGState:  # 데이터 폴더에서 다양한 형식 파일을 읽어 DataFrame 리스트로 반환하는 노드
    log("[Step1] file_load_node 함수 시작!")  # 평가 모드에서만 출력되는 로그
    folder_path = r"C:\Rookies_project\최종 프로젝트\Crop_Recommedations_DB"  # 데이터가 들어있는 폴더 경로(사용자 환경에 맞게 수정 가능)
    support_exts = [".csv", ".xlsx", ".pdf", ".txt", ".xml"]  # 지원하는 파일 확장자 목록
    all_dfs, success_files, failed_files = [], [], []  # 누적용 리스트 초기화
    try:
        file_list = [f for f in os.listdir(folder_path) if os.path.splitext(f)[-1].lower() in support_exts]  # 폴더 내 지원 확장자 파일만 수집
    except Exception as e:  # 폴더 접근 실패 예외 처리
        log(f"폴더 접근 실패: {e}")  # 에러 로그
        return state  # 상태 그대로 반환

    log(f"[Step1] 총 {len(file_list)}개의 파일을 불러옵니다.")  # 파일 개수 로그

    def clean_text(x):  # 파일별 로드 후 텍스트 기본 정리 함수
        if pd.isna(x):  # NaN 방어
            return ""
        return str(x).replace("<br>", " ").replace("\n", " ").replace("nan", "").strip()  # 줄바꿈/HTML 브레이크 제거 등

    for file_name in file_list:  # 각 파일에 대해 반복
        file_path = os.path.join(folder_path, file_name)  # 전체 경로 결합
        ext = os.path.splitext(file_name)[-1].lower()  # 확장자 소문자
        log(f"[Step1]    - {file_name} 불러오는 중...")  # 진행 로그
        try:
            if ext == ".csv":  # CSV 처리
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')  # 기본적으로 UTF-8 시도
                except UnicodeDecodeError:  # 인코딩 에러 시 CP949 시도(윈도우 한글 파일 대응)
                    df = pd.read_csv(file_path, encoding='cp949')
                df = safe_cell_map(df, clean_text)  # 셀 전체 정리
            elif ext == ".xlsx":  # 엑셀 처리
                df = pd.read_excel(file_path)  # 판다스로 읽기
                df = safe_cell_map(df, clean_text)  # 셀 정리
            elif ext == ".txt":  # 텍스트 파일 처리
                with open(file_path, encoding='utf-8') as f:  # UTF-8로 열기
                    text = f.read()  # 전체 읽기
                df = pd.DataFrame({"text": [clean_text(text)]})  # 단일 컬럼 DataFrame으로 감싸기
            elif ext == ".pdf":  # PDF 처리
                text = ""  # 누적 텍스트
                with pdfplumber.open(file_path) as pdf:  # PDF 열기
                    for page in pdf.pages:  # 각 페이지 순회
                        page_text = page.extract_text()  # 텍스트 추출
                        if page_text:  # 텍스트가 있으면
                            text += page_text + "\n"  # 누적
                df = pd.DataFrame({"text": [clean_text(text)]})  # DataFrame 구성
            elif ext == ".xml":  # XML 처리
                tree = ET.parse(file_path)  # XML 파싱
                root = tree.getroot()  # 루트 노드 획득
                texts = [elem.text.strip() for elem in root.iter() if elem.text]  # 모든 텍스트 노드 수집
                text = " ".join(texts)  # 공백으로 합치기
                df = pd.DataFrame({"text": [clean_text(text)]})  # DataFrame 구성
            else:
                log(f"[Step1]      ⚠️ 지원하지 않는 파일 형식: {file_name}")  # 미지원 확장자 경고
                failed_files.append(file_name)  # 실패 목록에 추가
                continue  # 다음 파일로

            df["source_file"] = file_name  # 원본 파일명 기록(추후 추적용)
            all_dfs.append(df)  # 누적 목록에 추가
            success_files.append(file_name)  # 성공 파일 기록
            log(f"[Step1]        ✔️ {file_name} 로딩 완료 (행 개수: {len(df)})")  # 성공 로그
        except Exception as e:  # 파일별 로딩 실패 처리
            log(f"[Step1]        ❌ {file_name} 로딩 실패: {e}")  # 에러 로그
            failed_files.append(file_name)  # 실패 목록에 추가

    log(f"[Step1] 성공 파일 ({len(success_files)}) :", *["    ✔️ " + fn for fn in success_files], sep="\n")  # 성공 리스트 출력
    log(f"[Step1] 실패 파일 ({len(failed_files)}) :", *["    ❌ " + fn for fn in failed_files], sep="\n")  # 실패 리스트 출력
    log(f"[Step1]    ✔️ 모든 파일 로딩 완료 (총 DataFrame 개수: {len(all_dfs)})")  # 요약 로그
    return {**state, "dfs": all_dfs}  # 상태에 dfs 추가하여 반환


# ========== 5. 상수/설정 ==========
VECTORSTORE_PATH = "my_faiss_db_all_columns"  # FAISS 벡터스토어 저장 경로(폴더)
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"  # 한국어 멀티태스크 임베딩 모델 이름


# ========== 6. 텍스트 분할 및 Document ==========

def split_node(state: RAGState) -> RAGState:  # 로드된 DF들을 합치고, LangChain Document로 만들고, 청크로 분할하는 노드
    log("[2/5] 텍스트 분할(split) 중입니다...")  # 상태 로그
    dfs = state.get("dfs")  # 이전 노드에서 저장한 DataFrame 리스트 가져오기
    if not dfs:  # dfs가 비어 있으면
        raise ValueError("불러온 데이터프레임이 없습니다.")  # 예외 발생(중단)
    df = pd.concat(dfs, ignore_index=True)  # 여러 DF를 하나로 합치기
    if "source_file" not in df.columns:  # 추적용 컬럼이 없다면
        raise ValueError("DataFrame에 source_file 컬럼이 없습니다!")  # 예외 발생

    crop_name_col = "작물명"  # 작물명이 들어있는 컬럼명(데이터 형식에 맞춰 조정 가능)
    context_col = "작물정보"  # 작물 상세 정보 컬럼명(없으면 text 컬럼 사용)

    # 1) 원본 Document 생성(행 단위)
    raw_docs = []  # Document 누적 리스트
    for _, row in df.iterrows():  # 각 행 순회
        crop_name = str(row[crop_name_col]) if crop_name_col in df.columns else ""  # 작물명 안전 추출
        if context_col in df.columns:  # 작물정보 컬럼이 있으면
            crop_info = str(row[context_col])  # 해당 텍스트 사용
        elif "text" in df.columns:  # 없으면 text 컬럼 사용(예: PDF/XML/TXT)
            crop_info = str(row["text"])  # 텍스트 사용
        else:
            crop_info = ""  # 둘 다 없으면 빈 문자열

        context_text = f"{crop_name}. {crop_info}" if crop_name else crop_info  # 작물명. 정보 형식으로 합치기
        raw_docs.append(  # Document 생성하여 누적
            Document(
                page_content=context_text,  # 본문 텍스트
                metadata={
                    "source_file": row["source_file"],  # 출처 파일명
                    "crop_name": crop_name,  # 작물명 메타데이터
                },
            )
        )

    # 2) 청크 전략 두 가지 (짧은/긴) 병행
    splitter_short = RecursiveCharacterTextSplitter(  # 짧은 청크 분할기
        chunk_size=300,  # 청크 최대 길이(조금 확장)
        chunk_overlap=50,  # 겹침 길이(문맥 보존)
        separators=["\n", "·", "•", "▶", "②", "①", "—", "-", "。", ".", "!", "?", ":", ";"]  # 문장/기호 기준 분할
    )
    splitter_long = RecursiveCharacterTextSplitter(  # 긴 청크 분할기
        chunk_size=800,  # 더 긴 문맥 유지
        chunk_overlap=120,  # 겹침도 더 길게
        separators=["\n", "·", "•", "▶", "②", "①", "—", "-", "。", ".", "!", "?", ":", ";"]  # 동일 구분자
    )

    def chunk_and_tag(docs, splitter, variant_label):  # 분할+메타데이터 태깅 헬퍼
        out = []  # 결과 리스트
        split_docs = splitter.split_documents(docs)  # 주어진 문서 리스트 전체를 분할
        per_key_counter = {}  # 파일/작물명/전략별 로컬 인덱스 카운터
        for d in split_docs:  # 분할된 각 문서에 대해
            key = (d.metadata.get("source_file",""), d.metadata.get("crop_name",""), variant_label)  # 키 구성
            idx = per_key_counter.get(key, 0)  # 해당 키의 현재 인덱스 조회
            per_key_counter[key] = idx + 1  # 다음 인덱스를 위해 +1

            base = f"{normalize_text(d.page_content)}|{d.metadata.get('source_file','')}|{d.metadata.get('crop_name','')}|{variant_label}|{idx}"  # 해시용 바탕 문자열
            d.metadata.update({  # 메타데이터 확장
                "chunk_variant": variant_label,   # 분할 전략 라벨('s' or 'l')
                "chunk_index": idx,  # 동일 키 내 순번
                "content_hash": sha256_hexdigest(base),  # 내용 기반 고유 해시(중복 방지)
            })
            out.append(d)  # 결과에 추가
        return out  # 분할+태그 완료 문서 리스트 반환

    short_docs = chunk_and_tag(raw_docs, splitter_short, "s")  # 짧은 청크 생성
    long_docs  = chunk_and_tag(raw_docs, splitter_long,  "l")  # 긴 청크 생성
    split_docs = short_docs + long_docs  # 두 전략 결과를 합침

    log(f"    ✔️ 분할 청크 수(짧은): {len(short_docs)} / (긴): {len(long_docs)} / 합계: {len(split_docs)}")  # 통계 로그
    if context_col in df.columns:  # 원본 csv 기반일 때
        total_length = int(sum(len(str(val)) for val in df[context_col]))  # 전체 텍스트 길이 합
        avg_len = int(total_length / max(1, len(split_docs)))  # 청크당 평균 길이
        if EVAL_MODE:  # 평가 모드에서만
            print(f"csv 전체 텍스트 길이: {total_length}자, 평균 청크 길이: {avg_len}자")  # 출력

    return {**state, "df": df, "docs": split_docs}  # 상태에 df/docs 저장 후 반환



def embedding_and_vectorstore_node(state: RAGState) -> RAGState:  # 임베딩 계산 및 FAISS 벡터스토어 구축/업데이트 노드
    log("[Step3] 임베딩 및 벡터스토어 작업 중...")  # 상태 로그
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})  # CPU로 임베딩 모델 로드
    docs = state["docs"]  # 분할된 문서 리스트 가져오기

    def ensure_hash(doc: Document) -> str:  # 문서에 content_hash가 없으면 생성 보장
        h = doc.metadata.get("content_hash")  # 기존 해시 조회
        if not h:  # 없으면
            base = f"{normalize_text(doc.page_content)}|{doc.metadata.get('source_file','')}|{doc.metadata.get('crop_name','')}|{doc.metadata.get('chunk_variant','')}|{doc.metadata.get('chunk_index','')}"  # 해시 입력
            h = sha256_hexdigest(base)  # 해시 생성
            doc.metadata["content_hash"] = h  # 메타데이터에 기록
        return h  # 해시 반환

    if os.path.exists(VECTORSTORE_PATH):  # 기존 벡터스토어 폴더가 있으면(증분 업데이트)
        log("[Step3] 기존 벡터스토어가 존재! 신규 문서만 추가")  # 로그
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)  # 기존 로드

        existing_hashes = set()  # 기존 청크들의 해시 집합
        for d in vectorstore.docstore._dict.values():  # 저장된 문서 순회(내부 구조 접근)
            h = d.metadata.get("content_hash")  # 해시 조회
            if not h:  # 없으면
                base = f"{normalize_text(d.page_content)}|{d.metadata.get('source_file','')}|{d.metadata.get('crop_name','')}|{d.metadata.get('chunk_variant','')}|{d.metadata.get('chunk_index','')}"  # 동일 규칙으로 생성
                h = sha256_hexdigest(base)  # 생성
            existing_hashes.add(h)  # 집합에 추가

        unique_docs = [doc for doc in docs if ensure_hash(doc) not in existing_hashes]  # 신규 문서만 필터
        log(f"[Step3] 신규 청크 수: {len(unique_docs)}")  # 개수 로그
        if unique_docs:  # 신규가 있으면
            vectorstore.add_documents(unique_docs)  # 문서 추가
            vectorstore.save_local(VECTORSTORE_PATH)  # 저장
            log("[Step3] 신규 임베딩 추가 및 저장 완료")  # 완료 로그
        else:
            log("[Step3] 추가할 신규 청크 없음. 기존 벡터스토어 재사용")  # 재사용 로그
    else:  # 벡터스토어가 없으면 새로 생성
        log("[Step3] 벡터스토어 새로 생성/저장")  # 로그
        _ = [ensure_hash(doc) for doc in docs]  # 모든 문서에 해시 보장
        vectorstore = FAISS.from_documents(docs, embeddings)  # 문서로부터 FAISS 인덱스 생성
        vectorstore.save_local(VECTORSTORE_PATH)  # 디스크에 저장
        log("[Step3] 벡터스토어 생성 및 저장 완료")  # 완료 로그

    return {**state, "vectorstore": vectorstore}  # 상태에 벡터스토어 저장 후 반환


# ========== 8. 컨텍스트 검색 ==========

def retriever_node(state: RAGState) -> RAGState:  # 질문에 대한 관련 컨텍스트를 벡터 검색으로 찾는 노드
    log("[Step4] 컨텍스트 검색(retriever_node) 실행")  # 로그
    vectorstore = state["vectorstore"]  # FAISS 핸들 가져오기
    user_question = state["question"]  # 사용자 질문 텍스트

    docs_with_scores: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(user_question, k=20)  # 유사 문서 20개 검색(점수 포함)
    docs_with_scores.sort(key=lambda x: x[1])  # 점수 기준 정렬(낮을수록 유사하다는 구현일 수 있음)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})  # 동일 임베딩 로더
    query_vec = embeddings.embed_query(user_question)  # 질문을 임베딩 벡터로 변환

    texts_for_embed = [doc.page_content for doc, _ in docs_with_scores]  # 문서 본문만 뽑아 리스트로
    if texts_for_embed:  # 비어있지 않으면
        doc_vecs = embeddings.embed_documents(texts_for_embed)  # 각 문서 본문을 임베딩
    else:
        doc_vecs = []  # 없으면 빈 리스트

    groups = defaultdict(list)  # 작물명으로 그룹핑할 dict: {crop_name: [(doc, cos), ...]}
    cosine_map = {}  # 문서 id -> 코사인값 저장(표시용)
    for (doc, _), dvec in zip(docs_with_scores, doc_vecs):  # 검색 결과와 임베딩을 함께 순회
        cos = cosine_similarity(query_vec, dvec)  # 질문-문서 코사인 유사도 계산
        cosine_map[id(doc)] = cos  # 문서 식별자 기준으로 저장
        crop = (doc.metadata.get("crop_name") or "").strip()  # 메타데이터에서 작물명 추출
        groups[crop].append((doc, cos))  # 해당 작물 그룹에 (문서, 유사도) 추가

    ranked = []  # 작물별 랭킹용 리스트
    for crop, items in groups.items():  # 각 작물 그룹 순회
        if not crop:  # 작물명이 비어있으면 스킵
            continue
        max_sim = max(s for _, s in items)  # 그 작물 그룹 내 최대 유사도
        cnt = len(items)  # 그 작물 그룹의 문서 수
        ranked.append((crop, -max_sim, -cnt))  # 최대유사도, 개수를 음수로 넣어 오름차순 정렬 시 내림차순 효과
    ranked.sort(key=lambda x: (x[1], x[2]))  # max_sim 우선, 동률 시 cnt 기준 정렬

    selected_crops = [r[0] for r in ranked[:2]]  # 상위 2개 작물 선택(없는 경우 대비 아래 else 처리)
    if selected_crops:  # 선택된 작물이 있으면
        context_docs: List[Document] = []  # 컨텍스트로 사용할 문서 리스트
        for c in selected_crops:  # 각 작물에 대해
            context_docs.extend([d for d, _ in sorted(groups[c], key=lambda x: -x[1])[:3]])  # 유사도 상위 3개 문서 선택
    else:
        context_docs = [doc for doc, _ in docs_with_scores[:5]]  # 작물명 비어있으면 상위 5개 문서 사용

    allowed_unique = sorted(set([d.metadata.get("crop_name", "") for d in context_docs if d.metadata.get("crop_name")]))  # 컨텍스트에서 나타난 작물명(중복 제거 정렬)
    allowed_str = ", ".join(allowed_unique[:20]) if allowed_unique else ""  # 최대 20개를 쉼표로 연결
    context_text = "\n".join(doc.page_content for doc in context_docs) if context_docs else ""  # LLM에 줄 컨텍스트 텍스트

    df_context = pd.DataFrame([  # 컨텍스트 상세를 표 형태로 구성(디버깅/로그/평가에 활용)
        {
            "file": doc.metadata.get("source_file", ""),  # 출처 파일명
            "crop_name": doc.metadata.get("crop_name", ""),  # 작물명
            # "chunk_text": doc.page_content[:110] + ("..." if len(doc.page_content) > 110 else ""),  # 요약본(주석 처리)
            "chunk_text": doc.page_content,  # 청크 전체 텍스트(길 수 있음)
            "cosine_similarity": round(cosine_map.get(id(doc), 0.0), 4),  # 질문-문서 코사인값(소수 4자리)
            "all_text": doc.page_content,  # 전체 텍스트(중복)
        }
        for doc in context_docs
    ])

    return {  # 다음 노드에서 사용할 정보들을 상태에 담아 반환
        **state,
        "context": context_text,  # LLM에 투입할 컨텍스트
        "retrieved_docs": context_docs,  # 선택된 문서 객체들
        "context_files": [doc.metadata.get("source_file", "") for doc in context_docs],  # 출처 파일명 목록
        "context_dataframe": df_context,  # 표 형태 컨텍스트(평가 출력용)
        "allowed_crops": allowed_unique,  # 컨텍스트에서 관측된 작물명 리스트
        "allowed_crops_str": allowed_str,  # 문자열 버전
    }


# ========== 9. 답변 생성 ==========

def generate_answer_node(state: RAGState) -> RAGState:  # LLM을 호출해 최종 답변을 생성하는 노드
    log("[Step5] 답변 생성(LangChain LLM 호출) 중입니다...")  # 로그

    cleaned_answer = "주어진 정보로는 답변할 수 없습니다."  # 기본값(정보 부족 시)
    context = state.get("context", "")  # 컨텍스트 텍스트 가져오기
    if not context.strip():  # 비어 있으면
        log("[Step5] 답변 생성 완료 (context 없음)")  # 로그
        return {**state, "answer": cleaned_answer}  # 기본 답변으로 반환
    
    allowed_str = state.get("allowed_crops_str", "")  # (현재 프롬프트에 직접 쓰이진 않지만 남겨둠)

    # PromptTemplate: 아래는 LLM에게 줄 지시문과 입력 슬롯 정의(문자열 내부는 변경하면 동작 바뀜)
    prompt_tmpl = PromptTemplate.from_template("""
    너는 대한민국 농업 작물 추천 분야의 전문가이자, 친근하게 상담해 주는 작물 추천 가이드야. 항상 한국어로 답변해줘.

    [핵심 원칙]
    1) <context> 안에서만 답해. context에 없는 사실은 추가하지 마.
    2) 대표 작물명만 사용해서 추천해. 품종·계통·상표명 금지.
    3) 숫자·월·기간은 <context> 값을 그대로 써(임의 보정 금지).
    4) 답변할때, "<context>","컨텍스트" 등 언급 하지마.

    [길이 규칙]
    - 기본 3~4문장, 정보가 많으면 최대 8~10문장.

    [내용 가이드]
    - 추천 작물 1~3개 + 근거(기후/토양/재배 환경/리스크).
    - 재배 핵심(파종/정식, 일조·온도, 토양/배수, 물주기/관리, 병해충, 수확).
    - 시작 팁 2~3개, 조건 달라질 때 대안 1개.

    [정보 부족 시]
    - “주어진 정보로는 답변할 수 없습니다.” 한 줄만 출력.

    <context>
    {context}
    </context>

    [질문]
    {question}
    """)  # 여기서 {context}, {question} 슬롯이 실제 값으로 치환됨

    _prompt = prompt_tmpl.format(  # 템플릿을 실제 값으로 포맷팅
        context=context,  # 위에서 준비한 컨텍스트 넣기
        question=state.get("question", ""),  # 사용자 질문 넣기
        allowed_crops=allowed_str if allowed_str else "(없음)"  # (현재 템플릿에 사용되지 않지만 남김)
    )

    try:
        if EVAL_MODE:  # 평가 모드 설정(일관성 높은 출력)
            temp_val = 0.2  # 낮은 temperature로 랜덤성 줄이기
            max_tok = 1024  # 충분한 토큰 허용
        else:  # 일반 채팅 모드 설정(표현 다양성)
            temp_val = 0.7  # 창의성/다양성 증가
            max_tok = 512  # 적당한 길이 제한

        llm = ChatOpenAI(  # OpenAI 호환형 클라이언트 초기화(여기선 Groq 엔드포인트 사용)
            api_key=OPENAI_API_KEY,  # .env에서 읽은 키 사용(환경에 따라 GROQ 키를 여기에 넣기도 함)
            base_url="https://api.groq.com/openai/v1",  # Groq OpenAI 호환 API 엔드포인트
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # 사용할 모델 이름
            temperature=temp_val,  # 위에서 정한 temperature
            max_tokens=max_tok  # 최대 토큰 수
        )

        msg = llm.invoke(_prompt)  # LLM에 프롬프트를 보내 응답 받기

        # ✅ 후처리 제거 — LLM 출력 그대로 사용(문장 정제는 별도 필요 시 clean_korean_answer로 가능)
        final_answer = msg.content if hasattr(msg, "content") else str(msg)  # 메시지 객체 형태에 안전하게 대응

    except Exception as e:  # LLM 호출 중 예외 처리
        log(f"[Step5] LLM 호출 오류: {e}")  # 오류 로그
        final_answer = "주어진 정보로는 답변할 수 없습니다."  # 실패 시 기본 답변

    log("[Step5] 답변 생성 완료")  # 완료 로그
    return {**state, "answer": final_answer}  # 상태에 answer 저장 후 반환


# ========== 10. 평가 출력/평가 ==========

def print_context_dataframe(df):  # 검색에 사용된 컨텍스트 청크들을 표로 출력하고 누적 로그 파일에 저장
    if isinstance(df, pd.DataFrame) and not df.empty:  # DF가 유효하고 비어있지 않으면
        print("\n[참고한 context/chunk 목록]")  # 안내 출력
        
        # ▼▼ 파일 저장 코드 (누적 저장) ▼▼
        log_path = "used_chunks_log.txt"  # 로그 파일 경로(덮어쓰기 아님)
        try:
            with open(log_path, "a", encoding="utf-8") as f:  # 추가 모드로 열기
                f.write("[사용한 컨텍스트 청크]\n")  # 헤더 기록
                f.write(df[["crop_name",  "chunk_text"]].to_string(index=True))  # 작물명과 청크 텍스트 저장
                f.write("\n-------------------------\n")  # 구분선
        except Exception as e:  # 파일 저장 오류 처리
            print(f"[로그 저장 오류] {e}")  # 오류 출력
        # ▲▲ 파일 저장 코드 ▲▲
        
        print(df[["file", "crop_name", "cosine_similarity", "chunk_text"]].to_string(index=True))  # 표 콘솔 출력
        
    else:
        print("\n[참고한 context/chunk 없음]")  # 비어있을 때 메시지


def evaluate_chatbot(app, golden_dataset: List[Dict[str, str]]):  # 골든셋을 이용해 챗봇 성능(유사도) 평가
    print("\n--- 챗봇 성능 평가 시작 (유사도 기반) ---")  # 시작 배너
    evaluation_results = []  # 결과 누적 리스트
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})  # 평가용 임베딩 로더
    SIMILARITY_THRESHOLD = 0.75  # 합격 기준(골든↔생성 답변 코사인 유사도 임계값)

    for i, data in enumerate(golden_dataset):  # 각 샘플 반복
        question, golden_answer = data['question'], data['answer']  # 질문/정답 추출
        print(f"\n[평가 {i+1}] 질문: {question}")  # 진행 출력
        
        # ▼▼ 파일 저장 코드 (누적 저장) ▼▼
        log_path = "used_chunks_log.txt"  # 같은 로그 파일 사용
        try:
            with open(log_path, "a", encoding="utf-8") as f:  # 추가 기록
                f.write(f"\n----- [평가 {i+1}] -----\n")  # 구분 헤더
                f.write(f"[질문] {question}\n")  # 질문 기록
        except Exception as e:
            print(f"[로그 저장 오류] {e}")  # 오류 출력
        # ▲▲ 파일 저장 코드 ▲▲
        
        print(f"  - 정답: {golden_answer}")  # 골든 답변 표시
        try:
            final_state = app.invoke({"question": question, "golden_answer": golden_answer})  # 그래프 실행하여 답변 생성
            generated_answer = final_state['answer']  # 생성 답변 추출
            df_context = final_state.get('context_dataframe')  # 컨텍스트 DF 추출
            print(f"  - 생성된 답변: {generated_answer}")  # 생성 답변 출력
            print_context_dataframe(df_context)  # 컨텍스트 표 출력 및 로그 저장

            golden_emb = embeddings.embed_query(golden_answer)  # 골든 답변 임베딩
            generated_emb = embeddings.embed_query(generated_answer)  # 생성 답변 임베딩
            answer_similarity = cosine_similarity(golden_emb, generated_emb)  # 두 임베딩의 코사인 유사도

            search_similarity_max = float(df_context['cosine_similarity'].max()) if df_context is not None and not df_context.empty else 0.0  # 검색된 컨텍스트 중 최대 코사인값

            is_correct = bool(answer_similarity >= SIMILARITY_THRESHOLD)  # 임계값 통과 여부
            print(f"  - 검색 유사도(질문↔청크): {search_similarity_max:.4f}")  # 검색 최대 유사도 표시
            print(f"  - 답변 유사도(골든↔답변): {answer_similarity:.4f} (합격 기준: {SIMILARITY_THRESHOLD})")  # 답변 유사도 표시
            print(f"  - 정답 여부: {'✅ 정답' if is_correct else '❌ 오답'}")  # 합격/불합격 출력

            evaluation_results.append({  # 결과 누적
                'question': question,
                'golden_answer': golden_answer,
                'generated_answer': generated_answer,
                'search_similarity_max': search_similarity_max,
                'answer_similarity': float(answer_similarity),
                'is_correct': is_correct
            })
        except Exception as e:  # 그래프 실행 중 예외 처리
            print(f"  - 오류 발생: {e}")  # 오류 출력
            evaluation_results.append({  # 실패 항목도 결과에 기록
                'question': question,
                'golden_answer': golden_answer,
                'generated_answer': '오류 발생',
                'search_similarity_max': 0.0,
                'answer_similarity': 0.0,
                'is_correct': False
            })

    correct = sum(1 for res in evaluation_results if res['is_correct'])  # 정답 개수 집계
    total = len(evaluation_results)  # 총 평가 개수
    print("\n--- 챗봇 성능 평가 완료 ---")  # 완료 배너
    print(f"\n총 질문 수: {total}")  # 통계 출력
    print(f"정답 수: {correct}")  # 통계 출력
    print(f"정확도: {(correct/total)*100:.2f}%")  # 정확도 퍼센트 출력
    with open('evaluation_report.json', 'w', encoding='utf-8') as f:  # 상세 결과를 JSON 파일로 저장
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)  # 한글 깨짐 방지 및 보기 좋은 들여쓰기
    print("상세 평가 결과가 'evaluation_report.json' 파일에 저장되었습니다.")  # 저장 안내


# ========== 11. 전체 RAG 파이프라인 ==========

def build_rag_graph():  # LangGraph로 파이프라인 구성(노드 연결 흐름)
    graph = StateGraph(RAGState)  # 상태 타입을 지정한 그래프 생성
    graph.add_node("file_load", file_load_node)  # 파일 로딩 노드 등록
    graph.add_node("split", split_node)  # 분할 노드 등록
    graph.add_node("embedding_vectorstore", embedding_and_vectorstore_node)  # 임베딩/벡터스토어 노드 등록
    graph.add_node("retriever", retriever_node)  # 검색 노드 등록
    graph.add_node("generate_answer", generate_answer_node)  # 답변 생성 노드 등록
    graph.add_edge("file_load", "split")  # 파일 로딩 → 분할
    graph.add_edge("split", "embedding_vectorstore")  # 분할 → 임베딩/벡터스토어
    graph.add_edge("embedding_vectorstore", "retriever")  # 임베딩/벡터스토어 → 검색
    graph.add_edge("retriever", "generate_answer")  # 검색 → 답변 생성
    graph.add_edge("generate_answer", END)  # 답변 생성 → 종료
    graph.set_entry_point("file_load")  # 시작 노드 지정
    return graph.compile()  # 그래프 컴파일(실행 가능한 앱 반환)


# ========== 12. 평가/실행 진입점 ==========

def load_golden_dataset(file_path: str):  # CSV에서 골든셋(질문/정답) 로드
    return pd.read_csv(file_path, encoding='utf-8').to_dict('records')  # 레코드 리스트(dict 리스트)로 반환


if __name__ == "__main__":  # 스크립트를 직접 실행했을 때만 동작하는 메인 블록
    print("\n---농작물 챗봇 에이전트 시작---")  # 시작 배너 출력
    app = build_rag_graph()  # 그래프 빌드 및 컴파일(앱 객체 획득)
    mode = input("실행 모드를 선택하세요 (1=챗봇, 2=평가): ").strip()  # 사용자에게 모드 선택 받기

    if mode == "2":  # 평가 모드 선택 시
        EVAL_MODE = True  # 평가 모드 활성화(로그 상세)
        golden_path = r"C:\Rookies_project\최종 프로젝트\Project_test\Goldenset_test1.csv"  # 기본 골든셋 경로
        # golden_path = r"C:\Rookies_project\최종 프로젝트\Project_test\Goldenset_test2.csv"  # 다른 골든셋 예시(주석)
        print(f"[평가모드] 골든셋 CSV: {golden_path}")  # 경로 안내
        try:
            df_check = pd.read_csv(golden_path, encoding="utf-8")  # 파일 확인 차 읽기
            print("\n[골든셋 CSV 컬럼 구조]:", list(df_check.columns))  # 컬럼명 출력
            print("[골든셋 샘플 데이터]")  # 안내
            print(df_check.head(3))  # 샘플 출력(상위 3개)
        except Exception as e:  # 파일 읽기 실패 시
            print("[오류] 골든셋 파일을 읽을 수 없습니다:", e)  # 오류 안내
            exit(1)  # 비정상 종료

        golden_dataset = load_golden_dataset(golden_path)  # 골든셋 로드(레코드 리스트)
        golden_dataset = golden_dataset[:5]  # 평가 샘플을 5개로 제한(테스트 용)
        evaluate_chatbot(app, golden_dataset)  # 평가 실행

    else:  # 모드가 1 또는 기타일 때: 채팅 모드
        EVAL_MODE = False  # 평가 모드 비활성화(로그 간소)
        warnings.filterwarnings("ignore")  # 경고 메시지 무시(콘솔 깔끔하게)
        while True:  # 무한 루프(사용자가 종료할 때까지)
            user_question = input("\n질문을 입력하세요: ")  # 사용자 질문 입력 받기
            if user_question.lower() in ["exit", "quit"]:  # 종료 명령 처리
                print("챗봇을 종료합니다.")  # 안내 출력
                break  # 루프 탈출
            if not user_question.strip():  # 빈 입력 처리
                print("질문을 입력해주세요.")  # 안내
                continue  # 다시 입력 받기
            try:
                state = {"question": user_question}  # 상태 딕셔너리 준비
                final_state = app.invoke(state)  # 그래프 실행하여 응답 생성
                print(final_state["answer"])  # 최종 답변 출력
            except Exception as e:  # 실행 중 예외 처리
                print(f"\n오류가 발생했습니다: {e}")  # 오류 안내
                print("다시 시도해주세요.")  # 재시도 안내
    print("\n---농작물 챗봇 에이전트 종료---")  # 종료 배너 출력
