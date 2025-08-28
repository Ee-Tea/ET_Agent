# ──────────────────────────────────────────────────────────────────────────────
# 임포트 섹션: 필요한 라이브러리 및 모듈을 가져옵니다.
# ──────────────────────────────────────────────────────────────────────────────
import os                                           # 운영체제 기능(파일 경로, 환경 변수)을 사용하기 위해 임포트
import sys                                          # 시스템 관련 기능(진행률 표시)을 사용하기 위해 임포트
import time                                         # 시간 관련 기능(ETA 계산)을 사용하기 위해 임포트
import math                                         # 수학 관련 기능(시간 포맷팅)을 사용하기 위해 임포트
from typing import TypedDict, Optional, List        # 타입 힌팅을 지원하기 위해 임포트
from pathlib import Path                            # 파일 시스템 경로를 객체처럼 다루기 위해 임포트
from dotenv import load_dotenv, find_dotenv         # .env 파일에서 환경 변수를 로드하기 위해 임포트
from collections import defaultdict                 # 파일별로 문서를 그룹화하기 위해 defaultdict 임포트

# LangChain 관련 클래스들을 임포트합니다.
from langchain_community.document_loaders import TextLoader, PyPDFLoader  # 텍스트 및 PDF 파일 로더 임포트
from langchain_text_splitters import RecursiveCharacterTextSplitter     # 텍스트를 청크로 분할하는 클래스 임포트
from langchain_huggingface import HuggingFaceEmbeddings                 # HuggingFace 임베딩 모델을 사용하기 위해 임포트
from langchain_community.vectorstores import Milvus                     # Milvus 벡터스토어를 사용하기 위해 임포트
from langchain_core.documents import Document                         # LangChain의 기본 문서 객체 구조 임포트

# ──────────────────────────────────────────────────────────────────────────────
# 환경 변수 로드 및 기본 설정
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())                          # .env 파일을 찾아 환경 변수를 로드

# Milvus / Embedding 설정
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")           # Milvus 서버 주소 설정
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:milvus")                 # Milvus 인증 토큰 설정
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "test")              # 사용할 Milvus 컬렉션 이름 설정
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask") # 사용할 한국어 임베딩 모델 이름 설정

# 입력 문서 폴더
DOCS_DIR = Path(os.getenv("DOCS_DIR", r"C:\Rookies_project\cropinfo"))   # 문서가 저장된 폴더 경로 설정
DOCS_DIR.mkdir(parents=True, exist_ok=True)         # 폴더가 없으면 생성

# 청크/임베딩 파라미터
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))     # 텍스트를 나눌 청크 크기 설정
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120")) # 청크 간 중첩될 글자 수 설정

# 임베딩 진행률 설정
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))             # 한 번에 임베딩할 문서(청크) 수 설정
EMBED_PROGRESS_INTERVAL = float(os.getenv("EMBED_PROGRESS_INTERVAL", "0.2")) # 진행률을 업데이트할 시간 간격(초) 설정

# ──────────────────────────────────────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(            # HuggingFace 임베딩 모델 초기화
    model_name=EMBED_MODEL_NAME,                    # 사용할 모델 이름 지정
    model_kwargs={"device": "cpu"}                  # 모델을 CPU에서 실행하도록 설정
)
EMBEDDING_DIM = len(embedding_model.embed_query("test")) # 임베딩 벡터의 차원 수 계산

# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────────────────────────────────────
def _format_eta(seconds: Optional[float]) -> str:   # 남은 시간을 보기 좋은 형식(HH:MM:SS)으로 변환하는 함수
    if not seconds or seconds < 0 or math.isinf(seconds) or math.isnan(seconds): # 유효하지 않은 시간 값 처리
        return "--:--"
    m, s = divmod(int(seconds), 60)                 # 초를 분과 초로 분리
    if m >= 60:                                     # 60분이 넘으면 시간으로 변환
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _render_progress(prefix: str, done: int, total: int, start_ts: float, task_name: Optional[str] = None) -> None: # 콘솔에 진행률 바를 그려주는 함수
    done = min(done, total)                         # 완료된 개수가 전체 개수를 넘지 않도록 조정
    percent = int((done / total) * 100) if total else 100 # 완료율(%) 계산
    elapsed = time.time() - start_ts                # 경과 시간 계산
    rate = (done / elapsed) if elapsed > 0 else None # 처리 속도(개/초) 계산
    remain = ((total - done) / rate) if rate else None # 남은 시간(초) 계산
    eta = _format_eta(remain)                       # 남은 시간을 포맷팅
    bar_len = 24                                    # 진행률 바의 길이
    filled = int(bar_len * percent / 100)           # 채워질 바의 길이 계산
    bar = "█" * filled + "░" * (bar_len - filled)   # 진행률 바 문자열 생성
    task_str = f"| {task_name}" if task_name else "" # 현재 작업 이름 문자열 생성
    full_line = f"{prefix} [{bar}] {percent:3d}%  ({done}/{total})  ETA {eta}{task_str}" # 전체 출력 라인 생성
    padded_line = full_line.ljust(120)              # 라인 길이를 고정하여 깜빡임 방지

    sys.stdout.write(f"\r{padded_line}")            # 현재 라인에 덮어쓰기
    sys.stdout.flush()                              # 버퍼를 비워 즉시 출력
    
    if done >= total:                               # 작업이 완료되면
        sys.stdout.write("\n")                      # 줄바꿈 문자를 출력
        
class ProgressEmbeddings:                           # 임베딩 진행률을 추적하고 표시하는 래퍼 클래스
    def __init__(self, base: HuggingFaceEmbeddings, batch_size: int = 32, desc: str = "임베딩"): # 생성자
        self.base = base                            # 원본 임베딩 모델
        self.batch_size = max(1, batch_size)        # 배치 크기 설정
        self.desc = desc                            # 작업 설명
        self.task_name = ""                         # 현재 처리 중인 파일 이름
        self.total_texts = 1                        # 현재 파일의 전체 청크 수
        self._last_print = 0.0                      # 마지막으로 진행률을 출력한 시간

    def update_task(self, task_name: str, total_texts: int): # 새로운 파일 처리 시 진행률 정보를 업데이트하는 메서드
        """진행률 표시줄에 표시될 작업 이름과 전체 개수를 업데이트합니다."""
        self.task_name = task_name                  # 작업 이름(파일명) 갱신
        self.total_texts = max(total_texts, 1)      # 전체 텍스트(청크) 수 갱신
        self._last_print = 0.0                      # 마지막 출력 시간 초기화

    def embed_query(self, text: str) -> List[float]: # 단일 텍스트(쿼리)를 임베딩하는 메서드
        return self.base.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]: # 여러 텍스트(문서)를 임베딩하는 메서드
        n = len(texts)                              # 현재 배치 내 텍스트 수
        total = self.total_texts                    # 처리할 파일의 전체 청크 수
        start_ts = time.time()                      # 시작 시간 기록
        results: List[List[float]] = []             # 임베딩 결과를 저장할 리스트
        processed = 0                               # 처리된 청크 수

        task_str = f" ({self.task_name})" if self.task_name else "" # 작업 이름 문자열
        print(f"🧮 {self.desc}{task_str} 시작: 총 {total}개 청크 | 배치 {self.batch_size}")

        for i in range(0, n, self.batch_size):      # 배치 크기만큼 반복 처리
            batch = texts[i:i + self.batch_size]    # 현재 처리할 배치
            emb = self.base.embed_documents(batch)  # 실제 임베딩 수행
            results.extend(emb)                     # 결과 리스트에 추가
            processed = min(i + len(batch), total)  # 처리된 개수 업데이트

            now = time.time()                       # 현재 시간
            if now - self._last_print >= EMBED_PROGRESS_INTERVAL or processed == total: # 일정 시간이 지났거나 완료되었으면
                _render_progress(f"🔄 {self.desc}", processed, total, start_ts, task_name=self.task_name) # 진행률 표시
                self._last_print = now              # 마지막 출력 시간 갱신

        _render_progress(f"✅ {self.desc}", total, total, start_ts, task_name=self.task_name) # 최종 완료 상태 표시
        return results

# ──────────────────────────────────────────────────────────────────────────────
# 상태 정의
# ──────────────────────────────────────────────────────────────────────────────
class IngestState(TypedDict):                       # LangGraph에서 노드 간에 전달될 데이터의 구조를 정의
    docsPath: str                                   # 문서 폴더 경로
    files: List[str]                                # 처리할 파일 목록
    rawDocs: List[Document]                         # 로드 및 분할된 모든 문서 청크
    vectorstore: Optional[Milvus]                   # Milvus 벡터스토어 객체
    inserted: int                                   # DB에 삽입된 청크 수
    collectionName: str                             # Milvus 컬렉션 이름

# ──────────────────────────────────────────────────────────────────────────────
# LangGraph 노드들
# ──────────────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END         # LangGraph의 StateGraph와 END를 임포트
from langchain_core.runnables.graph import MermaidDrawMethod # 그래프 시각화 도구 임포트
from pymilvus import connections, MilvusClient, DataType # Milvus 연결 및 클라이언트 도구 임포트

def ensure_milvus_node(state: IngestState) -> IngestState: # Milvus 연결을 확인하고 컬렉션을 정리하는 노드
    print("🧩 노드: ensure_milvus (컬렉션 확인)")

    if "default" in connections.list_connections(): # 기존 'default' 연결이 있으면
        connections.disconnect("default")           # 연결 해제
    connections.connect(host="localhost", port="19530") # Milvus에 새로 연결

    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN) # Milvus 클라이언트 생성
    if client.has_collection(MILVUS_COLLECTION):    # 만약 기존에 같은 이름의 컬렉션이 있다면
        print(f"  ↪ 기존 컬렉션 '{MILVUS_COLLECTION}' 삭제")
        client.drop_collection(MILVUS_COLLECTION)   # 기존 컬렉션 삭제
    print(f"  ↪ Milvus 연결 및 컬렉션 준비 완료.")
    return state                                    # 상태를 그대로 반환
    
def list_files_node(state: IngestState) -> IngestState: # 지정된 폴더에서 처리할 파일 목록을 찾는 노드
    print("🧩 노드: list_files")
    docs_path = Path(state["docsPath"])             # 문서 폴더 경로 가져오기
    allow_ext = {".txt", ".md", ".pdf"}             # 허용할 파일 확장자
    files = [str(p) for p in sorted(docs_path.rglob("*")) if p.is_file() and p.suffix.lower() in allow_ext] # 모든 하위 폴더를 검색하여 파일 목록 생성
    print(f"  ↪ 대상 파일 {len(files)}개")
    if not files:                                   # 파일이 없으면 경고 메시지 출력
        print("  ⚠️ 'ingest_docs' 폴더에 .txt/.md/.pdf 파일을 넣어주세요.")
    return {**state, "files": files}                # 찾은 파일 목록을 상태에 추가하여 반환


def load_and_ingest_node(state: IngestState) -> IngestState: # 문서를 로드, 분할하고 Milvus에 저장하는 메인 노드
    print("🧩 노드: load_and_ingest_node (문서 로드 & Milvus에 인제스트)")
    all_docs: List[Document] = []                   # 모든 문서 청크를 저장할 리스트
    text_splitter = RecursiveCharacterTextSplitter( # 텍스트 분할기 초기화
        chunk_size=CHUNK_SIZE,                      # 청크 크기 설정
        chunk_overlap=CHUNK_OVERLAP,                # 중첩 크기 설정
        length_function=len,                        # 텍스트 길이를 계산할 함수 지정
    )

    # --- 1단계: 문서 로딩 & 청크 분할 ---
    files_to_process = state["files"]               # 처리할 파일 목록 가져오기
    total_files = len(files_to_process)             # 전체 파일 수
    if total_files == 0:                            # 처리할 파일이 없으면
        print("  ⚠️ 처리할 파일이 없습니다.")
        return {**state, "inserted": 0, "rawDocs": [], "vectorstore": None} # 빈 결과 반환

    print(f"\n--- [ 1단계: 문서 로딩 & 청크 분할 ] ---")
    load_start_ts = time.time()                     # 시작 시간 기록
    completion_logs = []                            # 완료 로그를 저장할 리스트
    
    for i, fp in enumerate(files_to_process):       # 각 파일을 순회
        path = Path(fp)                             # 파일 경로를 Path 객체로 변환
        _render_progress("🔄 로드 & 청크", i, total_files, load_start_ts, task_name=path.name) # 진행률 표시
        try:                                        # 오류 처리 시작
            if path.suffix.lower() in [".txt", ".md"]: # 텍스트 또는 마크다운 파일인 경우
                docs = TextLoader(str(path), autodetect_encoding=True).load() # TextLoader로 로드
            elif path.suffix.lower() == ".pdf":     # PDF 파일인 경우
                docs = PyPDFLoader(str(path)).load() # PyPDFLoader로 로드
            else:                                   # 지원하지 않는 파일 형식인 경우
                continue                            # 건너뛰기

            chunks = text_splitter.split_documents(docs) # 문서를 청크로 분할
            all_docs.extend(chunks)                 # 전체 청크 리스트에 추가
            completion_logs.append(f"  - {path.name} 로드 완료 ({len(chunks)}개 청크)") # 완료 로그 추가
        except Exception as e:                      # 오류 발생 시
            completion_logs.append(f"  - {path.name} ❌ 로드 오류") # 오류 로그 추가
            sys.stdout.write("\n")                  # 줄바꿈
            print(f"  └─ ❌ 오류 발생: {path.name} | {e}") # 상세 오류 메시지 출력
            
    _render_progress("✅ 로드 & 청크", total_files, total_files, load_start_ts, task_name="완료") # 최종 완료 상태 표시
    
    print("\n--- 개별 파일 로드 결과 ---")
    for log_entry in completion_logs:               # 저장된 로그를 모두 출력
        print(log_entry)
    print("--------------------------\n")

    if not all_docs:                                # 처리된 문서가 하나도 없으면
        print("  ⚠️ 처리할 문서가 없습니다.")
        return {**state, "inserted": 0, "rawDocs": [], "vectorstore": None} # 빈 결과 반환

    # --- 2단계: 임베딩 & DB 삽입 ---
    print(f"--- [ 2단계: 임베딩 & DB 삽입 ] ---")
    
    docs_by_source = defaultdict(list)              # 파일 이름별로 문서를 그룹화할 딕셔너리
    for doc in all_docs:                            # 모든 청크를 순회
        source_name = os.path.basename(doc.metadata.get("source", "unknown")) # 청크의 원본 파일 이름 가져오기
        docs_by_source[source_name].append(doc)     # 파일 이름에 해당하는 리스트에 청크 추가
    
    progress_embedder = ProgressEmbeddings(         # 임베딩 진행률 추적기 객체 생성
        base=embedding_model,                       # 기본 임베딩 모델 전달
        batch_size=EMBED_BATCH_SIZE,                # 배치 크기 설정
        desc="임베딩"                                # 작업 설명 설정
    )

    vectorstore = None                              # 벡터스토어 객체를 저장할 변수
    inserted_count = 0                              # 삽입된 총 청크 수
    
    total_source_files = len(docs_by_source)        # 처리할 총 파일(그룹) 수
    processed_source_files = 0                      # 처리된 파일(그룹) 수
    for source_name, doc_list in docs_by_source.items(): # 파일 그룹별로 순회
        processed_source_files += 1                 # 처리된 파일 수 증가
        print(f"\n[{processed_source_files}/{total_source_files}] '{source_name}' 파일 처리 시작...")
        try:                                        # 오류 처리
            progress_embedder.update_task(          # 현재 처리할 파일 정보로 진행률 추적기 업데이트
                task_name=source_name,              # 파일 이름 전달
                total_texts=len(doc_list)           # 해당 파일의 청크 수 전달
            )

            if vectorstore is None:                 # 첫 번째 파일인 경우
                vectorstore = Milvus.from_documents( # Milvus 벡터스토어를 새로 생성
                    documents=doc_list,             # 저장할 문서(청크) 목록
                    embedding=progress_embedder,    # 임베딩을 수행할 객체 (진행률 추적기)
                    collection_name=state["collectionName"], # 컬렉션 이름 지정
                    connection_args={"host": "localhost", "port": "19530"} # Milvus 연결 정보
                )
            else:                                   # 두 번째 파일부터는
                # 이미 생성된 벡터스토어는 내부적으로 임베딩 객체를 알고 있음
                vectorstore.add_documents(doc_list) # 기존 벡터스토어에 문서 추가
            
            inserted_count += len(doc_list)         # 삽입된 청크 수 누적

        except Exception as e:                      # 오류 발생 시
            print(f"❌ '{source_name}' 파일 임베딩/삽입 중 오류 발생: {e}")

    print("\n✅ 모든 파일의 임베딩 및 벡터스토어 삽입 완료.")

    return {**state, "inserted": inserted_count, "rawDocs": all_docs, "vectorstore": vectorstore} # 최종 결과 상태 반환

# ──────────────────────────────────────────────────────────────────────────────
# 그래프 빌드
# ──────────────────────────────────────────────────────────────────────────────
def build_graph():                                  # LangGraph 워크플로우를 정의하는 함수
    g = StateGraph(IngestState)                     # IngestState를 상태로 사용하는 그래프 생성
    g.add_node("ensure_milvus", ensure_milvus_node) # 'ensure_milvus' 노드 추가
    g.add_node("list_files", list_files_node)       # 'list_files' 노드 추가
    g.add_node("load_and_ingest", load_and_ingest_node) # 'load_and_ingest' 노드 추가

    g.set_entry_point("ensure_milvus")              # 'ensure_milvus'를 시작 노드로 설정
    g.add_edge("ensure_milvus", "list_files")       # 'ensure_milvus' 다음에 'list_files' 실행
    g.add_edge("list_files", "load_and_ingest")     # 'list_files' 다음에 'load_and_ingest' 실행
    g.add_edge("load_and_ingest", END)              # 'load_and_ingest'가 끝나면 워크플로우 종료

    return g.compile()                              # 정의된 그래프를 컴파일하여 실행 가능한 객체로 반환

# ──────────────────────────────────────────────────────────────────────────────
# 메인 실행 함수
# ──────────────────────────────────────────────────────────────────────────────
def main():                                         # 프로그램의 메인 로직을 담고 있는 함수
    print("🚀 LangGraph 기반 Milvus Ingest 파이프라인 시작")
    agent_app = build_graph()                       # 그래프를 빌드하여 실행 앱 생성

    try:                                            # 그래프 시각화 오류 처리
        graph_image_path = "milvus_agent_workflow_rag.png" # 저장할 이미지 파일 이름
        png_bytes = agent_app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API) # 그래프 구조를 PNG 이미지로 렌더링
        with open(graph_image_path, "wb") as f:     # 이미지 파일을 바이너리 쓰기 모드로 열기
            f.write(png_bytes)                      # 파일에 이미지 데이터 쓰기
        print(f"\n✅ LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:                          # 오류 발생 시
        print(f"그래프 시각화 중 오류 발생: {e}")

    initial_state: IngestState = {                  # 워크플로우를 시작할 때의 초기 상태 정의
        "docsPath": str(DOCS_DIR),                  # 문서 폴더 경로
        "files": [],                                # 파일 목록 (초기에는 비어있음)
        "rawDocs": [],                              # 로드된 문서 (초기에는 비어있음)
        "vectorstore": None,                        # 벡터스토어 객체 (초기에는 없음)
        "inserted": 0,                              # 삽입된 청크 수 (초기값 0)
        "collectionName": MILVUS_COLLECTION,        # 사용할 컬렉션 이름
    }

    final_state = agent_app.invoke(initial_state)   # 초기 상태를 입력하여 그래프 워크플로우 실행

    print("\n📦 결과 요약")
    print(f"  - 처리된 파일 수: {len(final_state['files'])}") # 최종 상태에서 처리된 파일 수 출력
    print(f"  - Milvus 컬렉션: {final_state['collectionName']}") # 사용된 컬렉션 이름 출력
    print(f"  - 삽입된 청크 수: {final_state['inserted']}") # 최종적으로 삽입된 청크 수 출력

if __name__ == "__main__":                          # 이 스크립트 파일이 직접 실행될 때
    main()                                          # main 함수를 호출