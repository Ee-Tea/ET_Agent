from dotenv import load_dotenv
import os, pandas as pd, numpy as np, re, json, hashlib, warnings
from typing import TypedDict, Optional, List, Dict, Tuple
from numpy.linalg import norm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
import pdfplumber
import xml.etree.ElementTree as ET
from collections import defaultdict

# ========== 2. 상태 및 글로벌 ==========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EVAL_MODE = False   # 평가모드만 내부 로그 출력

class RAGState(TypedDict, total=False):
    question: str
    golden_answer: Optional[str]
    dfs: Optional[List[pd.DataFrame]]
    docs: Optional[List]
    vectorstore: Optional['FAISS']
    context: Optional[str]
    answer: Optional[str]
    folder_path: Optional[str]
    df: Optional[pd.DataFrame]
    context_dataframe: Optional[pd.DataFrame]
    allowed_crops: Optional[List[str]]
    allowed_crops_str: Optional[str]
    retrieved_docs: Optional[List[Document]]
    context_files: Optional[List[str]]

def log(*args, **kwargs):
    if EVAL_MODE:
        print(*args, **kwargs)

# ========== 3. 유틸 함수 ==========
def cosine_similarity(vec1, vec2) -> float:
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    n1, n2 = norm(vec1), norm(vec2)
    return float(np.dot(vec1, vec2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0

def _fix_number_ranges(text: str) -> str:
    text = re.sub(r'(?<![~\-–—])(\d{2,})\s+(\d{2,})(?![\d~\-–—])', r'\1~\2', text)
    text = re.sub(r'(?<![\d~\-–—])(\d{2,})(\d{2,})(?![\d~\-–—])', r'\1~\2', text)
    return text

def clean_korean_answer(answer: str) -> str:
    allowed = r'[^가-힣A-Za-z0-9\s\.,\?!:%/\(\)\-\–\—~\u223C\u301C\uFF5E°℃℉·µμ]'
    text = re.sub(allowed, '', str(answer))
    text = re.sub(r'([.,?!])\1+', r'\1', text)
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = _fix_number_ranges(text)
    return text if len(text) >= 5 else "주어진 정보로는 답변할 수 없습니다."

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip().lower()
    return s

def sha256_hexdigest(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_cell_map(df: pd.DataFrame, func) -> pd.DataFrame:
    if hasattr(pd.DataFrame, "map"):
        return df.map(func)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return df.applymap(func)

# ========== 4. 파일 불러오기 ==========
def file_load_node(state: RAGState) -> RAGState:
    log("[Step1] file_load_node 함수 시작!")
    folder_path = r"C:\Rookies_project\최종 프로젝트\Crop Recommedations DB"
    support_exts = [".csv", ".xlsx", ".pdf", ".txt", ".xml"]
    all_dfs, success_files, failed_files = [], [], []
    try:
        file_list = [f for f in os.listdir(folder_path) if os.path.splitext(f)[-1].lower() in support_exts]
    except Exception as e:
        log(f"폴더 접근 실패: {e}")
        return state

    log(f"[Step1] 총 {len(file_list)}개의 파일을 불러옵니다.")

    def clean_text(x):
        if pd.isna(x):
            return ""
        return str(x).replace("<br>", " ").replace("\n", " ").replace("nan", "").strip()

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        ext = os.path.splitext(file_name)[-1].lower()
        log(f"[Step1]    - {file_name} 불러오는 중...")
        try:
            if ext == ".csv":
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='cp949')
                df = safe_cell_map(df, clean_text)
            elif ext == ".xlsx":
                df = pd.read_excel(file_path)
                df = safe_cell_map(df, clean_text)
            elif ext == ".txt":
                with open(file_path, encoding='utf-8') as f:
                    text = f.read()
                df = pd.DataFrame({"text": [clean_text(text)]})
            elif ext == ".pdf":
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                df = pd.DataFrame({"text": [clean_text(text)]})
            elif ext == ".xml":
                tree = ET.parse(file_path)
                root = tree.getroot()
                texts = [elem.text.strip() for elem in root.iter() if elem.text]
                text = " ".join(texts)
                df = pd.DataFrame({"text": [clean_text(text)]})
            else:
                log(f"[Step1]      ⚠️ 지원하지 않는 파일 형식: {file_name}")
                failed_files.append(file_name)
                continue

            df["source_file"] = file_name
            all_dfs.append(df)
            success_files.append(file_name)
            log(f"[Step1]        ✔️ {file_name} 로딩 완료 (행 개수: {len(df)})")
        except Exception as e:
            log(f"[Step1]        ❌ {file_name} 로딩 실패: {e}")
            failed_files.append(file_name)

    log(f"[Step1] 성공 파일 ({len(success_files)}) :", *["    ✔️ " + fn for fn in success_files], sep="\n")
    log(f"[Step1] 실패 파일 ({len(failed_files)}) :", *["    ❌ " + fn for fn in failed_files], sep="\n")
    log(f"[Step1]    ✔️ 모든 파일 로딩 완료 (총 DataFrame 개수: {len(all_dfs)})")
    return {**state, "dfs": all_dfs}

# ========== 5. 상수/설정 ==========
VECTORSTORE_PATH = "my_faiss_db_all_columns"
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

# ========== 6. 텍스트 분할 및 Document ==========
def split_node(state: RAGState) -> RAGState:
    log("[2/5] 텍스트 분할(split) 중입니다...")
    dfs = state.get("dfs")
    if not dfs:
        raise ValueError("불러온 데이터프레임이 없습니다.")
    df = pd.concat(dfs, ignore_index=True)
    if "source_file" not in df.columns:
        raise ValueError("DataFrame에 source_file 컬럼이 없습니다!")

    crop_name_col = "작물명"
    context_col = "작물정보"

    # 1) 원본 Document 생성(행 단위)
    raw_docs = []
    for _, row in df.iterrows():
        crop_name = str(row[crop_name_col]) if crop_name_col in df.columns else ""
        if context_col in df.columns:
            crop_info = str(row[context_col])
        elif "text" in df.columns:
            crop_info = str(row["text"])
        else:
            crop_info = ""

        context_text = f"{crop_name}. {crop_info}" if crop_name else crop_info
        raw_docs.append(
            Document(
                page_content=context_text,
                metadata={
                    "source_file": row["source_file"],
                    "crop_name": crop_name,
                },
            )
        )

    # 2) 청크 전략 두 가지 (짧은/긴) 병행
    splitter_short = RecursiveCharacterTextSplitter(
        chunk_size=300,  # 기존 256보다 조금 키움
        chunk_overlap=50,
        separators=["\n", "·", "•", "▶", "②", "①", "—", "-", "。", ".", "!", "?", ":", ";"]
    )
    splitter_long = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 긴 문맥도 유지
        chunk_overlap=120,
        separators=["\n", "·", "•", "▶", "②", "①", "—", "-", "。", ".", "!", "?", ":", ";"]
    )

    def chunk_and_tag(docs, splitter, variant_label):
        out = []
        # splitter가 documents 리스트 전체를 분할
        split_docs = splitter.split_documents(docs)
        # 동일 원문 기준으로도 인덱스가 섞이지 않도록 파일/작물명 조합으로 로컬 인덱싱
        per_key_counter = {}
        for d in split_docs:
            key = (d.metadata.get("source_file",""), d.metadata.get("crop_name",""), variant_label)
            idx = per_key_counter.get(key, 0)
            per_key_counter[key] = idx + 1

            # content_hash는 청크 내용 + 변형정보로 계산 (중복 방지)
            base = f"{normalize_text(d.page_content)}|{d.metadata.get('source_file','')}|{d.metadata.get('crop_name','')}|{variant_label}|{idx}"
            d.metadata.update({
                "chunk_variant": variant_label,   # 's' or 'l'
                "chunk_index": idx,
                "content_hash": sha256_hexdigest(base),
            })
            out.append(d)
        return out

    short_docs = chunk_and_tag(raw_docs, splitter_short, "s")
    long_docs  = chunk_and_tag(raw_docs, splitter_long,  "l")
    split_docs = short_docs + long_docs

    log(f"    ✔️ 분할 청크 수(짧은): {len(short_docs)} / (긴): {len(long_docs)} / 합계: {len(split_docs)}")
    if context_col in df.columns:
        total_length = int(sum(len(str(val)) for val in df[context_col]))
        avg_len = int(total_length / max(1, len(split_docs)))
        if EVAL_MODE:
            print(f"csv 전체 텍스트 길이: {total_length}자, 평균 청크 길이: {avg_len}자")

    return {**state, "df": df, "docs": split_docs}


def embedding_and_vectorstore_node(state: RAGState) -> RAGState:
    log("[Step3] 임베딩 및 벡터스토어 작업 중...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    docs = state["docs"]

    # content_hash 보장 함수 (혹시 누락된 경우 대비)
    def ensure_hash(doc: Document) -> str:
        h = doc.metadata.get("content_hash")
        if not h:
            base = f"{normalize_text(doc.page_content)}|{doc.metadata.get('source_file','')}|{doc.metadata.get('crop_name','')}|{doc.metadata.get('chunk_variant','')}|{doc.metadata.get('chunk_index','')}"
            h = sha256_hexdigest(base)
            doc.metadata["content_hash"] = h
        return h

    if os.path.exists(VECTORSTORE_PATH):
        log("[Step3] 기존 벡터스토어가 존재! 신규 문서만 추가")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

        existing_hashes = set()
        for d in vectorstore.docstore._dict.values():
            h = d.metadata.get("content_hash")
            if not h:
                base = f"{normalize_text(d.page_content)}|{d.metadata.get('source_file','')}|{d.metadata.get('crop_name','')}|{d.metadata.get('chunk_variant','')}|{d.metadata.get('chunk_index','')}"
                h = sha256_hexdigest(base)
            existing_hashes.add(h)

        unique_docs = [doc for doc in docs if ensure_hash(doc) not in existing_hashes]
        log(f"[Step3] 신규 청크 수: {len(unique_docs)}")
        if unique_docs:
            vectorstore.add_documents(unique_docs)
            vectorstore.save_local(VECTORSTORE_PATH)
            log("[Step3] 신규 임베딩 추가 및 저장 완료")
        else:
            log("[Step3] 추가할 신규 청크 없음. 기존 벡터스토어 재사용")
    else:
        log("[Step3] 벡터스토어 새로 생성/저장")
        # 모든 문서는 content_hash를 보장
        _ = [ensure_hash(doc) for doc in docs]
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        log("[Step3] 벡터스토어 생성 및 저장 완료")

    return {**state, "vectorstore": vectorstore}
# ========== 8. 컨텍스트 검색 ==========
def retriever_node(state: RAGState) -> RAGState:
    log("[Step4] 컨텍스트 검색(retriever_node) 실행")
    vectorstore = state["vectorstore"]
    user_question = state["question"]

    docs_with_scores: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(user_question, k=20)
    docs_with_scores.sort(key=lambda x: x[1])

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    query_vec = embeddings.embed_query(user_question)

    texts_for_embed = [doc.page_content for doc, _ in docs_with_scores]
    if texts_for_embed:
        doc_vecs = embeddings.embed_documents(texts_for_embed)
    else:
        doc_vecs = []

    groups = defaultdict(list)
    cosine_map = {}
    for (doc, _), dvec in zip(docs_with_scores, doc_vecs):
        cos = cosine_similarity(query_vec, dvec)
        cosine_map[id(doc)] = cos
        crop = (doc.metadata.get("crop_name") or "").strip()
        groups[crop].append((doc, cos))

    ranked = []
    for crop, items in groups.items():
        if not crop:
            continue
        max_sim = max(s for _, s in items)
        cnt = len(items)
        ranked.append((crop, -max_sim, -cnt))
    ranked.sort(key=lambda x: (x[1], x[2]))

    selected_crops = [r[0] for r in ranked[:2]]
    if selected_crops:
        context_docs: List[Document] = []
        for c in selected_crops:
            context_docs.extend([d for d, _ in sorted(groups[c], key=lambda x: -x[1])[:3]])
    else:
        context_docs = [doc for doc, _ in docs_with_scores[:5]]

    allowed_unique = sorted(set([d.metadata.get("crop_name", "") for d in context_docs if d.metadata.get("crop_name")]))
    allowed_str = ", ".join(allowed_unique[:20]) if allowed_unique else ""
    context_text = "\n".join(doc.page_content for doc in context_docs) if context_docs else ""

    df_context = pd.DataFrame([
        {
            "file": doc.metadata.get("source_file", ""),
            "crop_name": doc.metadata.get("crop_name", ""),
            # "chunk_text": doc.page_content[:110] + ("..." if len(doc.page_content) > 110 else ""),
            "chunk_text": doc.page_content,
            "cosine_similarity": round(cosine_map.get(id(doc), 0.0), 4),
            "all_text": doc.page_content,
        }
        for doc in context_docs
    ])

    return {
        **state,
        "context": context_text,
        "retrieved_docs": context_docs,
        "context_files": [doc.metadata.get("source_file", "") for doc in context_docs],
        "context_dataframe": df_context,
        "allowed_crops": allowed_unique,
        "allowed_crops_str": allowed_str,
    }

# ========== 9. 답변 생성 ==========
def generate_answer_node(state: RAGState) -> RAGState:
    log("[Step5] 답변 생성(LangChain LLM 호출) 중입니다...")

    cleaned_answer = "주어진 정보로는 답변할 수 없습니다."
    context = state.get("context", "")
    if not context.strip():
        log("[Step5] 답변 생성 완료 (context 없음)")
        return {**state, "answer": cleaned_answer}
    
    allowed_str = state.get("allowed_crops_str", "")

    prompt_tmpl = PromptTemplate.from_template("""
    너는 대한민국 농업 작물 추천 분야의 전문가이자, 친근하게 상담해 주는 작물 추천 가이드야. 항상 한국어로 답변해줘.

    [핵심 원칙]
    1) <context> 안에서만 답해. context에 없는 사실은 추가하지 마.
    2) 대표 작물명만 사용해서 추천해. 품종·계통·상표명 금지.
    3) 숫자·월·기간은 <context> 값을 그대로 써(임의 보정 금지).

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
    """)

    _prompt = prompt_tmpl.format(
        context=context,
        question=state.get("question", ""),
        allowed_crops=allowed_str if allowed_str else "(없음)"
    )

    try:
        if EVAL_MODE:
            # 평가 모드 → 안정성과 정확도 우선
            temp_val = 0.2
            max_tok = 1024
        else:
            # 채팅 모드 → 다양한 응답과 창의성 우선
            temp_val = 0.7
            max_tok = 512

        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=temp_val,
            max_tokens=max_tok
        )

        msg = llm.invoke(_prompt)

        # ✅ 후처리 제거 — LLM 출력 그대로 사용
        final_answer = msg.content if hasattr(msg, "content") else str(msg)

    except Exception as e:
        log(f"[Step5] LLM 호출 오류: {e}")
        final_answer = "주어진 정보로는 답변할 수 없습니다."

    log("[Step5] 답변 생성 완료")
    return {**state, "answer": final_answer}

# ========== 10. 평가 출력/평가 ==========
def print_context_dataframe(df):
    if isinstance(df, pd.DataFrame) and not df.empty:
        print("\n[참고한 context/chunk 목록]")
        
        # ▼▼ 파일 저장 코드 (누적 저장) ▼▼
        log_path = "used_chunks_log.txt"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("[사용한 컨텍스트 청크]\n")
                f.write(df[["crop_name",  "chunk_text"]].to_string(index=True))
                f.write("\n-------------------------\n")
        except Exception as e:
            print(f"[로그 저장 오류] {e}")
        # ▲▲ 파일 저장 코드 ▲▲
        
        print(df[["file", "crop_name", "cosine_similarity", "chunk_text"]].to_string(index=True))
        
    else:
        print("\n[참고한 context/chunk 없음]")

def evaluate_chatbot(app, golden_dataset: List[Dict[str, str]]):
    print("\n--- 챗봇 성능 평가 시작 (유사도 기반) ---")
    evaluation_results = []
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    SIMILARITY_THRESHOLD = 0.75

    for i, data in enumerate(golden_dataset):
        question, golden_answer = data['question'], data['answer']
        print(f"\n[평가 {i+1}] 질문: {question}")
        
        # ▼▼ 파일 저장 코드 (누적 저장) ▼▼
        log_path = "used_chunks_log.txt"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n----- [평가 {i+1}] -----\n")
                f.write(f"[질문] {question}\n")
        except Exception as e:
            print(f"[로그 저장 오류] {e}")
        # ▲▲ 파일 저장 코드 ▲▲
        
        print(f"  - 정답: {golden_answer}")
        try:
            final_state = app.invoke({"question": question, "golden_answer": golden_answer})
            generated_answer = final_state['answer']
            df_context = final_state.get('context_dataframe')
            print(f"  - 생성된 답변: {generated_answer}")
            print_context_dataframe(df_context)

            golden_emb = embeddings.embed_query(golden_answer)
            generated_emb = embeddings.embed_query(generated_answer)
            answer_similarity = cosine_similarity(golden_emb, generated_emb)

            search_similarity_max = float(df_context['cosine_similarity'].max()) if df_context is not None and not df_context.empty else 0.0

            is_correct = bool(answer_similarity >= SIMILARITY_THRESHOLD)
            print(f"  - 검색 유사도(질문↔청크): {search_similarity_max:.4f}")
            print(f"  - 답변 유사도(골든↔답변): {answer_similarity:.4f} (합격 기준: {SIMILARITY_THRESHOLD})")
            print(f"  - 정답 여부: {'✅ 정답' if is_correct else '❌ 오답'}")

            evaluation_results.append({
                'question': question,
                'golden_answer': golden_answer,
                'generated_answer': generated_answer,
                'search_similarity_max': search_similarity_max,
                'answer_similarity': float(answer_similarity),
                'is_correct': is_correct
            })
        except Exception as e:
            print(f"  - 오류 발생: {e}")
            evaluation_results.append({
                'question': question,
                'golden_answer': golden_answer,
                'generated_answer': '오류 발생',
                'search_similarity_max': 0.0,
                'answer_similarity': 0.0,
                'is_correct': False
            })

    correct = sum(1 for res in evaluation_results if res['is_correct'])
    total = len(evaluation_results)
    print("\n--- 챗봇 성능 평가 완료 ---")
    print(f"\n총 질문 수: {total}")
    print(f"정답 수: {correct}")
    print(f"정확도: {(correct/total)*100:.2f}%")
    with open('evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
    print("상세 평가 결과가 'evaluation_report.json' 파일에 저장되었습니다.")

# ========== 11. 전체 RAG 파이프라인 ==========
def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("file_load", file_load_node)
    graph.add_node("split", split_node)
    graph.add_node("embedding_vectorstore", embedding_and_vectorstore_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_edge("file_load", "split")
    graph.add_edge("split", "embedding_vectorstore")
    graph.add_edge("embedding_vectorstore", "retriever")
    graph.add_edge("retriever", "generate_answer")
    graph.add_edge("generate_answer", END)
    graph.set_entry_point("file_load")
    return graph.compile()

# ========== 12. 평가/실행 진입점 ==========
def load_golden_dataset(file_path: str):
    return pd.read_csv(file_path, encoding='utf-8').to_dict('records')

if __name__ == "__main__":
    print("\n---농작물 챗봇 에이전트 시작---")
    app = build_rag_graph()
    mode = input("실행 모드를 선택하세요 (1=챗봇, 2=평가): ").strip()

    if mode == "2":
        EVAL_MODE = True
        golden_path = r"C:\Rookies_project\최종 프로젝트\Project_test\Goldenset_test1.csv"
        print(f"[평가모드] 골든셋 CSV: {golden_path}")
        try:
            df_check = pd.read_csv(golden_path, encoding="utf-8")
            print("\n[골든셋 CSV 컬럼 구조]:", list(df_check.columns))
            print("[골든셋 샘플 데이터]")
            print(df_check.head(3))
        except Exception as e:
            print("[오류] 골든셋 파일을 읽을 수 없습니다:", e)
            exit(1)

        golden_dataset = load_golden_dataset(golden_path)
        # golden_dataset = golden_dataset[:5]
        evaluate_chatbot(app, golden_dataset)

    else:
        EVAL_MODE = False
        warnings.filterwarnings("ignore")
        while True:
            user_question = input("\n질문을 입력하세요: ")
            if user_question.lower() in ["exit", "quit"]:
                print("챗봇을 종료합니다.")
                break
            if not user_question.strip():
                print("질문을 입력해주세요.")
                continue
            try:
                state = {"question": user_question}
                final_state = app.invoke(state)
                print(final_state["answer"])
            except Exception as e:
                print(f"\n오류가 발생했습니다: {e}")
                print("다시 시도해주세요.")
    print("\n---농작물 챗봇 에이전트 종료---")
