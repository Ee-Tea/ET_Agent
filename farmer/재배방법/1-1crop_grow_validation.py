import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict
from numpy.linalg import norm # 코사인 유사도 계산을 위해 numpy.linalg.norm 추가

# Langchain 및 LangGraph 관련 라이브러리
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- 1. 환경 설정 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

print(f"OPENAI_API_KEY 로드 완료: {OPENAI_API_KEY[:5]}...") # 보안을 위해 키 앞부분만 출력

# 데이터 파일 경로 및 벡터 DB 저장 경로
CSV_FILE_PATH = 'data/농림수산식품교육문화정보원_영농가이드 재배정보_20230911.csv'
VECTOR_DB_PATH = 'faiss_crop_guide_db' # 벡터 DB 저장 경로 (함수형 전용으로 경로 변경)
GOLDEN_DATASET_PATH = 'data/golden_dataset.csv'

# --- 2. LLM 및 프롬프트 설정 ---
llm = ChatGroq(model_name="llama3-70b-8192", 
               temperature=0.5, 
               api_key=OPENAI_API_KEY)

RAG_PROMPT_TEMPLATE = """
당신은 대한민국 농업 작물 재배 방법 전문가입니다. 

다음은 작물 재배에 대한 정보를 담고 있습니다. 제공된 다음의 정보를 사용하여 사용자의 질문에 답변해 주세요.
{context}

당신이 지킬 규칙들은 다음과 같습니다:
- 제공된 정보에 없는 정보, 저의 상식, 추측, 거짓 정보, 한자 등은 절대 답변에 넣지 마세요.
- 모든 답변은 한글로만 답해야 합니다. 한글이 아니면 절대 출력하지 말고, 한글로만 답변을 완성해야 합니다.
- 각 재배 단계나 설명은 반드시 "한 문장씩 줄바꿈"해서 써주세요.
- 답변이 불가능하거나 제공된 정보에 관련 정보가 없으면 "주어진 정보로는 답변할 수 없습니다."라고만 답해야 합니다.
- 친근하게 대화하듯, 자연스럽게 설명해주세요.

질문: {question}
답변:
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# --- 3. LangGraph 상태 정의 ---
class GraphState(TypedDict):
    """
    LangGraph 상태를 나타내는 TypedDict.
    LangGraph는 TypedDict 기반 상태를 더 유연하게 처리합니다.
    """
    question: Optional[str]
    documents: Optional[List[Dict[str, Any]]]
    splits: Optional[List[Dict[str, Any]]]
    vectorstore: Optional[FAISS]
    context: Optional[str]
    answer: Optional[str]

# --- 4. 핵심 기능 함수 정의 ---

def load_csv_data(file_path: str) -> List[Dict[str, Any]]:
    """CSV 파일을 로드하여 문서 리스트를 반환합니다."""
    print("---기능: 데이터 로드 시작---")
    loader = CSVLoader(file_path=file_path, encoding='utf-8')
    documents = loader.load()
    print(f"로드된 문서 개수: {len(documents)}개")
    return documents

def split_documents(documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """문서 리스트를 청크로 분할합니다."""
    print("---기능: 문서 분할 시작---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(documents)
    print(f"분할된 청크 개수: {len(splits)}개")
    return splits

def embed_and_store_vector_db(splits: List[Dict[str, Any]], db_path: str) -> FAISS:
    """청크를 임베딩하고 벡터 DB에 저장(또는 로드)합니다."""
    print("---기능: 임베딩 및 벡터 DB 저장 시작---")
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    if not os.path.exists(db_path):
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(db_path)
        print(f"벡터 DB를 '{db_path}' 경로에 새로 생성 및 저장했습니다.")
    else:
        vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print(f"'{db_path}' 경로에서 기존 벡터 DB를 로드했습니다.")
    return vectorstore

def retrieve_relevant_chunks(vectorstore: FAISS, question: str) -> str:
    """벡터 DB에서 사용자 질문에 가장 관련성 높은 청크를 유사도 검색으로 찾습니다."""
    print("---기능: 관련 청크 유사도 검색 시작---")
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"검색된 청크 수: {len(docs)}개")
    return context

def generate_answer_with_llm(context: str, question: str, llm: ChatGroq, prompt_template: ChatPromptTemplate) -> str:
    """검색된 문맥과 질문을 바탕으로 LLM을 사용하여 답변을 생성합니다."""
    print("---기능: 답변 생성 시작---")
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    inputs = {"context": context, "question": question}
    answer = rag_chain.invoke(inputs)
    return answer

# --- 5. LangGraph 노드 함수 정의 ---

def load_data_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 데이터 로드"""
    print("\n---노드: 데이터 로드 (Load Data) 실행---")
    documents = load_csv_data(CSV_FILE_PATH)
    
    question_from_state = state.get('question') 
    
    return {"question": question_from_state, "documents": documents}

def split_documents_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 문서 분할"""
    print("\n---노드: 문서 분할 (Split Documents) 실행---")
    if "documents" not in state or not state["documents"]: 
        raise ValueError("문서 분할을 위해 로드된 문서가 필요합니다.")
    splits = split_documents(state["documents"])
    
    return {**state, "splits": splits} 

def embed_and_store_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 임베딩 및 벡터 DB 저장"""
    print("\n---노드: 임베딩 및 벡터 DB 저장 (Embed & Store) 실행---")
    if "splits" not in state or not state["splits"]:
        raise ValueError("임베딩을 위해 분할된 청크가 필요합니다.")
    vectorstore = embed_and_store_vector_db(state["splits"], VECTOR_DB_PATH)
    
    return {**state, "vectorstore": vectorstore}

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 검색"""
    print("\n---노드: 검색 (Retrieve) 실행---")
    if "vectorstore" not in state or not state["vectorstore"]:
        raise ValueError("검색을 위해 벡터 DB가 필요합니다.")
    question = state["question"] 
    vectorstore = state["vectorstore"]
    context = retrieve_relevant_chunks(vectorstore, question)
    
    return {**state, "context": context}

def generate_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph 노드: 답변 생성"""
    print("\n---노드: 생성 (Generate) 실행---")
    if ("context" not in state or not state["context"] or 
        "question" not in state or not state["question"]):
        raise ValueError("답변 생성을 위해 문맥과 질문이 필요합니다.")
    context = state["context"]
    question = state["question"]
    answer = generate_answer_with_llm(context, question, llm, rag_prompt)
    
    return {**state, "answer": answer}

# --- 6. LangGraph 워크플로우 빌드 및 컴파일 ---
def build_rag_workflow():
    """RAG 워크플로우를 빌드하고 컴파일합니다."""
    workflow = StateGraph(GraphState) 

    workflow.add_node("load_data", load_data_node)
    workflow.add_node("split_documents", split_documents_node)
    workflow.add_node("embed_and_store", embed_and_store_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge("load_data", "split_documents")
    workflow.add_edge("split_documents", "embed_and_store")
    workflow.add_edge("embed_and_store", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    workflow.set_entry_point("load_data")

    return workflow.compile()

# --- 7. 평가 기능 추가 (수정) ---
def load_golden_dataset(file_path: str):
    """골든 데이터셋을 CSV 파일에서 로드하여 리스트로 반환합니다."""
    df = pd.read_csv(file_path, encoding='utf-8')
    return df.to_dict('records')

def cosine_similarity(vec1, vec2):
    """두 벡터 간의 코사인 유사도를 계산합니다."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = norm(vec1)
    norm_vec2 = norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def evaluate_chatbot(app, golden_dataset: List[Dict[str, str]]):
    """골든 데이터셋을 사용하여 챗봇의 성능을 평가합니다."""
    print("\n--- 챗봇 성능 평가 시작 (유사도 기반) ---")
    
    evaluation_results = []
    
    # 임베딩 모델 로드
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    # 유사도 평가 임계값 (threshold) 설정
    SIMILARITY_THRESHOLD = 0.8
    
    for i, data in enumerate(golden_dataset):
        question = data['question']
        golden_answer = data['answer']
        
        print(f"\n[평가 {i+1}] 질문: {question}")
        print(f"  - 정답: {golden_answer}")
        
        try:
            final_state = app.invoke({"question": question})
            generated_answer = final_state['answer']
            
            print(f"  - 생성된 답변: {generated_answer}")
            
            golden_embedding = embeddings.embed_query(golden_answer)
            generated_embedding = embeddings.embed_query(generated_answer)
            
            similarity_score = cosine_similarity(golden_embedding, generated_embedding)
            
            # 파이썬의 기본 bool 타입으로 변환
            is_correct = bool(similarity_score >= SIMILARITY_THRESHOLD)
            
            print(f"  - 유사도 점수: {similarity_score:.4f} (합격 기준: {SIMILARITY_THRESHOLD})")
            print(f"  - 정답 여부: {'✅ 정답' if is_correct else '❌ 오답'}")
            
            evaluation_results.append({
                'question': question,
                'golden_answer': golden_answer,
                'generated_answer': generated_answer,
                'similarity_score': float(similarity_score), # numpy float을 파이썬 float으로 변환
                'is_correct': is_correct # 이미 bool로 변환됨
            })
            
        except Exception as e:
            print(f"  - 오류 발생: {e}")
            evaluation_results.append({
                'question': question,
                'golden_answer': golden_answer,
                'generated_answer': '오류 발생',
                'similarity_score': 0.0,
                'is_correct': False
            })

    print("\n--- 챗봇 성능 평가 완료 ---")
    
    total_questions = len(evaluation_results)
    correct_answers = sum(1 for res in evaluation_results if res['is_correct'])
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    print(f"\n총 질문 수: {total_questions}")
    print(f"정답 수: {correct_answers}")
    print(f"정확도: {accuracy:.2f}%")
    
    with open('evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
    print("상세 평가 결과가 'evaluation_report.json' 파일에 저장되었습니다.")
    
# --- 8. 메인 실행 부분 ---
if __name__ == "__main__":
    app = build_rag_workflow()
    
    try:
        golden_dataset = load_golden_dataset(GOLDEN_DATASET_PATH)
        evaluate_chatbot(app, golden_dataset)
    except FileNotFoundError:
        print(f"오류: {GOLDEN_DATASET_PATH} 파일이 존재하지 않습니다.")
        print("골든 데이터셋 파일을 'data' 폴더에 준비해주세요.")