# teacher/agents/TestGenerator/milvus_store.py
import json
import hashlib
from typing import Dict, Any, List
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility


def _build_embeddings() -> HuggingFaceEmbeddings:
    # 정규화 + 배치 권장
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def _connect_milvus(host: str, port: str, alias: str = "default") -> None:
    if alias in connections.list_connections():
        connections.disconnect(alias=alias)
    connections.connect(alias=alias, host=host, port=port)


def _make_docs(questions: List[Dict[str, Any]]) -> List[Document]:
    docs, seen = [], set()
    for q in questions:
        question_text = (q.get("question") or "").strip()
        options = q.get("options") or []
        answer = q.get("answer")
        explanation = (q.get("explanation") or "").strip()
        subject = q.get("subject") or "정보처리기사"

        # 간단 중복 방지 키 (질문+보기+정답)
        sig = hashlib.sha1(
            (question_text + "|" + json.dumps(options, ensure_ascii=False) + "|" + str(answer)).encode("utf-8")
        ).hexdigest()
        if sig in seen:
            continue
        seen.add(sig)

        docs.append(Document(
            page_content=question_text,
            metadata={
                "doc_id": sig,
                "options": options,  # 문자열화하지 않습니다.
                "answer": str(answer) if answer is not None else "",
                "explanation": explanation,
                "subject": subject,
            }
        ))
    return docs


def load_questions_from_json(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    inputs 예시:
    {
      "question_path": "...json",
      "collection_name": "problems",
      "milvus_host": "localhost",
      "milvus_port": "19530",
      "drop_old": False,
      "k": 10
    }
    반환: inputs 에 vectorstore, retriever 추가하여 그대로 리턴
    """
    # 1) JSON 로드 (키 호환: all_questions / questions)
    with open(inputs["question_path"], "r", encoding="utf-8") as f:
        payload = json.load(f)
    questions = payload.get("all_questions") or payload.get("questions") or []

    docs = _make_docs(questions)

    # 2) 임베딩/접속/파라미터
    embedding_model = _build_embeddings()
    collection_name = inputs.get("collection_name", "problems")
    host = inputs.get("milvus_host", "localhost")
    port = inputs.get("milvus_port", "19530")
    drop_old = bool(inputs.get("drop_old", False))
    topk = int(inputs.get("k", 10))

    _connect_milvus(host, port, alias="default")

    index_params = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    search_params = {"metric_type": "IP"}  # 필요 시 {"params": {"ef": 64}} 등 추가

    # 3) 컬렉션 관리: 드롭 or 생성/추가
    if drop_old and utility.has_collection(collection_name):
        print(f"⚠️  Drop existing collection: {collection_name}")
        utility.drop_collection(collection_name)

    if not utility.has_collection(collection_name):
        print("🆕 컬렉션이 없어 새로 생성합니다… (from_documents)")
        vs = Milvus.from_documents(
            documents=docs,
            embedding=embedding_model,
            collection_name=collection_name,
            connection_args={"host": host, "port": port},
            index_params=index_params,
            search_params=search_params,
        )
    else:
        print("✅ 기존 컬렉션에 문서를 추가합니다…")
        vs = Milvus(
            embedding_function=embedding_model,
            collection_name=collection_name,
            connection_args={"host": host, "port": port},
            index_params=index_params,
            search_params=search_params,
        )
        if docs:
            vs.add_documents(docs)

    # 4) retriever 준비
    retriever = vs.as_retriever(search_kwargs={"k": topk})
    inputs["vectorstore"] = vs
    inputs["retriever"] = retriever
    inputs["loaded_count"] = len(docs)
    return inputs


if __name__ == "__main__":
    inputs = {
        "question_path": "./teacher/agents/TestGenerator/test/설계 중심 2개_2과목_24문제.json",
        "collection_name": "problems",
        "milvus_host": "localhost",
        "milvus_port": "19530",
        "drop_old": False,  # 개발 중 초기화가 필요하면 True
        "k": 10,
    }
    out = load_questions_from_json(inputs)
    print(f"기출문제 로드 및 벡터스토어 설정 완료 (적재 {out['loaded_count']}건, k={inputs['k']})")
