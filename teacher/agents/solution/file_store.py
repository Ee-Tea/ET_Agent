import json
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility
import uuid


def load_questions_from_json(inputs):

    # 문제 경로에서 JSON 파일 로드
    with open(inputs["question_path"], "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    docs = []
    for q in questions:
        question_text = q.get("question", "").strip()
        options = q.get("options", [])
        answer = q.get("answer", None)
        explanation = q.get("explanation", "")
        subject = q.get("subject", "정보처리기사")  # 기본 과목 설정



        # 문장은 본문만 저장, 옵션은 metadata에 따로
        doc = Document(
            page_content=question_text,
            metadata={
                "options": json.dumps(options),
                "answer": str(answer) if answer is not None else "",
                "explanation": explanation.strip(),
                "subject": subject,
            }
        )
        docs.append(doc)

    # 🧠 임베딩 모델 및 벡터스토어 설정
    collection_name = "problems"
    embedding_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"}
    )

    if "default" in connections.list_connections():
        connections.disconnect(alias="default")

    # Milvus 연결 설정
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )

    # if not utility.has_collection(collection_name):
    print("🆕 문제 벡터스토어 생성 중...")
    inputs["vectorstore"] = Milvus.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name=collection_name,
        connection_args={"host": "localhost", "port": "19530"}
    )
    # else:
    #     print("✅ 문제 벡터스토어 로드 중...")
    #     inputs["vectorstore"] = Milvus(
    #         embedding_function=embedding_model,
    #         collection_name=collection_name,
    #         connection_args={"host": "localhost", "port": "19530"}
    #     )

if __name__ == "__main__":

    inputs = {
            "question_path": "./x_files/x_sample_pre.json",  # JSON 기출문제 모음
            "docs": []
        }
    load_questions_from_json(inputs)
    print("기출문제 로드 및 벡터스토어 설정 완료")