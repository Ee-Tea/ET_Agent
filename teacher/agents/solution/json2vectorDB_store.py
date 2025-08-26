import json
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections

def load_docs_from_json_file(path: Path) -> List[Document]:
    print(f"🔍 JSON 파일 로드 시도: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", data if isinstance(data, list) else [])
    print(f"📄 문항 개수: {len(questions)}")

    docs = []
    for q in questions:
        question_text = (q.get("question") or "").strip()
        if not question_text:
            continue
        options = q.get("options", [])
        answer = q.get("answer")
        explanation = (q.get("explanation") or "").strip()
        subject = q.get("subject", "정보처리기사")

        docs.append(
            Document(
                page_content=question_text,
                metadata={
                    "options": json.dumps(options, ensure_ascii=False),
                    "answer": "" if answer is None else str(answer),
                    "explanation": explanation,
                    "subject": subject,
                },
            )
        )
    print(f"✅ 변환된 Document 개수: {len(docs)}")
    return docs

def main():
    print("🚩 벡터스토어 저장 프로세스 시작")
    if "default" in connections.list_connections():
        print("🔌 기존 Milvus 연결 해제")
        connections.disconnect("default")
    print("🔗 Milvus 연결 시도")
    connections.connect(alias="default", host="127.0.0.1", port="19530")

    embedding = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
    )

    collection_name = "problems"

    base_dir = Path(__file__).parent.parent.parent
    print(f"📁 base_dir: {base_dir}")
    json_folder = base_dir  / "exam" / "parsed_exam_json"
    print(f"📁 JSON 폴더 경로: {json_folder.resolve()}")

    json_files = sorted(Path(json_folder).glob("*.json"))
    print(f"🔎 발견된 JSON 파일 개수: {len(json_files)}")
    if not json_files:
        print("⚠️ JSON 파일이 없습니다. 경로를 확인하세요.")
        return

    for json_file in json_files:
        print(f"➡️ 처리 중: {json_file.name}")
        docs = load_docs_from_json_file(json_file)
        if not docs:
            print(f"⚠️ 문항 없음: {json_file.name} - 구조/내용 확인 필요")
            continue

        try:
            print(f"ℹ️ 기존 콜렉션 열기 시도... ({json_file.name})")
            vs = Milvus(
                embedding_function=embedding,
                collection_name=collection_name,
                connection_args={"host": "127.0.0.1", "port": "19530"},
            )
            print(f"➕ 기존 콜렉션에 문서 추가 중... ({json_file.name})")
            vs.add_documents(docs)
        except Exception as e:
            print(f"🆕 콜렉션이 없거나 열기에 실패했습니다. 새로 생성합니다... ({json_file.name})")
            print(f"에러: {e}")
            vs = Milvus.from_documents(
                documents=docs,
                embedding=embedding,
                collection_name=collection_name,
                connection_args={"host": "127.0.0.1", "port": "19530"},
            )

        print(f"✅ {json_file.name} 벡터스토어 저장 완료")

    print("✅ 모든 파일 벡터스토어 저장 완료")

if __name__ == "__main__":
    main()  