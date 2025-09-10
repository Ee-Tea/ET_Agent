import os
import sys
import asyncio
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# RAGAS 모듈
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.persona import Persona

# LangChain & Hugging Face
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Windows asyncio 정책
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PDF 로딩 함수
def load_pdfs(directory_path):
    pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    return pdf_files

# 단일 PDF 파일을 처리하는 비동기 함수
async def process_pdf(pdf_path, generator, all_questions, output_filename, pbar):
    try:
        # PDF 로딩 단계에 예외 처리를 추가하여 파일 로드 오류 시 건너뛰도록 합니다.
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        except Exception as e:
            pbar.write(f"❌ PDF 로딩 실패: {os.path.basename(pdf_path)} - {e}. 이 파일을 건너뜁니다.")
            return

        pbar.write(f"\n파일: {os.path.basename(pdf_path)}에서 질문 생성 중...")
        
        # generate_with_langchain_docs 함수를 asyncio.wait_for로 감싸서 시간 제한을 5분(300초)으로 늘립니다.
        testset_task = generator.generate_with_langchain_docs(
            documents,
            testset_size=3
        )
        testset = await asyncio.wait_for(testset_task, timeout=300.0)

        # 생성된 문제를 리스트에 추가
        new_questions = testset.to_pandas().to_dict('records')
        all_questions.extend(new_questions)

        # 성공할 때마다 즉시 파일에 저장
        df = pd.DataFrame(all_questions)
        df.to_csv(output_filename, index=False, encoding="utf-8-sig")
        pbar.write(f"✅ {len(new_questions)}개 문제 생성 완료. 현재까지 총 {len(all_questions)} 문제가 {output_filename}에 저장되었습니다.")

    except asyncio.TimeoutError:
        pbar.write(f"❌ 타임아웃 오류: {os.path.basename(pdf_path)} 파일 처리 중 300초를 초과하여 건너뜁니다.")
    except Exception as e:
        pbar.write(f"❌ 질문 생성 실패: {os.path.basename(pdf_path)} - {e}")
        pbar.write(f"오류가 발생했지만, 마지막 성공 배치까지는 저장되었습니다.")
    
# 비동기 실행용 main
async def main():
    pdf_files = load_pdfs('./data/cropinfo/cropinfo2')

    # RAGAS LLM 및 임베딩 설정 (LangchainWrapper 사용)
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0.7))
    embeddings_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings_model)
    
    # 페르소나 정의 (simple, reasoning, multi-context를 유도)
    personas = [
        Persona(
            name="기초_학습자",
            role_description="문서의 핵심 개념과 사실을 파악하려는 초보자입니다. 문서를 꼼꼼히 읽고 간단한 사실적 한글 질문을 합니다."
        ),
        Persona(
            name="시장_분석가",
            role_description="문서의 내용을 바탕으로 논리적 관계나 추론이 필요한 한글 질문을 생성합니다. 여러 정보를 조합하고 분석하는 데 능숙합니다."
        ),
        Persona(
            name="전략_기획자",
            role_description="문서 전체를 아우르는 포괄적인 시각을 가졌습니다. 여러 단락이나 페이지에 흩어져 있는 정보를 연결하여 복합적인 한글 질문을 만듭니다."
        )
    ]

    # TestsetGenerator를 페르소나 리스트와 함께 초기화
    generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings_wrapper, persona_list=personas)
    
    output_filename = "golden_dataset_ko.csv"

    # 기존 파일에서 이어서 작업
    if os.path.exists(output_filename):
        existing_df = pd.read_csv(output_filename)
        all_questions = existing_df.to_dict('records')
        print(f"기존 파일({output_filename})에서 {len(all_questions)} 문제 로드 완료. 이어서 작업합니다.")
    else:
        all_questions = []

    # tqdm 진행바와 함께 각 PDF를 처리
    with tqdm(total=len(pdf_files), desc="PDF 파일 처리 중") as pbar:
        for pdf_path in pdf_files:
            await process_pdf(pdf_path, generator, all_questions, output_filename, pbar)
            pbar.update(1)

    print("\n---")
    print(f"최종적으로 총 {len(all_questions)}개의 질문이 생성 및 저장되었습니다.")
    print("---")

# asyncio로 전체 실행
if __name__ == "__main__":
    asyncio.run(main())