import os
import glob
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from collections import Counter


class RAGEngine:
    """
    RAG(Retrieval-Augmented Generation) 엔진
    PDF 파일 로딩, 임베딩, 벡터 스토어 관리, 문서 검색을 담당
    """
    
    def __init__(self, data_folder: str = os.path.join(os.path.dirname(__file__),"data")):
        """
        RAG 엔진 초기화
        
        Args:
            data_folder: PDF 파일이 저장된 폴더 경로
        """
        self.data_folder = data_folder
        os.makedirs(self.data_folder, exist_ok=True)
        
        # 임베딩 모델 초기화
        self.embeddings_model = None
        self.vectorstore = None
        self.retriever = None
        
        # 벡터 스토어에 포함된 파일 목록 추적
        self.files_in_vectorstore = []
        
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """임베딩 모델 초기화"""
        try:
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            raise ValueError(f"임베딩 모델 초기화 중 오류 발생: {e}")
    
    def get_pdf_files(self) -> List[str]:
        """
        data 폴더에서 PDF 파일 목록 가져오기
        
        Returns:
            PDF 파일 경로 리스트
        """
        return glob.glob(os.path.join(self.data_folder, "*.pdf"))
    
    def build_vectorstore_from_all_pdfs(self) -> bool:
        """
        모든 PDF를 로드하여 벡터 스토어를 생성/업데이트 (증분 방식)
        
        Returns:
            성공 여부
        """
        pdf_files = self.get_pdf_files()
        if not pdf_files:
            return False

        # 기존 벡터스토어가 있고 파일 목록이 같으면 재사용
        if self.vectorstore and set(self.files_in_vectorstore) == set(pdf_files):
            return True

        # 새로운 파일만 찾기 (첫 번째 실행 시에는 모든 파일)
        new_files = []
        if self.vectorstore:
            new_files = [f for f in pdf_files if f not in self.files_in_vectorstore]
            print(f"📁 새로운 파일 {len(new_files)}개 발견")
        else:
            # 첫 번째 실행 시에는 모든 파일을 처리
            new_files = pdf_files
            print(f"📁 첫 번째 실행: {len(new_files)}개 PDF 파일 처리")
        
        if not new_files and self.vectorstore:
            return True  # 변경사항 없음

        # 새로운 파일들만 처리
        new_documents = []
        for pdf_path in new_files:
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(pdf_path)
                new_documents.extend(documents)
            except Exception as e:
                print(f"PDF 로드 실패: {pdf_path}, 오류: {e}")
                continue

        if new_documents:
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            new_splits = text_splitter.split_documents(new_documents)

            if self.vectorstore:
                # 기존 벡터스토어에 추가
                self.vectorstore.add_documents(new_splits)
                print(f"✅ 기존 벡터스토어에 {len(new_splits)}개 청크 추가")
            else:
                # 새로 생성
                self.vectorstore = FAISS.from_documents(new_splits, self.embeddings_model)
                print(f"✅ 새 벡터스토어 생성: {len(new_splits)}개 청크")

            # retriever 업데이트
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
            
            # 파일 목록 업데이트
            self.files_in_vectorstore = pdf_files
            
            return True
        
        return False
    
    def retrieve_documents(self, query: str, subject_area: str = "", weakness_concepts: List[str] = None) -> Dict[str, Any]:
        """
        쿼리에 관련된 문서를 검색
        
        Args:
            query: 검색 쿼리
            subject_area: 과목 영역
            weakness_concepts: 취약점 개념 리스트
            
        Returns:
            검색 결과와 사용된 소스 정보
        """
        try:
            if not self.retriever:
                return {"error": "벡터 스토어가 초기화되지 않았습니다."}
            
            # 취약점 개념이 있는 경우 쿼리 강화
            if weakness_concepts:
                enhanced_query = f"{subject_area} {' '.join(weakness_concepts)} {query}"
            else:
                enhanced_query = f"{subject_area} {query}"
            
            # 문서 검색
            documents = self.retriever.invoke(enhanced_query)
            
            # 사용된 소스 파일 분석
            source_files = [doc.metadata.get('source_file', 'Unknown') for doc in documents]
            used_sources = list(Counter(source_files).keys())
            
            return {
                "documents": documents,
                "used_sources": used_sources,
                "query": enhanced_query
            }
            
        except Exception as e:
            return {"error": f"문서 검색 오류: {e}"}
    
    def prepare_context(self, documents: List[Document], weakness_concepts: List[str] = None) -> str:
        """
        검색된 문서에서 컨텍스트 준비
        
        Args:
            documents: 검색된 문서 리스트
            weakness_concepts: 취약점 개념 리스트
            
        Returns:
            준비된 컨텍스트 문자열
        """
        if not documents:
            return ""
        
        # 취약점 개념과 관련된 내용을 우선적으로 선별
        key_sents = []
        weakness_related_sents = []
        
        for doc in documents:
            lines = doc.page_content.split("\n")
            for line in lines:
                line = line.strip()
                if len(line) > 50:  # 최소 길이 확보
                    # 취약점 개념과 관련된 내용 우선 선별
                    is_weakness_related = False
                    if weakness_concepts:
                        is_weakness_related = any(
                            concept.lower() in line.lower() 
                            for concept in weakness_concepts
                        )
                    
                    # 중요한 키워드가 포함된 문장
                    is_important = any(k in line for k in [
                        "정의", "특징", "종류", "예시", "원리", 
                        "구성", "절차", "장점", "단점", "방법", "기능"
                    ])
                    
                    if is_weakness_related:
                        weakness_related_sents.append(line)
                    elif is_important or len(line) > 100:
                        key_sents.append(line)
        
        # 취약점 관련 내용을 앞쪽에, 일반 내용을 뒤쪽에 배치
        all_sents = weakness_related_sents + key_sents
        context = "\n".join(all_sents[:20])  # 최대 20개 문장으로 제한
        
        return context
    
    def get_vectorstore_info(self) -> Dict[str, Any]:
        """
        벡터 스토어 정보 반환
        
        Returns:
            벡터 스토어 상태 정보
        """
        return {
            "is_initialized": self.vectorstore is not None,
            "total_files": len(self.files_in_vectorstore),
            "files": self.files_in_vectorstore.copy(),
            "embeddings_model": "jhgan/ko-sroberta-multitask" if self.embeddings_model else None
        }
    
    def clear_vectorstore(self):
        """벡터 스토어 초기화"""
        self.vectorstore = None
        self.retriever = None
        self.files_in_vectorstore = []
    
    def update_data_folder(self, new_data_folder: str):
        """
        데이터 폴더 경로 변경
        
        Args:
            new_data_folder: 새로운 데이터 폴더 경로
        """
        self.data_folder = new_data_folder
        os.makedirs(self.data_folder, exist_ok=True)
        self.clear_vectorstore()  # 기존 벡터 스토어 초기화
