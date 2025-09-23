import os
import glob
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS  # FAISS 제거
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
# 🔁 Milvus 관련 임포트 추가
from langchain_milvus import Milvus
from pymilvus import connections, utility
# from teacher.agents.milvus_utils import connect_milvus_fallback
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
        
        # 🔧 Milvus 설정 추가
        self.milvus_conf = {
            "host": os.getenv("MILVUS_HOST", "standalone"),  # 컨테이너 내부 기본값 standalone
            "port": os.getenv("MILVUS_PORT", "19530"),
            "collection": os.getenv("MILVUS_COLLECTION", "rag_documents"),
            "topk": int(os.getenv("MILVUS_TOPK", "15")),
            "drop_existing": os.getenv("MILVUS_DROP_EXISTING", "false").lower() == "true",
        }
        
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
                    # 🔧 PDF 메타데이터 정리 및 Milvus 스키마에 맞게 조정
                    metadata = doc.metadata.copy()
                    
                    # 필수 필드 보장
                    metadata['source_file'] = os.path.basename(pdf_path)
                    
                    # title 필드가 없거나 None인 경우 기본값 설정
                    if not metadata.get('title') or metadata['title'] is None:
                        metadata['title'] = os.path.basename(pdf_path)
                    
                    # author 필드가 없거나 None인 경우 기본값 설정
                    if not metadata.get('author') or metadata['author'] is None:
                        metadata['author'] = 'Unknown'
                    
                    # 기타 필드들도 안전하게 처리
                    for key in ['producer', 'creator', 'creationdate', 'moddate', 'source']:
                        if key not in metadata or metadata[key] is None:
                            metadata[key] = ''
                    
                    # page 관련 필드 정수형으로 변환
                    if 'page' in metadata and metadata['page'] is not None:
                        try:
                            metadata['page'] = int(metadata['page'])
                        except (ValueError, TypeError):
                            metadata['page'] = 0
                    
                    if 'total_pages' in metadata and metadata['total_pages'] is not None:
                        try:
                            metadata['total_pages'] = int(metadata['total_pages'])
                        except (ValueError, TypeError):
                            metadata['total_pages'] = 1
                    
                    # 수정된 메타데이터로 문서 업데이트
                    doc.metadata = metadata
                    new_documents.append(doc)
                    
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

            # 🔧 Milvus 연결 및 컬렉션 관리
            host_env = self.milvus_conf["host"]
            port = self.milvus_conf["port"]
            collection = self.milvus_conf["collection"]
            
            # 연결 정리 후 재접속
            # host fallback (컨테이너/호스트 양쪽 지원)
            used_host = connect_milvus_fallback(hosts=[host_env], port=port)
            if not used_host:
                print("🚫 Milvus 연결 불가. 새 문서 추가 중단")
                return False
            host_env = used_host

            if self.vectorstore:
                # 기존 Milvus 벡터스토어에 추가
                self.vectorstore.add_documents(new_splits)
                print(f"✅ 기존 Milvus 벡터스토어에 {len(new_splits)}개 청크 추가")
            else:
                # 🔧 기존 컬렉션이 있고 drop_existing이 True인 경우 삭제
                if self.milvus_conf["drop_existing"] and utility.has_collection(collection):
                    print(f"⚠️ 기존 컬렉션 삭제 중: {collection}")
                    utility.drop_collection(collection)
                
                # 새로 생성
                index_params = {"index_type": "AUTOINDEX", "metric_type": "IP"}
                search_params = {"metric_type": "IP"}
                
                # 🔧 Milvus 스키마 설정 추가
                schema_config = {
                    "auto_id": True,
                    "enable_dynamic_field": True,  # 동적 필드 허용
                }
                
                if self.embeddings_model is None:
                    print("🚫 임베딩 모델이 초기화되지 않아 Milvus 생성 중단")
                    return False
                self.vectorstore = Milvus.from_documents(
                    documents=new_splits,
                    embedding=self.embeddings_model,  # type: ignore
                    collection_name=collection,
                    connection_args={"host": host_env, "port": port},
                    index_params=index_params,
                    search_params=search_params,
                    **schema_config
                )
                print(f"✅ 새 Milvus 벡터스토어 생성: {len(new_splits)}개 청크")

            # retriever 업데이트
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.milvus_conf["topk"]})
            
            # 파일 목록 업데이트
            self.files_in_vectorstore = pdf_files
            
            return True
        
        return False
    
    def retrieve_documents(self, query: str, subject_area: str = "", weakness_concepts: Optional[List[str]] = None) -> Dict[str, Any]:
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
            
            # 사용된 소스 파일 분석 (Milvus 문서에는 source_file이 없을 수 있으므로 보완)
            source_files = []
            for doc in documents:
                src = doc.metadata.get('source_file') or doc.metadata.get('subject') or 'milvus'
                source_files.append(src)
            used_sources = list(Counter(source_files).keys())
            
            return {
                "documents": documents,
                "used_sources": used_sources,
                "query": enhanced_query
            }
            
        except Exception as e:
            return {"error": f"문서 검색 오류: {e}"}
    
    def prepare_context(self, documents: List[Document], weakness_concepts: Optional[List[str]] = None) -> str:
        """
        검색된 문서에서 컨텍스트 준비
        
        Args:
            documents: 검색된 문서 리스트
            weakness_concepts: 취약점 개념 리스트
            
        Returns:
            준비된 컨텍스트 문자열
        """
        if not documents:
            print("[DEBUG] prepare_context: no documents provided")
            return ""
        
        print(f"[DEBUG] prepare_context: processing {len(documents)} documents")
        
        # 취약점 개념과 관련된 내용을 우선적으로 선별
        key_sents = []
        weakness_related_sents = []
        
        for i, doc in enumerate(documents):
            print(f"[DEBUG] Document {i+1}: content length={len(doc.page_content)}")
            lines = doc.page_content.split("\n")
            print(f"[DEBUG] Document {i+1}: {len(lines)} lines")
            
            for line_num, line in enumerate(lines):
                line = line.strip()
                if len(line) > 30:  # 최소 길이를 30으로 줄임
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
                        "구성", "절차", "장점", "단점", "방법", "기능",
                        "자료구조", "스택", "큐", "리스트", "트리", "그래프"
                    ])
                    
                    if is_weakness_related:
                        weakness_related_sents.append(line)
                        print(f"[DEBUG] Weakness related line: {line[:100]}...")
                    elif is_important or len(line) > 80:  # 길이 기준을 80으로 줄임
                        key_sents.append(line)
                        print(f"[DEBUG] Important line: {line[:100]}...")
        
        print(f"[DEBUG] Found {len(weakness_related_sents)} weakness-related sentences")
        print(f"[DEBUG] Found {len(key_sents)} important sentences")
        
        # 취약점 관련 내용을 앞쪽에, 일반 내용을 뒤쪽에 배치
        all_sents = weakness_related_sents + key_sents
        print(f"[DEBUG] Total sentences: {len(all_sents)}")
        
        # 최대 30개 문장으로 제한하고 최소 1개는 보장
        context_sents = all_sents[:30] if all_sents else []
        if not context_sents and documents:
            # 아무것도 선택되지 않았으면 첫 번째 문서의 첫 몇 줄을 사용
            first_doc = documents[0]
            lines = first_doc.page_content.split("\n")
            context_sents = [line.strip() for line in lines[:5] if len(line.strip()) > 20]
            print(f"[DEBUG] Using fallback context: {len(context_sents)} lines from first document")
        
        context = "\n".join(context_sents)
        print(f"[DEBUG] Final context length: {len(context)} characters")
        
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
            "embeddings_model": "jhgan/ko-sroberta-multitask" if self.embeddings_model else None,
            "vectorstore_type": "Milvus",
            "milvus_config": self.milvus_conf
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
