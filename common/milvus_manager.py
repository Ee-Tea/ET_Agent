"""
MilvusDB 연결 및 관리 클래스
Supervisor에서 중앙 집중식으로 MilvusDB를 관리하고 각 에이전트에 주입
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from pymilvus import connections, Collection, utility
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_community.retrievers import MilvusRetriever
from langchain.schema import Document

logger = logging.getLogger(__name__)

class MilvusDBManager:
    """MilvusDB 중앙 관리자"""
    
    def __init__(self, 
                 host: str = None, 
                 port: str = None,
                 embedding_model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        MilvusDB 관리자 초기화
        
        Args:
            host: MilvusDB 호스트 (기본값: localhost)
            port: MilvusDB 포트 (기본값: 19530)
            embedding_model_name: 임베딩 모델명
        """
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.embedding_model_name = embedding_model_name
        
        # 연결 상태
        self.is_connected = False
        self.embeddings_model = None
        self.collections = {}  # 컬렉션별 Milvus 객체 캐시
        
        
        logger.info(f"MilvusDB Manager 초기화: {self.host}:{self.port}")
    
    def connect(self) -> bool:
        """MilvusDB에 연결"""
        try:
            # 기존 연결이 있으면 우선 해제 (멱등)
            try:
                connections.disconnect("default")
            except Exception:
                pass
            
            # 새 연결 생성
            connections.connect(alias="default", host=self.host, port=self.port)
            self.is_connected = True
            
            # 임베딩 모델 초기화
            self._init_embedding_model()
            
            logger.info(f"✅ MilvusDB 연결 성공: {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ MilvusDB 연결 실패: {e}")
            self.is_connected = False
            return False
    
    def _init_embedding_model(self):
        """임베딩 모델 초기화 (MilvusDB용)"""
        try:
            # MilvusDB 검색을 위한 임베딩 모델 초기화
            import os
            import torch
            
            # torch 설정 최소화
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU 비활성화
            torch.set_default_tensor_type('torch.FloatTensor')
            
            # HuggingFace 임베딩 모델 초기화
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},  # CPU 명시
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"✅ HuggingFace 임베딩 모델 초기화 완료: {self.embedding_model_name}")
            
        except Exception as e:
            logger.error(f"❌ 임베딩 모델 초기화 실패: {e}")
            # MilvusDB 검색을 위해 None으로 설정하지 않음
            self.embeddings_model = None
    
    def get_milvus_vectorstore(self, collection_name: str) -> Optional[Milvus]:
        """
        컬렉션 이름으로 직접 벡터스토어 가져오기 (캐시 사용)
        
        Args:
            collection_name: 컬렉션 이름
            
        Returns:
            Milvus 벡터스토어 객체 또는 None
        """
        if not self.is_connected:
            logger.warning("MilvusDB가 연결되지 않았습니다.")
            return None
        
        # 캐시에서 확인
        if collection_name in self.collections:
            return self.collections[collection_name]
        
        # 컬렉션 존재 여부 확인
        if not utility.has_collection(collection_name):
            logger.warning(f"컬렉션 {collection_name}이 존재하지 않습니다.")
            return None
        
        try:
            # 임베딩 모델 확인
            if not self.embeddings_model:
                logger.warning(f"⚠️ 임베딩 모델이 없습니다. MilvusDB 검색을 건너뜁니다.")
                return None
            
            # 컬렉션 스키마 확인하여 벡터 필드명 찾기
            collection = Collection(collection_name)
            text_field, vector_field = self._infer_collection_fields(collection)
            
            if not vector_field:
                logger.error(f"❌ 컬렉션 '{collection_name}'에 벡터 필드를 찾을 수 없습니다.")
                return None
            
            # 메트릭 타입 자동 감지
            metric_type = self._get_collection_metric_type(collection_name)
            
            # Milvus 벡터스토어 생성 (올바른 필드명과 메트릭 타입 사용)
            vectorstore = Milvus(
                embedding_function=self.embeddings_model,
                collection_name=collection_name,
                connection_args={"host": self.host, "port": self.port},
                text_field=text_field,
                vector_field=vector_field,
                search_params={"metric_type": metric_type, "params": {"nprobe": 10}}
            )
            
            # 캐시에 저장
            self.collections[collection_name] = vectorstore
            logger.info(f"✅ 벡터스토어 생성 완료: {collection_name} (text_field={text_field}, vector_field={vector_field})")
            return vectorstore
            
        except Exception as e:
            logger.error(f"❌ 벡터스토어 생성 실패: {collection_name} - {e}")
            return None
    
    
    
    def disconnect(self):
        """MilvusDB 연결 해제"""
        try:
            if "default" in connections.list_connections():
                connections.disconnect("default")
            self.is_connected = False
            self.collections.clear()
            logger.info("✅ MilvusDB 연결 해제 완료")
        except Exception as e:
            logger.error(f"❌ MilvusDB 연결 해제 실패: {e}")
    
    def __enter__(self):
        """Context manager 진입"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.disconnect()
    
    def search_documents_by_collection(self, collection_name: str, query: str, k: int = 5) -> List[Document]:
        """
        컬렉션 이름으로 직접 문서 검색
        
        Args:
            collection_name: 검색할 컬렉션 이름
            query: 검색 쿼리
            k: 검색할 문서 수
            
        Returns:
            검색된 문서 리스트
        """
        try:
            if not self.is_connected:
                logger.warning("⚠️ MilvusDB가 연결되지 않음")
                return []
            
            if not utility.has_collection(collection_name):
                logger.warning(f"⚠️ 컬렉션 '{collection_name}' 존재하지 않음")
                return []
            
            # 벡터스토어 생성
            vectorstore = self.get_milvus_vectorstore(collection_name)
            if not vectorstore:
                logger.warning(f"⚠️ 컬렉션 '{collection_name}' 벡터스토어 생성 실패")
                return []
            
            # 문서 검색
            documents = vectorstore.similarity_search(query, k=k)
            logger.info(f"✅ '{collection_name}'에서 {len(documents)}개 문서 검색")
            return documents
            
        except Exception as e:
            logger.error(f"❌ 컬렉션 '{collection_name}' 검색 실패: {e}")
            return []
    
    def get_retriever_by_collection(self, collection_name: str, k: int = 5):
        """
        컬렉션 이름으로 직접 리트리버 가져오기
        
        Args:
            collection_name: 리트리버를 가져올 컬렉션 이름
            k: 검색할 문서 수
            
        Returns:
            리트리버 객체 또는 None
        """
        try:
            if not self.is_connected:
                logger.warning("⚠️ MilvusDB가 연결되지 않음")
                return None
            
            if not utility.has_collection(collection_name):
                logger.warning(f"⚠️ 컬렉션 '{collection_name}' 존재하지 않음")
                return None
            
            # 벡터스토어 생성
            vectorstore = self.get_milvus_vectorstore(collection_name)
            if not vectorstore:
                logger.warning(f"⚠️ 컬렉션 '{collection_name}' 벡터스토어 생성 실패")
                return None
            
            # 리트리버 생성
            retriever = MilvusRetriever(
                vectorstore=vectorstore,
                search_kwargs={"k": k}
            )
            logger.info(f"✅ '{collection_name}' 리트리버 생성 완료")
            return retriever
            
        except Exception as e:
            logger.error(f"❌ 컬렉션 '{collection_name}' 리트리버 생성 실패: {e}")
            return None

    def collection_exists(self, collection_name: str) -> bool:
        """컬렉션 존재 여부 확인"""
        try:
            if not self.is_connected:
                return False
            return utility.has_collection(collection_name)
        except Exception as e:
            logger.error(f"❌ 컬렉션 '{collection_name}' 존재 확인 실패: {e}")
            return False

    def search_documents_by_collection_with_filter(
        self, 
        collection_name: str, 
        query: str, 
        filter_expr: str = None,
        k: int = 5
    ) -> List[Document]:
        """필터를 사용하여 문서 검색 (Teacher용 과목명 검색)"""
        try:
            if not self.is_connected:
                logger.warning("⚠️ MilvusDB가 연결되지 않음")
                return []
            
            if not self.collection_exists(collection_name):
                logger.warning(f"⚠️ 컬렉션 '{collection_name}' 존재하지 않음")
                return []
            
            # 벡터스토어 생성 (필드명과 메트릭 타입 자동 추론)
            collection = Collection(collection_name)
            text_field, vector_field = self._infer_collection_fields(collection)
            
            if not vector_field:
                logger.warning(f"⚠️ 컬렉션 '{collection_name}'에 벡터 필드를 찾을 수 없습니다.")
                return []
            
            # 메트릭 타입 자동 감지
            metric_type = self._get_collection_metric_type(collection_name)
            
            vectorstore = Milvus(
                embedding_function=self.embeddings_model,
                collection_name=collection_name,
                connection_args={"host": self.host, "port": self.port},
                text_field=text_field,
                vector_field=vector_field,
                search_params={"metric_type": metric_type, "params": {"nprobe": 10}}
            )
            
            if not vectorstore:
                logger.warning(f"⚠️ 컬렉션 '{collection_name}' 벡터스토어 생성 실패")
                return []
            
            # 필터가 있으면 필터와 함께 검색, 없으면 일반 검색
            if filter_expr:
                documents = vectorstore.similarity_search(query, k=k, expr=filter_expr)
            else:
                documents = vectorstore.similarity_search(query, k=k)
            
            logger.info(f"✅ '{collection_name}' 필터 검색 완료: {len(documents)}개 문서")
            return documents
            
        except Exception as e:
            logger.error(f"❌ 컬렉션 '{collection_name}' 필터 검색 실패: {e}")
            return []

    def _infer_collection_fields(self, collection: Collection) -> tuple:
        """컬렉션 스키마에서 텍스트 필드와 벡터 필드 추론"""
        from pymilvus import DataType
        
        text_field = None
        vector_field = None
        
        for field in collection.schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                # embedding 필드를 우선적으로 선택
                if field.name == "embedding" or vector_field is None:
                    vector_field = field.name
            elif field.dtype == DataType.VARCHAR and text_field is None:
                # 텍스트 필드 후보들 (스키마에 맞게 우선순위 설정)
                if field.name in ["content", "text", "item_title", "page_content", "question", "title", "description"]:
                    text_field = field.name
        
        # 텍스트 필드를 찾지 못했으면 첫 번째 VARCHAR 필드 사용
        if not text_field:
            for field in collection.schema.fields:
                if field.dtype == DataType.VARCHAR:
                    text_field = field.name
                    break
        
        logger.info(f"컬렉션 필드 추론: text_field='{text_field}', vector_field='{vector_field}'")
        return text_field, vector_field

    def _get_collection_metric_type(self, collection_name: str) -> str:
        """컬렉션의 메트릭 타입 확인"""
        try:
            collection = Collection(collection_name)
            # 인덱스 정보에서 메트릭 타입 확인
            for index in collection.indexes:
                if hasattr(index, 'params') and 'metric_type' in index.params:
                    metric_type = index.params['metric_type']
                    logger.info(f"컬렉션 '{collection_name}' 메트릭 타입: {metric_type}")
                    return metric_type
            
            # 기본값은 L2
            logger.warning(f"컬렉션 '{collection_name}' 메트릭 타입을 찾을 수 없음, L2 사용")
            return "L2"
        except Exception as e:
            logger.error(f"컬렉션 '{collection_name}' 메트릭 타입 확인 실패: {e}")
            return "L2"


