from langchain.tools import Tool
from typing import List, Dict, Any
import os
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class MilvusSearchTool:
    """MilvusDB 벡터 유사도 검색을 위한 도구"""
    
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "127.0.0.1")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = "concept_summary"
        self.dimension = 768  # ko-sroberta-multitask 임베딩 차원
        self.collection = None
        self.embeddings_model = None
        
        # HuggingFace 임베딩 모델 초기화
        try:
            self.embeddings_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            logger.info("HuggingFace 임베딩 모델이 초기화되었습니다.")
        except Exception as e:
            logger.error(f"HuggingFace 임베딩 모델 초기화 실패: {e}")
            self.embeddings_model = None
        
        # MilvusDB 연결
        self._connect_milvus()
    
    def _connect_milvus(self):
        """MilvusDB에 연결"""
        try:
            # 이미 연결되어 있으면 재사용
            if "default" not in connections.list_connections():
                connections.connect(alias="default", host=self.host, port=self.port)
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"MilvusDB에 연결되었습니다: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"MilvusDB 연결 실패: {e}")
            self.collection = None
    
    def generate_embedding(self, text: str) -> List[float]:
        """텍스트를 HuggingFace 임베딩 벡터로 변환"""
        try:
            if not self.embeddings_model:
                logger.error("HuggingFace 임베딩 모델이 초기화되지 않았습니다.")
                return []
            
            # 텍스트 전처리 (너무 긴 텍스트는 잘라내기)
            if len(text) > 8000:
                text = text[:8000]
            
            # HuggingFace 임베딩 생성
            embedding = self.embeddings_model.encode(text)
            
            if embedding is not None and len(embedding) > 0:
                # numpy 배열을 리스트로 변환
                return embedding.tolist()
            else:
                logger.error("HuggingFace 임베딩 응답이 비어있습니다.")
                return []
                
        except Exception as e:
            logger.error(f"HuggingFace 임베딩 생성 실패: {e}")
            return []
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """유사한 내용 검색"""
        try:
            if not self.collection:
                logger.error("MilvusDB 컬렉션이 연결되지 않았습니다.")
                self._connect_milvus()
                if not self.collection:
                    return []
            
            # 쿼리 임베딩 생성
            query_embedding = self.generate_embedding(query)
            
            # 임베딩이 생성되지 않은 경우 빈 결과 반환
            if not query_embedding:
                logger.info("HuggingFace 임베딩 모델이 초기화되지 않았습니다. 검색을 스킵합니다.")
                return []
            
            # 검색 파라미터
            search_params_primary = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            search_params_fallback = {"metric_type": "COSINE", "params": {"ef": 64}}
            
            # 검색 실행
            try:
                results = self.collection.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params_primary,
                    limit=top_k,
                    output_fields=["subject", "content"]
                )
            except Exception:
                # 인덱스 파라미터 불일치 시 폴백
                results = self.collection.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params_fallback,
                    limit=top_k,
                    output_fields=["subject", "content"]
                )
            
            # 결과 정리
            search_results = []
            for hits in results:
                for hit in hits:
                    # entity 필드 접근은 컬렉션 스키마에 따라 다르므로 안전하게 처리
                    entity = getattr(hit, 'entity', None)
                    getter = getattr(entity, 'get', None)
                    subject = getter('subject') if callable(getter) else None
                    content = getter('content') if callable(getter) else None
                    search_results.append({
                        'score': hit.score,
                        'subject': subject,
                        'content': content,
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"MilvusDB 검색 실패: {e}")
            return []
    
    def run(self, query: str) -> str:
        """검색 실행 및 결과 반환"""
        try:
            # 유사도 검색 실행
            results = self.search_similar(query, top_k=5)
            
            if not results:
                # HuggingFace 임베딩 모델이 초기화되지 않은 경우
                if not self.embeddings_model:
                    return "HuggingFace 임베딩 모델이 초기화되지 않았습니다. 검색을 스킵합니다."
                else:
                    return "MilvusDB에서 관련 정보를 찾을 수 없습니다."
            
            # 검색 결과를 구조화된 텍스트로 변환
            context_parts = []
            for i, result in enumerate(results, 1):
                context_part = f"[{i}] 과목: {result['subject']}\n"
                context_part += f"제목: {result['item_title']}\n"
                context_part += f"내용: {result['content']}\n"
                context_part += f"유사도 점수: {result['score']:.4f}\n"
                context_part += "-" * 50 + "\n"
                context_parts.append(context_part)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"MilvusDB 검색 도구 실행 실패: {e}")
            return f"MilvusDB 검색 중 오류가 발생했습니다: {str(e)}"

# MilvusDB 검색 도구 인스턴스 생성
milvus_tool = Tool(
    name="MilvusDB Vector Search",
    description="MilvusDB에서 벡터 유사도 검색을 통해 관련 정보를 찾습니다. 정보처리기사 시험 자료에서 질문과 유사한 내용을 검색할 때 사용합니다.",
    func=lambda q: MilvusSearchTool().run(q)
)
