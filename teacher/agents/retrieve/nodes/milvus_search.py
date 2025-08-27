from langchain.tools import Tool
from typing import List, Dict, Any
import os
from pymilvus import connections, Collection
import openai
import logging

logger = logging.getLogger(__name__)

class MilvusSearchTool:
    """MilvusDB 벡터 유사도 검색을 위한 도구"""
    
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = "concept_summary"
        self.dimension = 1536
        self.collection = None
        self.openai_client = None
        
        # OpenAI API 키 설정
        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = openai.OpenAI(api_key=openai.api_key)
        
        # MilvusDB 연결
        self._connect_milvus()
    
    def _connect_milvus(self):
        """MilvusDB에 연결"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"MilvusDB에 연결되었습니다: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"MilvusDB 연결 실패: {e}")
            self.collection = None
    
    def generate_embedding(self, text: str) -> List[float]:
        """텍스트를 OpenAI 임베딩 벡터로 변환"""
        try:
            if not self.openai_client:
                raise ValueError("OpenAI 클라이언트가 초기화되지 않았습니다.")
            
            # 텍스트 전처리 (너무 긴 텍스트는 잘라내기)
            if len(text) > 8000:
                text = text[:8000]
            
            # OpenAI 임베딩 생성
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            else:
                logger.error("OpenAI 임베딩 응답이 비어있습니다.")
                return [0.0] * self.dimension
                
        except Exception as e:
            logger.error(f"OpenAI 임베딩 생성 실패: {e}")
            return [0.0] * self.dimension
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """유사한 내용 검색"""
        try:
            if not self.collection:
                logger.error("MilvusDB 컬렉션이 연결되지 않았습니다.")
                return []
            
            # 쿼리 임베딩 생성
            query_embedding = self.generate_embedding(query)
            
            # 검색 파라미터
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 검색 실행
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["subject", "item_title", "content", "chunk_size"]
            )
            
            # 결과 정리
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        'score': hit.score,
                        'subject': hit.entity.get('subject'),
                        'item_title': hit.entity.get('item_title'),
                        'content': hit.entity.get('content'),
                        'chunk_size': hit.entity.get('chunk_size')
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
