import json
import os
import uuid
from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import openai
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MilvusDBManager:
    def __init__(self, host: str = None, port: str = None, openai_api_key: str = None):
        """MilvusDB 연결 관리자 초기화"""
        # 환경변수에서 MilvusDB 연결 정보 가져오기
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        print(self.host)
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        print(self.port)
        self.collection_name = "concept_summary"
        self.dimension = 1536  # OpenAI text-embedding-ada-002 임베딩 차원
        self.openai_client = None
        self.collection = None
        
        # OpenAI API 키 설정
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            logger.warning("OpenAI API 키가 설정되지 않았습니다. 환경변수 OPENAI_API_KEY를 설정하거나 openai_api_key 파라미터를 전달하세요.")
        
    def connect(self):
        """MilvusDB에 연결"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            logger.info(f"MilvusDB에 연결되었습니다: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"MilvusDB 연결 실패: {e}")
            return False
    
    def load_embedding_model(self):
        """OpenAI 임베딩 모델 초기화"""
        try:
            if not openai.api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            
            # OpenAI 클라이언트 초기화
            self.openai_client = openai.OpenAI(api_key=openai.api_key)
            
            # 간단한 테스트 임베딩 생성
            test_response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input="테스트"
            )
            
            if test_response.data and len(test_response.data[0].embedding) == self.dimension:
                logger.info("OpenAI 임베딩 모델이 성공적으로 초기화되었습니다.")
                return True
            else:
                logger.error("OpenAI 임베딩 모델 초기화 실패")
                return False
                
        except Exception as e:
            logger.error(f"OpenAI 임베딩 모델 초기화 실패: {e}")
            return False
    
    def create_collection(self):
        """컬렉션 생성"""
        try:
            # 기존 컬렉션이 있다면 삭제
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"기존 컬렉션 '{self.collection_name}'을 삭제했습니다.")
            
            # 필드 스키마 정의 - item_title 길이를 2000자로 증가
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
                FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="item_id", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="item_title", dtype=DataType.VARCHAR, max_length=2000),  # 500 -> 2000으로 증가
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="chunk_size", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            
            # 컬렉션 스키마 생성
            schema = CollectionSchema(fields=fields, description="정보처리기사 시험 자료 임베딩 컬렉션")
            
            # 컬렉션 생성
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # 인덱스 생성
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            
            logger.info(f"컬렉션 '{self.collection_name}'이 생성되었습니다.")
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 생성 실패: {e}")
            return False
    
    def generate_embedding(self, text: str) -> List[float]:
        """텍스트를 OpenAI 임베딩 벡터로 변환"""
        try:
            if not self.openai_client:
                raise ValueError("OpenAI 클라이언트가 초기화되지 않았습니다.")
            
            # 텍스트 전처리 (너무 긴 텍스트는 잘라내기)
            if len(text) > 8000:  # OpenAI text-embedding-ada-002 토큰 제한
                text = text[:8000]
            
            # OpenAI 임베딩 생성
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            if response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                return embedding
            else:
                logger.error("OpenAI 임베딩 응답이 비어있습니다.")
                return [0.0] * self.dimension
                
        except Exception as e:
            logger.error(f"OpenAI 임베딩 생성 실패: {e}")
            return [0.0] * self.dimension
    
    def chunk_content(self, content: str, max_length: int = 8000) -> List[str]:
        """긴 내용을 적절한 크기로 청킹"""
        if len(content) <= max_length:
            return [content]
        
        chunks = []
        current_chunk = ""
        
        # 문장 단위로 나누기 (마침표, 느낌표, 물음표 기준)
        sentences = content.replace('!', '.').replace('?', '.').split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 현재 청크에 문장을 추가했을 때 길이 확인
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += (sentence + ". ")
            else:
                # 현재 청크가 있으면 저장
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 새 청크 시작
                current_chunk = sentence + ". "
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content[:max_length]]

    def truncate_title(self, title: str, max_length: int = 1800) -> str:
        """제목을 적절한 길이로 자르기"""
        if len(title) <= max_length:
            return title
        return title[:max_length-3] + "..."

    def process_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """JSON 파일을 처리하여 데이터 추출"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            processed_data = []
            
            # 파일명에서 과목 추출
            filename = os.path.basename(file_path)
            subject = data.get('subject', '정보처리기사')
            
            # items 배열이 있는 경우
            if 'items' in data and isinstance(data['items'], list):
                for item in data['items']:
                    # 제목 길이 제한
                    title = self.truncate_title(item.get('item_title', ''))
                    # 내용은 청킹
                    content_chunks = self.chunk_content(item.get('content', ''), max_length=8000)
                    
                    for i, chunk in enumerate(content_chunks):
                        # 청크가 여러 개인 경우 제목에 청크 번호 추가
                        final_title = title + f" (청크 {i+1})" if len(content_chunks) > 1 else title
                        
                        processed_item = {
                            'id': str(uuid.uuid4()),
                            'subject': subject,
                            'item_id': str(item.get('item_id', '')) + f"_chunk_{i+1}" if len(content_chunks) > 1 else str(item.get('item_id', '')),
                            'item_title': final_title,
                            'content': chunk,
                            'chunk_size': len(chunk)
                        }
                        processed_data.append(processed_item)
            
            # 다른 구조의 JSON 파일 처리
            elif 'content' in data:
                # 제목 길이 제한
                title = self.truncate_title(data.get('item_title', ''))
                # 내용은 청킹
                content_chunks = self.chunk_content(data.get('content', ''), max_length=8000)
                
                for i, chunk in enumerate(content_chunks):
                    # 청크가 여러 개인 경우 제목에 청크 번호 추가
                    final_title = title + f" (청크 {i+1})" if len(content_chunks) > 1 else title
                    
                    processed_item = {
                        'id': str(uuid.uuid4()),
                        'subject': subject,
                        'item_id': str(data.get('item_id', '')) + f"_chunk_{i+1}" if len(content_chunks) > 1 else str(data.get('item_id', '')),
                        'item_title': final_title,
                        'content': chunk,
                        'chunk_size': len(chunk)
                    }
                    processed_data.append(processed_item)
            
            logger.info(f"파일 '{filename}'에서 {len(processed_data)}개의 항목을 추출했습니다.")
            return processed_data
            
        except Exception as e:
            logger.error(f"JSON 파일 처리 실패 '{file_path}': {e}")
            return []
    
    def insert_data(self, data_list: List[Dict[str, Any]]):
        """데이터를 MilvusDB에 삽입"""
        try:
            if not data_list:
                logger.warning("삽입할 데이터가 없습니다.")
                return
            
            # 임베딩 생성
            embeddings = []
            for i, item in enumerate(data_list):
                logger.info(f"임베딩 생성 중... ({i+1}/{len(data_list)})")
                
                # 제목과 내용을 결합하여 임베딩 생성
                combined_text = f"{item['item_title']} {item['content']}"
                embedding = self.generate_embedding(combined_text)
                embeddings.append(embedding)
                
                # API 호출 제한을 위한 짧은 대기
                if (i + 1) % 10 == 0:
                    import time
                    time.sleep(0.1)
            
            # 데이터 준비
            insert_data = [
                [item['id'] for item in data_list],
                [item['subject'] for item in data_list],
                [item['item_id'] for item in data_list],
                [item['item_title'] for item in data_list],
                [item['content'] for item in data_list],
                [item['chunk_size'] for item in data_list],
                embeddings
            ]
            
            # 데이터 삽입
            self.collection.insert(insert_data)
            self.collection.flush()
            
            logger.info(f"{len(data_list)}개의 항목이 성공적으로 삽입되었습니다.")
            
        except Exception as e:
            logger.error(f"데이터 삽입 실패: {e}")
    
    def load_all_json_files(self, json_dir: str):
        """지정된 디렉토리의 모든 JSON 파일을 로드하여 MilvusDB에 저장"""
        try:
            if not os.path.exists(json_dir):
                logger.error(f"디렉토리가 존재하지 않습니다: {json_dir}")
                return
            
            # JSON 파일 목록 가져오기
            json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
            logger.info(f"총 {len(json_files)}개의 JSON 파일을 발견했습니다.")
            
            total_items = 0
            
            for json_file in json_files:
                file_path = os.path.join(json_dir, json_file)
                logger.info(f"파일 처리 중: {json_file}")
                
                # JSON 파일 처리
                data_list = self.process_json_file(file_path)
                if data_list:
                    # 데이터 삽입
                    self.insert_data(data_list)
                    total_items += len(data_list)
                
                logger.info(f"파일 '{json_file}' 처리 완료")
            
            logger.info(f"모든 파일 처리 완료. 총 {total_items}개의 항목이 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"JSON 파일 로드 실패: {e}")
    
    def search_similar(self, query: str, top_k: int = 5):
        """유사한 내용 검색"""
        try:
            if self.collection is None:
                logger.error("컬렉션이 로드되지 않았습니다.")
                return []
            
            # 컬렉션이 로드되지 않은 경우 로드
            try:
                self.collection.load()
            except Exception as e:
                logger.warning(f"컬렉션 로드 중 오류 (이미 로드된 경우 무시): {e}")
            
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
            logger.error(f"검색 실패: {e}")
            return []

def main():
    """메인 함수"""
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("환경변수 OPENAI_API_KEY를 설정해주세요.")
        logger.info("예시: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # MilvusDB 관리자 초기화
    db_manager = MilvusDBManager()
    
    # MilvusDB 연결
    if not db_manager.connect():
        logger.error("MilvusDB 연결에 실패했습니다.")
        return
    
    # OpenAI 임베딩 모델 초기화
    if not db_manager.load_embedding_model():
        logger.error("OpenAI 임베딩 모델 초기화에 실패했습니다.")
        return
    
    # 컬렉션 생성
    if not db_manager.create_collection():
        logger.error("컬렉션 생성에 실패했습니다.")
        return
    
    # JSON 파일들이 있는 디렉토리 경로
    json_dir = "teacher/agents/retrieve/data/json"
    
    # 모든 JSON 파일 로드 및 저장
    db_manager.load_all_json_files(json_dir)
    
    # 테스트 검색
    logger.info("테스트 검색을 실행합니다...")
    test_query = "소프트웨어 생명주기"
    results = db_manager.search_similar(test_query, top_k=3)
    
    logger.info(f"쿼리: '{test_query}'에 대한 검색 결과:")
    for i, result in enumerate(results, 1):
        logger.info(f"{i}. 점수: {result['score']:.4f}")
        logger.info(f"   과목: {result['subject']}")
        logger.info(f"   제목: {result['item_title']}")
        logger.info(f"   내용: {result['content'][:100]}...")
        logger.info("---")

if __name__ == "__main__":
    main()
