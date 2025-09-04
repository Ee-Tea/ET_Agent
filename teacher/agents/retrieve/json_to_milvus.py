import json
import os
import uuid
from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MilvusDBManager:
    def __init__(self, host: str = None, port: str = None, 
                 chunk_tokens: int = 256, chunk_overlap: int = 32, 
                 min_chunk_chars: int = 50, min_chunk_tokens: int = 0):
        
        """MilvusDB 연결 관리자 초기화"""
        # 환경변수에서 MilvusDB 연결 정보 가져오기
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        logger.debug(f"Milvus host={self.host}, port={self.port}")
        self.collection_name = "concepts"
        self.dimension = 768  # ko-sroberta-multitask 임베딩 차원
        self.embeddings_model = None
        self.collection = None
        self.chunk_tokens = int(os.getenv("CHUNK_TOKENS", chunk_tokens))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", chunk_overlap))
        self.min_chunk_chars = int(os.getenv("CHUNK_MIN_CHARS", min_chunk_chars))   # 문자 기준
        self.min_chunk_tokens = int(os.getenv("CHUNK_MIN_TOKENS", min_chunk_tokens))  # 토큰 기준(0이면 비활성)
        self._tokenizer = None
        
    def connect(self):
        """MilvusDB에 연결"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            logger.info(f"MilvusDB에 연결되었습니다: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"MilvusDB 연결 실패: {e}")
            return False
        
    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            if not self.embeddings_model:
                raise ValueError("임베딩 모델이 먼저 초기화되어야 합니다.")
            # SentenceTransformer는 내부에 tokenizer를 갖고 있음
            self._tokenizer = getattr(self.embeddings_model, "tokenizer", None)
            if self._tokenizer is None:
                # 호환성이 떨어지는 모델일 경우 대비
                raise RuntimeError("SentenceTransformer tokenizer를 찾지 못했습니다.")
        return self._tokenizer

    
    def _passes_min_size(self, text: str) -> bool:
        # 토큰 기준 활성화 시
        if self.min_chunk_tokens > 0:
            tok = self._ensure_tokenizer()
            n_tokens = len(tok(text, add_special_tokens=False, truncation=False)["input_ids"])
            if n_tokens < self.min_chunk_tokens:
                return False
        # 문자 기준
        if self.min_chunk_chars > 0 and len(text) < self.min_chunk_chars:
            return False
        return True

    def load_embedding_model(self):
        """HuggingFace 임베딩 모델 초기화"""
        try:
            # HuggingFace 임베딩 모델 로드
            self.embeddings_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            
            # 간단한 테스트 임베딩 생성
            test_embedding = self.embeddings_model.encode("테스트")
            
            if len(test_embedding) == self.dimension:
                logger.info("HuggingFace 임베딩 모델이 성공적으로 초기화되었습니다.")
                return True
            else:
                logger.error("HuggingFace 임베딩 모델 초기화 실패")
                return False
                
        except Exception as e:
            logger.error(f"HuggingFace 임베딩 모델 초기화 실패: {e}")
            return False
    
    def create_collection(self, drop: bool = False):
        if utility.has_collection(self.collection_name):
            if drop:
                utility.drop_collection(self.collection_name)
                logger.info(f"기존 '{self.collection_name}' 삭제")
            else:
                self.collection = Collection(self.collection_name)
                logger.info(f"기존 '{self.collection_name}' 로드")
                return True

        # ↓ 여기부터는 '없거나(drop 후)' 새로 만들기
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="item_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="item_title", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="chunk_size", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        ]
        schema = CollectionSchema(fields=fields, description="개념 요약 임베딩")
        self.collection = Collection(name=self.collection_name, schema=schema)
        logger.info(f"컬렉션 '{self.collection_name}' 생성")
        return True

    
    def build_index(self):
        index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
        self.collection.create_index(field_name="embedding", index_params=index_params)
        self.collection.load()
        logger.info("컬렉션 인덱스가 성공적으로 생성되었습니다.")
        
    def generate_embedding(self, text: str) -> List[float]:
        """텍스트를 HuggingFace 임베딩 벡터로 변환"""
        try:
            if not self.embeddings_model:
                raise ValueError("HuggingFace 임베딩 모델이 초기화되지 않았습니다.")
            
            
            # HuggingFace 임베딩 생성
            embedding = self.embeddings_model.encode(text)
            
            if embedding is not None and len(embedding) > 0:
                # numpy 배열을 리스트로 변환
                return embedding.tolist()
            else:
                logger.error("HuggingFace 임베딩 응답이 비어있습니다.")
                return [0.0] * self.dimension
                
        except Exception as e:
            logger.error(f"HuggingFace 임베딩 생성 실패: {e}")
            return [0.0] * self.dimension
    
    def chunk_by_tokens(self, text: str, chunk_tokens: int = None, overlap: int = None) -> list[str]:
        tok = self._ensure_tokenizer()
        chunk_tokens = chunk_tokens or self.chunk_tokens
        overlap = overlap or self.chunk_overlap

        # 모델 max seq length 방어
        try:
            model_max = int(getattr(self.embeddings_model, "max_seq_length", 0)) \
                        or int(getattr(self.embeddings_model, "get_max_seq_length", lambda: 0)() or 0)
            if model_max and chunk_tokens > model_max:
                chunk_tokens = max(8, model_max - 8)
        except Exception:
            pass

        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        ids = enc.get("input_ids", [])
        offs = enc.get("offset_mapping", [])
        if not ids:
            return [text] if text else []

        chunks, start = [], 0
        step = max(1, chunk_tokens - overlap)
        while start < len(ids):
            end = min(len(ids), start + chunk_tokens)
            s, e = offs[start][0], offs[end - 1][1]
            piece = text[s:e].strip()
            if piece:
                chunks.append(piece)
            start += step
        return chunks


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
                    content_chunks = self.chunk_by_tokens(
                        item.get('content', ''),
                        chunk_tokens=self.chunk_tokens,
                        overlap=self.chunk_overlap
                    )
                    content_chunks = [c for c in content_chunks if self._passes_min_size(c)]

                    
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
                content_chunks = self.chunk_by_tokens(
                    data.get('content', ''),
                    chunk_tokens=self.chunk_tokens,
                    overlap=self.chunk_overlap
                )
                content_chunks = [c for c in content_chunks if self._passes_min_size(c)]

                
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
            
            # ① 텍스트 모으기
            texts = [f"{it['item_title']} {it['content']}" for it in data_list]

            # ② 배치 임베딩 (환경변수로 배치 크기 조절)
            batch_size = int(os.getenv("EMBED_BATCH", 64))
            embeddings = self.embeddings_model.encode(
                texts, batch_size=batch_size, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True
            ).tolist()

            # ③ 나머지는 동일
            insert_data = [
                [it['id'] for it in data_list],
                [it['subject'] for it in data_list],
                [it['item_id'] for it in data_list],
                [it['item_title'] for it in data_list],
                [it['content'] for it in data_list],
                [it['chunk_size'] for it in data_list],
                embeddings
            ]
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
            search_params = {"metric_type": "COSINE", "params": {"nprobe": int(os.getenv("NPROBE", 32))}}
            results = self.collection.search(
                [query_embedding], "embedding", search_params, limit=top_k,
                output_fields=["id","item_id","subject","item_title","content","chunk_size"]
            )
            
            # 결과 정리
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        'id': hit.entity.get('id'),
                        'item_id': hit.entity.get('item_id'),
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
    db = MilvusDBManager()
    if not db.connect(): return
    if not db.load_embedding_model(): return

    # 1) 컬렉션 먼저
    drop = os.getenv("MILVUS_DROP_COLLECTION", "false").lower() == "true"
    if not db.create_collection(drop=drop): return

    json_dir = "teacher/agents/retrieve/data/json"
    # 2) 데이터 적재
    json_dir = os.getenv("JSON_DIR", "teacher/agents/retrieve/data/json")
    db.load_all_json_files(json_dir)

    # 3) 마지막에 인덱스 빌드 & 로드
    db.build_index()

    # 테스트 검색
    logger.info("테스트 검색을 실행합니다...")
    test_query = "소프트웨어 생명주기"
    results = db.search_similar(test_query, top_k=3)
    
    # JSON 파일들이 있는 디렉토리 경로
    logger.info(f"쿼리: '{test_query}'에 대한 검색 결과:")
    for i, result in enumerate(results, 1):
        logger.info(f"{i}. 점수: {result['score']:.4f}")
        logger.info(f"   과목: {result['subject']}")
        logger.info(f"   제목: {result['item_title']}")
        logger.info(f"   내용: {result['content'][:100]}...")
        logger.info("---")

if __name__ == "__main__":
    main()