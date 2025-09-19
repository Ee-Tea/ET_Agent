"""
MilvusDB 사용을 위한 헬퍼 함수들
각 에이전트에서 MilvusDB 연결 정보를 쉽게 사용할 수 있도록 도와주는 유틸리티
"""

import time
from typing import Dict, List, Any, Optional
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, Collection, utility

def get_milvus_connection_info(milvus_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    MilvusDB 연결 정보에서 기본 정보 추출
    
    Args:
        milvus_data: Supervisor에서 전달받은 MilvusDB 연결 정보
        
    Returns:
        연결 상태 및 MilvusDBManager 인스턴스
    """
    if not milvus_data or not milvus_data.get("connection_status", False):
        return {"connected": False}
    
    # MilvusDBManager 인스턴스 생성
    from common.milvus_manager import MilvusDBManager
    
    milvus_manager = MilvusDBManager(
        host=milvus_data.get("host") or os.getenv("MILVUS_HOST", "localhost"),
        port=milvus_data.get("port") or os.getenv("MILVUS_PORT", "19530"),
        embedding_model_name=milvus_data.get("embedding_model_name", "jhgan/ko-sroberta-multitask")
    )
    
    # 연결 시도 (재시도 로직 포함)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if milvus_manager.connect():
                return {
                    "connected": True,
                    "milvus_manager": milvus_manager
                }
            else:
                if attempt < max_retries - 1:
                    print(f"⚠️ MilvusDB 연결 실패, 재시도 {attempt + 1}/{max_retries}")
                    time.sleep(1)  # 1초 대기
                else:
                    print(f"❌ MilvusDB 연결 최종 실패")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ MilvusDB 연결 예외, 재시도 {attempt + 1}/{max_retries}: {e}")
                time.sleep(1)
            else:
                print(f"❌ MilvusDB 연결 최종 예외: {e}")
    
    return {"connected": False}

def search_milvus_documents(
    milvus_data: Dict[str, Any], 
    collection_name: str, 
    query: str, 
    k: int = 5
) -> List[Document]:
    """
    MilvusDB에서 문서 검색 (통합 함수)
    
    Args:
        milvus_data: MilvusDB 연결 정보
        collection_name: 검색할 컬렉션 이름 (예: "concept_summary", "problems", "agri_disaster_docs" 등)
        query: 검색 쿼리
        k: 검색할 문서 수
        
    Returns:
        검색된 문서 리스트
    """
    connection_info = get_milvus_connection_info(milvus_data)
    
    if not connection_info["connected"]:
        print("⚠️ MilvusDB가 연결되지 않음")
        return []
    
    milvus_manager = connection_info.get("milvus_manager")
    if not milvus_manager:
        print("⚠️ MilvusDB 관리자가 없음")
        return []
    
    try:
        # 컬렉션 존재 확인
        if not milvus_manager.collection_exists(collection_name):
            print(f"⚠️ 컬렉션 '{collection_name}' 존재하지 않음")
            return []
        
        # 문서 검색
        documents = milvus_manager.search_documents_by_collection(collection_name, query, k)
        print(f"✅ MilvusDB 검색 완료: {collection_name} - {len(documents)}개 문서")
        return documents
    except Exception as e:
        print(f"❌ MilvusDB 검색 실패: {e}")
        return []

def search_milvus_documents_by_subject(
    milvus_data: Dict[str, Any], 
    collection_name: str, 
    subject_area: str, 
    k: int = 5
) -> List[Document]:
    """
    Teacher용: 과목명으로 MilvusDB에서 문서 검색
    
    Args:
        milvus_data: MilvusDB 연결 정보
        collection_name: 검색할 컬렉션 이름 (예: "concepts", "problems")
        subject_area: 과목명 (예: "소프트웨어개발", "데이터베이스구축")
        k: 검색할 문서 수
        
    Returns:
        검색된 문서 리스트
    """
    connection_info = get_milvus_connection_info(milvus_data)
    
    if not connection_info["connected"]:
        print("⚠️ MilvusDB가 연결되지 않음")
        return []
    
    milvus_manager = connection_info.get("milvus_manager")
    if not milvus_manager:
        print("⚠️ MilvusDB 관리자가 없음")
        return []
    
    # 과목명 별칭 매핑 (띄어쓰기 포함)
    subject_aliases = {
        "소프트웨어설계": ["소프트웨어 설계"],
        "소프트웨어개발": ["소프트웨어 개발"],
        "데이터베이스구축": ["데이터베이스 구축"],
        "프로그래밍언어활용": ["프로그래밍 언어 활용"],
        "정보시스템구축관리": ["정보시스템 구축 관리"],
    }
    
    # 띄어쓰기가 있는 과목명으로 변환
    actual_subject = subject_aliases.get(subject_area, subject_area)
    if isinstance(actual_subject, list):
        actual_subject = actual_subject[0]  # 첫 번째 별칭 사용
    
    print(f"🔍 과목명 변환: '{subject_area}' → '{actual_subject}'")
    
    try:
        # 먼저 필터 없이 검색해보기
        documents = milvus_manager.search_documents_by_collection_with_filter(
            collection_name=collection_name,
            query=actual_subject,  # 띄어쓰기가 있는 과목명으로 검색
            filter_expr=None,  # 필터 없이 검색
            k=k
        )
        
        # 필터가 있는 경우도 시도해보기
        if not documents:
            try:
                # subject 필드로 필터링 (띄어쓰기가 있는 과목명으로)
                documents = milvus_manager.search_documents_by_collection_with_filter(
                    collection_name=collection_name,
                    query=actual_subject,
                    filter_expr=f'subject == "{actual_subject}"',
                    k=k
                )
                print(f"🔍 필터 검색 시도: subject == '{actual_subject}'")
            except Exception as filter_error:
                print(f"⚠️ 필터 검색 실패, 일반 검색 사용: {filter_error}")
                # 필터 없이 다시 시도
                try:
                    documents = milvus_manager.search_documents_by_collection_with_filter(
                        collection_name=collection_name,
                        query=actual_subject,
                        filter_expr=None,
                        k=k
                    )
                except Exception as e:
                    print(f"❌ 일반 검색도 실패: {e}")
        
        print(f"✅ MilvusDB 과목 검색 완료: {collection_name} - {actual_subject} - {len(documents)}개 문서")
        return documents
    except Exception as e:
        print(f"❌ MilvusDB 과목 검색 실패: {e}")
        return []

def get_milvus_retriever(
    milvus_data: Dict[str, Any], 
    collection_name: str, 
    k: int = 5
):
    """
    MilvusDB 리트리버 가져오기 (통합 함수)
    
    Args:
        milvus_data: MilvusDB 연결 정보
        collection_name: 리트리버를 가져올 컬렉션 이름
        k: 검색할 문서 수
        
    Returns:
        리트리버 객체 또는 None
    """
    connection_info = get_milvus_connection_info(milvus_data)
    
    if not connection_info["connected"]:
        print("⚠️ MilvusDB가 연결되지 않음")
        return None
    
    milvus_manager = connection_info.get("milvus_manager")
    if not milvus_manager:
        print("⚠️ MilvusDB 관리자가 없음")
        return None
    
    try:
        # 컬렉션 존재 확인
        if not milvus_manager.collection_exists(collection_name):
            print(f"⚠️ 컬렉션 '{collection_name}' 존재하지 않음")
            return None
        
        # 리트리버 생성
        retriever = milvus_manager.get_retriever_by_collection(collection_name, k)
        print(f"✅ MilvusDB 리트리버 생성 완료: {collection_name}")
        return retriever
    except Exception as e:
        print(f"❌ MilvusDB 리트리버 생성 실패: {e}")
        return None

def create_context_from_documents(documents: List[Document], 
                                 max_length: int = 500000) -> str:
    """
    검색된 문서들로부터 컨텍스트 생성
    
    Args:
        documents: 검색된 문서 리스트
        max_length: 최대 컨텍스트 길이
        
    Returns:
        생성된 컨텍스트 문자열
    """
    if not documents:
        return ""
    
    # 문서 내용 합치기
    context_parts = []
    current_length = 0
    
    for doc in documents:
        content = doc.page_content
        if current_length + len(content) <= max_length:
            context_parts.append(content)
            current_length += len(content)
        else:
            # 남은 공간만큼만 추가
            remaining = max_length - current_length
            if remaining > 100:  # 최소 100자 이상일 때만 추가
                context_parts.append(content[:remaining] + "...")
            break
    
    return "\n\n".join(context_parts)

