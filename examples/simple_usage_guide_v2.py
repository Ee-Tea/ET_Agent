"""
MilvusDB 사용법 간단 가이드 v2 - 통합된 함수 사용
"""

# 1. 기본 사용법 (통합된 함수)
from common.milvus_helpers import search_milvus_documents, get_milvus_retriever, create_context_from_documents

def basic_usage_example():
    """기본 사용법 예시 - collection_name으로 직접 검색"""
    
    def teacher_example(state):
        milvus_data = state.get("milvus_data", {})
        
        # 직접 컬렉션 이름 사용
        documents = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name="concept_summary",  # 직접 컬렉션 이름
            query="데이터베이스 정규화",
            k=5
        )
        
        # 검색된 문서 사용
        if documents:
            for doc in documents:
                print(f"제목: {doc.metadata.get('title', 'N/A')}")
                print(f"내용: {doc.page_content[:100]}...")
        
        return state
    
    def farmer_example(state):
        milvus_data = state.get("milvus_data", {})
        
        # 직접 컬렉션 이름 사용
        documents = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name="crop_info",  # 직접 컬렉션 이름
            query="토마토 재배",
            k=5
        )
        
        # 검색된 문서 사용
        if documents:
            for doc in documents:
                print(f"제목: {doc.metadata.get('title', 'N/A')}")
                print(f"내용: {doc.page_content[:100]}...")
        
        return state

# 2. 리트리버 사용법 (통합된 함수)
def retriever_usage_example():
    """리트리버 사용법 예시 - collection_name으로 직접 검색"""
    
    def teacher_retriever_example(state):
        milvus_data = state.get("milvus_data", {})
        
        # 리트리버 생성
        retriever = get_milvus_retriever(
            milvus_data=milvus_data,
            collection_name="concept_summary",  # 직접 컬렉션 이름
            k=5
        )
        
        if retriever:
            # 리트리버로 검색
            documents = retriever.invoke("데이터베이스 질의")
            print(f"검색된 문서: {len(documents)}개")
        
        return state
    
    def farmer_retriever_example(state):
        milvus_data = state.get("milvus_data", {})
        
        # 리트리버 생성
        retriever = get_milvus_retriever(
            milvus_data=milvus_data,
            collection_name="crop_info",  # 직접 컬렉션 이름
            k=5
        )
        
        if retriever:
            # 리트리버로 검색
            documents = retriever.invoke("토마토 재배")
            print(f"검색된 문서: {len(documents)}개")
        
        return state

# 3. 컨텍스트 생성 사용법
def context_creation_example():
    """컨텍스트 생성 사용법 예시"""
    
    def create_context_example(state):
        milvus_data = state.get("milvus_data", {})
        
        # 문서 검색
        documents = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name="concept_summary",  # 직접 컬렉션 이름
            query="데이터베이스",
            k=5
        )
        
        # 컨텍스트 생성
        context = create_context_from_documents(
            documents=documents,
            max_length=2000  # 최대 2000자
        )
        
        if context:
            print(f"생성된 컨텍스트: {len(context)}자")
            # 기존 컨텍스트에 추가
            existing_context = state.get("context", "")
            state["context"] = f"{existing_context}\n\n{context}"
        
        return state

# 4. 연결 상태 확인
def connection_check_example():
    """연결 상태 확인 예시"""
    
    def check_connection(state):
        milvus_data = state.get("milvus_data", {})
        
        # 연결 정보 확인
        from common.milvus_helpers import get_milvus_connection_info
        
        connection_info = get_milvus_connection_info(milvus_data)
        
        if connection_info["connected"]:
            print("✅ MilvusDB 연결됨")
        else:
            print("❌ MilvusDB 연결 안됨")
        
        return state

# 5. 실제 Teacher 에이전트 노드에 통합하는 예시
def integrate_with_teacher_node(state):
    """Teacher 노드에 MilvusDB 통합 예시"""
    
    # 기존 노드 로직
    user_query = state.get("user_query", "")
    
    # MilvusDB 컨텍스트 추가
    milvus_data = state.get("milvus_data", {})
    
    if milvus_data.get("connection_status", False):
        # 관련 문서 검색 - 직접 컬렉션 이름 사용
        concept_docs = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name="concept_summary",  # 직접 컬렉션 이름
            query=user_query,
            k=3
        )
        
        problem_docs = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name="problems",  # 직접 컬렉션 이름
            query=user_query,
            k=2
        )
        
        # 컨텍스트 생성
        all_docs = concept_docs + problem_docs
        milvus_context = create_context_from_documents(all_docs, max_length=1500)
        
        if milvus_context:
            # 기존 컨텍스트에 추가
            existing_context = state.get("context", "")
            state["context"] = f"{existing_context}\n\n[MilvusDB 검색 결과]\n{milvus_context}"
    
    # 나머지 기존 로직 계속...
    return state

# 6. 실제 Farmer 에이전트 노드에 통합하는 예시
def integrate_with_farmer_node(state):
    """Farmer 노드에 MilvusDB 통합 예시"""
    
    # 기존 노드 로직
    user_query = state.get("query", "")
    
    # MilvusDB 컨텍스트 추가
    milvus_data = state.get("milvus_data", {})
    
    if milvus_data.get("connection_status", False):
        # 작물 정보 검색 - 직접 컬렉션 이름 사용
        crop_docs = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name="crop_info",  # 직접 컬렉션 이름
            query=user_query,
            k=5
        )
        
        # 컨텍스트 생성
        milvus_context = create_context_from_documents(crop_docs, max_length=2000)
        
        if milvus_context:
            # 기존 컨텍스트에 추가
            existing_context = state.get("context", "")
            state["context"] = f"{existing_context}\n\n[MilvusDB 검색 결과]\n{milvus_context}"
    
    # 나머지 기존 로직 계속...
    return state

# 7. 사용 가능한 컬렉션 목록
def available_collections():
    """사용 가능한 컬렉션 목록"""
    
    # Teacher 관련 컬렉션
    teacher_collections = [
        "concept_summary",    # 개념 요약
        "problems",           # 문제
        "info_exam_chunks"    # 정보처리기사 시험 문제
    ]
    
    # Farmer 관련 컬렉션
    farmer_collections = [
        "agri_disaster_docs",  # 농업 재해 문서
        "crop_grow",           # 작물 재배
        "crop_info",           # 작물 정보
        "market_price_docs"    # 시장 가격 문서
    ]
    
    print("Teacher 컬렉션:")
    for col in teacher_collections:
        print(f"  - {col}")
    
    print("\nFarmer 컬렉션:")
    for col in farmer_collections:
        print(f"  - {col}")
    
    return teacher_collections + farmer_collections

