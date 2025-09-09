"""
통합된 MilvusDB 사용법 예시
collection_name으로 직접 검색하는 방법
"""

from common.milvus_helpers import search_milvus_documents, get_milvus_retriever, create_context_from_documents

def teacher_usage_example(state):
    """Teacher 에이전트에서 통합된 MilvusDB 사용법"""
    
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("user_query", "")
    
    # 1. 개념 관련 문서 검색
    concept_docs = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name="concept_summary",  # 직접 컬렉션 이름 사용
        query=user_query,
        k=3
    )
    
    # 2. 문제 관련 문서 검색
    problem_docs = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name="problems",  # 직접 컬렉션 이름 사용
        query=user_query,
        k=2
    )
    
    # 3. 컨텍스트 생성
    all_docs = concept_docs + problem_docs
    context = create_context_from_documents(all_docs, max_length=2000)
    
    if context:
        print(f"✅ Teacher MilvusDB 컨텍스트 생성: {len(context)}자")
        state["milvus_context"] = context
    
    return state

def farmer_usage_example(state):
    """Farmer 에이전트에서 통합된 MilvusDB 사용법"""
    
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("query", "")
    
    # 1. 작물 정보 검색
    crop_docs = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name="crop_info",  # 직접 컬렉션 이름 사용
        query=user_query,
        k=5
    )
    
    # 2. 재배 정보 검색
    grow_docs = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name="crop_grow",  # 직접 컬렉션 이름 사용
        query=user_query,
        k=3
    )
    
    # 3. 컨텍스트 생성
    all_docs = crop_docs + grow_docs
    context = create_context_from_documents(all_docs, max_length=2000)
    
    if context:
        print(f"✅ Farmer MilvusDB 컨텍스트 생성: {len(context)}자")
        state["milvus_context"] = context
    
    return state

def retriever_usage_example(state):
    """리트리버 사용법 예시"""
    
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("user_query", "")
    
    # 1. 리트리버 생성
    retriever = get_milvus_retriever(
        milvus_data=milvus_data,
        collection_name="concept_summary",  # 직접 컬렉션 이름 사용
        k=5
    )
    
    if retriever:
        # 2. 리트리버로 검색
        documents = retriever.invoke(user_query)
        print(f"✅ 리트리버로 {len(documents)}개 문서 검색")
        
        # 3. 컨텍스트 생성
        context = create_context_from_documents(documents, max_length=1500)
        if context:
            state["milvus_context"] = context
    
    return state

def mixed_collections_example(state):
    """여러 컬렉션을 섞어서 사용하는 예시"""
    
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("user_query", "")
    
    # Teacher와 Farmer 컬렉션을 모두 사용
    collections_to_search = [
        "concept_summary",      # Teacher 컬렉션
        "problems",             # Teacher 컬렉션
        "crop_info",            # Farmer 컬렉션
        "agri_disaster_docs"    # Farmer 컬렉션
    ]
    
    all_documents = []
    
    for collection_name in collections_to_search:
        docs = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name=collection_name,
            query=user_query,
            k=2
        )
        all_documents.extend(docs)
        print(f"✅ {collection_name}: {len(docs)}개 문서")
    
    # 전체 컨텍스트 생성
    context = create_context_from_documents(all_documents, max_length=3000)
    if context:
        print(f"✅ 통합 컨텍스트 생성: {len(context)}자")
        state["milvus_context"] = context
    
    return state

def dynamic_collection_search(state):
    """동적으로 컬렉션을 선택해서 검색하는 예시"""
    
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("user_query", "")
    
    # 쿼리 내용에 따라 다른 컬렉션 선택
    if "문제" in user_query or "퀴즈" in user_query:
        collection_name = "problems"
    elif "개념" in user_query or "이론" in user_query:
        collection_name = "concept_summary"
    elif "작물" in user_query or "재배" in user_query:
        collection_name = "crop_info"
    elif "재해" in user_query or "피해" in user_query:
        collection_name = "agri_disaster_docs"
    else:
        collection_name = "concept_summary"  # 기본값
    
    print(f"🔍 선택된 컬렉션: {collection_name}")
    
    # 선택된 컬렉션에서 검색
    documents = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name=collection_name,
        query=user_query,
        k=5
    )
    
    if documents:
        context = create_context_from_documents(documents, max_length=2000)
        if context:
            print(f"✅ 동적 검색 컨텍스트 생성: {len(context)}자")
            state["milvus_context"] = context
    
    return state

