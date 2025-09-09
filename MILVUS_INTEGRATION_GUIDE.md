# MilvusDB 하위 노드 사용 가이드

## 개요

이 가이드는 각 에이전트(Teacher, Farmer)의 하위 노드에서 MilvusDB를 사용하는 방법을 설명합니다.

## 기본 사용법

### 1. 필수 import

```python
from common.milvus_helpers import search_milvus_documents, search_milvus_documents_by_subject, get_milvus_retriever, create_context_from_documents
```

### 2. 기본 검색 (LangGraph 노드에서)

```python
def your_node(state):
    # MilvusDB 연결 정보 가져오기
    milvus_data = state.get("milvus_data", {})
    
    # 일반 유사도 검색
    documents = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name="concepts",
        query="데이터베이스 정규화",
        k=5
    )
    
    # 컨텍스트 생성
    if documents:
        context = create_context_from_documents(documents, max_length=2000)
        # 기존 컨텍스트에 추가
        existing_context = state.get("context", "")
        state["context"] = f"{existing_context}\n\n[MilvusDB 검색 결과]\n{context}"
    
    return state
```

### 2-1. Teacher용 과목명 검색 (권장)

```python
def teacher_node(state):
    # MilvusDB 연결 정보 가져오기
    milvus_data = state.get("milvus_data", {})
    subject_area = state.get("subject_area", "")
    
    # 과목명으로 직접 검색 (더 정확함)
    concept_docs = search_milvus_documents_by_subject(
        milvus_data=milvus_data,
        collection_name="concepts",
        subject_area=subject_area,  # 과목명으로 필터링
        k=10
    )
    
    problem_docs = search_milvus_documents_by_subject(
        milvus_data=milvus_data,
        collection_name="problems",
        subject_area=subject_area,  # 과목명으로 필터링
        k=5
    )
    
    # 문서 합치기
    all_docs = concept_docs + problem_docs
    
    # 컨텍스트 생성
    if all_docs:
        context = create_context_from_documents(all_docs, max_length=2000)
        state["context"] = context
        print(f"✅ {subject_area} 과목 검색 완료: {len(all_docs)}개 문서")
    
    return state
```

### 3. Generator에서의 사용법 (직렬화 문제 해결)

Generator에서는 LangGraph state에 직렬화 불가능한 객체를 넣을 수 없으므로, 전역 변수를 사용합니다:

```python
class TestGenerator:
    def __init__(self):
        self._current_milvus_data = None  # MilvusDB 연결 정보 저장용
    
    def _generate_subject_quiz(self, subject_area: str, target_count: int = 5, 
                              difficulty: str = "중급", milvus_data: Dict[str, Any] = None):
        # MilvusDB 연결 정보를 전역 변수로 저장
        self._current_milvus_data = milvus_data
        
        # 나머지 로직...
    
    def _retrieve_documents(self, state: GraphState) -> GraphState:
        # 전역 milvus_data 사용
        if hasattr(self, '_current_milvus_data') and self._current_milvus_data:
            documents = search_milvus_documents(
                milvus_data=self._current_milvus_data,
                collection_name="concepts",
                query=enhanced_query,
                k=20
            )
        # 나머지 로직...
```

## 최신 기능

### 1. 동적 스키마 추론 및 메트릭 타입 자동 감지

MilvusDB 컬렉션의 스키마를 자동으로 감지하여 올바른 필드명과 메트릭 타입을 사용합니다:

```python
# 자동 스키마 감지 사용
vectorstore = milvus_manager.get_milvus_vectorstore("concepts")
# 자동으로 올바른 필드명과 메트릭 타입 사용

# 수동으로 스키마 확인
text_field, vector_field = milvus_manager._infer_collection_fields("concepts")
metric_type = milvus_manager._get_collection_metric_type("concepts")
print(f"텍스트 필드: {text_field}, 벡터 필드: {vector_field}, 메트릭: {metric_type}")
```

**자동 감지 우선순위:**
- **텍스트 필드**: `content` → `item_title` → `title` 순
- **벡터 필드**: `embedding` → `vector` 순  
- **메트릭 타입**: 컬렉션 인덱스에서 자동 감지 (COSINE, L2 등)

### 2. 과목명 별칭 매핑

내부 과목명과 MilvusDB의 실제 과목명을 자동으로 매핑합니다:

```python
# 자동 과목명 매핑
concept_docs = search_milvus_documents_by_subject(
    milvus_data=milvus_data,
    collection_name="concepts",
    subject_area="소프트웨어개발",  # 내부명
    k=10
)
# 자동으로 "소프트웨어 개발"로 변환되어 검색
```

## 사용 가능한 컬렉션

### Teacher 관련 컬렉션
- `concepts` - 개념 요약 (기존 `concept_summary`에서 변경)
- `problems` - 문제 데이터

### Farmer 관련 컬렉션
- `crop_info` - 작물 정보
- `crop_grow` - 작물 재배 정보
- `agri_disaster_docs` - 농업 재해 문서
- `market_price_docs` - 시장 가격 문서

## 실제 사용 예시

### 1. Teacher 노드에서 사용

```python
def teacher_generate_questions(state):
    """문제 생성 노드에서 MilvusDB 사용"""
    
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("user_query", "")
    
    # 1. 개념 관련 문서 검색
    concept_docs = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name="concepts",
        query=user_query,
        k=3
    )
    
    # 2. 문제 관련 문서 검색
    problem_docs = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name="problems",
        query=user_query,
        k=2
    )
    
    # 3. 컨텍스트 생성
    all_docs = concept_docs + problem_docs
    context = create_context_from_documents(all_docs, max_length=2000)
    
    if context:
        # 기존 컨텍스트에 추가
        existing_context = state.get("context", "")
        state["context"] = f"{existing_context}\n\n[MilvusDB 검색 결과]\n{context}"
        print(f"✅ Teacher MilvusDB 컨텍스트 추가: {len(context)}자")
    
    return state

def teacher_solution_agent(state):
    """해답 생성 노드에서 MilvusDB 사용"""
    
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("user_query", "")
    
    # 해답 생성용 컨텍스트 검색
    documents = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name="concepts",
        query=user_query,
        k=5
    )
    
    if documents:
        context = create_context_from_documents(documents, max_length=1500)
        if context:
            state["milvus_context"] = context
            print(f"✅ SolutionAgent MilvusDB 컨텍스트: {len(context)}자")
    
    return state
```

### 2. Generator에서의 실제 사용 (TestGenerator)

```python
class TestGenerator:
    def __init__(self):
        self._current_milvus_data = None  # MilvusDB 연결 정보 저장용
    
    def _generate_subject_quiz(self, subject_area: str, target_count: int = 5, 
                              difficulty: str = "중급", milvus_data: Dict[str, Any] = None):
        """과목별 문제 생성 - MilvusDB 연결 정보를 전역 변수로 저장"""
        # MilvusDB 연결 정보를 전역 변수로 저장
        self._current_milvus_data = milvus_data
        
        if not milvus_data or not milvus_data.get("connection_status", False):
            print("⚠️ MilvusDB 연결 안됨 - 컨텍스트 없이 문제 생성")
        
        # 나머지 로직...
        result = self.workflow.invoke(initial_state)
        return result
    
    def _retrieve_documents(self, state: GraphState) -> GraphState:
        """문서 검색 - 전역 milvus_data 사용"""
        query = state["query"]
        subject_area = state.get("subject_area", "")
        enhanced_query = f"{subject_area} {query}".strip()
        
        documents = []
        
        # 전역 milvus_data 사용
        if hasattr(self, '_current_milvus_data') and self._current_milvus_data:
            print("🔍 MilvusDB에서 문서 검색 중...")
            
            # 과목명으로 개념 관련 문서 검색
            concept_docs = search_milvus_documents_by_subject(
                milvus_data=self._current_milvus_data,
                collection_name="concepts",
                subject_area=subject_area,
                k=20
            )
            
            # 과목명으로 문제 관련 문서 검색
            problem_docs = search_milvus_documents_by_subject(
                milvus_data=self._current_milvus_data,
                collection_name="problems",
                subject_area=subject_area,
                k=30
            )
            
            # 문서 합치기
            documents = concept_docs + problem_docs
            
            if documents:
                print(f"✅ MilvusDB 검색 완료: {len(documents)}개 문서")
            else:
                print("⚠️ MilvusDB에서 관련 문서를 찾지 못함")
        else:
            print("⚠️ MilvusDB 연결 안됨 - 빈 문서로 진행")
        
        return {**state, "documents": documents}
```

### 3. Farmer 노드에서 사용

```python
def farmer_crop_recommendation(state):
    """작물 추천 노드에서 MilvusDB 사용"""
    
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("query", "")
    
    # 작물 정보 검색
    crop_docs = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name="crop_info",
        query=user_query,
        k=5
    )
    
    # 재배 정보 검색
    grow_docs = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name="crop_grow",
        query=user_query,
        k=3
    )
    
    # 컨텍스트 생성
    all_docs = crop_docs + grow_docs
    context = create_context_from_documents(all_docs, max_length=2000)
    
    if context:
        state["milvus_context"] = context
        print(f"✅ 작물 추천 MilvusDB 컨텍스트: {len(context)}자")
    
    return state

def farmer_disaster_response(state):
    """재해 대응 노드에서 MilvusDB 사용"""
    
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("query", "")
    
    # 재해 관련 문서 검색
    disaster_docs = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name="agri_disaster_docs",
        query=user_query,
        k=5
    )
    
    if disaster_docs:
        context = create_context_from_documents(disaster_docs, max_length=1500)
        if context:
            state["milvus_context"] = context
            print(f"✅ 재해 대응 MilvusDB 컨텍스트: {len(context)}자")
    
    return state
```

## 리트리버 사용법

### 1. 기본 리트리버 사용

```python
def your_node_with_retriever(state):
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("user_query", "")
    
    # 리트리버 생성
    retriever = get_milvus_retriever(
        milvus_data=milvus_data,
        collection_name="concepts",
        k=5
    )
    
    if retriever:
        # 리트리버로 검색
        documents = retriever.invoke(user_query)
        
        # 컨텍스트 생성
        context = create_context_from_documents(documents, max_length=2000)
        if context:
            state["context"] = f"{state.get('context', '')}\n\n{context}"
    
    return state
```

### 2. 여러 컬렉션 동시 검색

```python
def multi_collection_search(state):
    """여러 컬렉션을 동시에 검색"""
    
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("user_query", "")
    
    # 검색할 컬렉션 목록
    collections = [
        "concepts",
        "problems", 
        "crop_info",
        "agri_disaster_docs"
    ]
    
    all_documents = []
    
    for collection_name in collections:
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
        state["milvus_context"] = context
        print(f"✅ 통합 컨텍스트 생성: {len(context)}자")
    
    return state
```

## 동적 컬렉션 선택

```python
def dynamic_collection_search(state):
    """쿼리 내용에 따라 동적으로 컬렉션 선택"""
    
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("user_query", "")
    
    # 쿼리 내용에 따라 컬렉션 선택
    if "문제" in user_query or "퀴즈" in user_query:
        collection_name = "problems"
    elif "개념" in user_query or "이론" in user_query:
        collection_name = "concepts"
    elif "작물" in user_query or "재배" in user_query:
        collection_name = "crop_info"
    elif "재해" in user_query or "피해" in user_query:
        collection_name = "agri_disaster_docs"
    else:
        collection_name = "concepts"  # 기본값
    
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
            state["milvus_context"] = context
            print(f"✅ 동적 검색 컨텍스트 생성: {len(context)}자")
    
    return state
```

## 연결 상태 확인

```python
def check_milvus_connection(state):
    """MilvusDB 연결 상태 확인"""
    
    milvus_data = state.get("milvus_data", {})
    
    if milvus_data.get("connection_status", False):
        print("✅ MilvusDB 연결됨")
        
        # 간단한 검색 테스트
        test_docs = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name="concepts",
            query="테스트",
            k=1
        )
        
        if test_docs:
            print("✅ MilvusDB 검색 정상 작동")
        else:
            print("⚠️ MilvusDB 검색 결과 없음")
    else:
        print("❌ MilvusDB 연결 안됨")
    
    return state
```

## 에러 처리

```python
def safe_milvus_search(state):
    """안전한 MilvusDB 검색 (에러 처리 포함)"""
    
    milvus_data = state.get("milvus_data", {})
    user_query = state.get("user_query", "")
    
    try:
        # 연결 상태 확인
        if not milvus_data.get("connection_status", False):
            print("⚠️ MilvusDB 연결 안됨 - 기본 방식으로 진행")
            return state
        
        # 문서 검색
        documents = search_milvus_documents(
            milvus_data=milvus_data,
            collection_name="concepts",
            query=user_query,
            k=5
        )
        
        if documents:
            context = create_context_from_documents(documents, max_length=2000)
            if context:
                state["milvus_context"] = context
                print(f"✅ MilvusDB 검색 성공: {len(context)}자")
        else:
            print("⚠️ MilvusDB에서 관련 문서를 찾지 못함")
    
    except Exception as e:
        print(f"❌ MilvusDB 검색 실패: {e}")
        # 에러 발생 시에도 노드는 계속 진행
    
    return state
```

## 주의사항 및 문제 해결

### 1. 직렬화 문제 해결

**문제**: `Type is not msgpack serializable: MilvusDBManager`

**원인**: LangGraph state에 직렬화 불가능한 객체를 넣었을 때 발생

**해결방법**:
```python
# ❌ 잘못된 방법 - state에 직렬화 불가능한 객체 넣기
state["milvus_data"] = {"milvus_manager": MilvusDBManager()}  # 에러!

# ✅ 올바른 방법 - 전역 변수 사용
class YourAgent:
    def __init__(self):
        self._current_milvus_data = None
    
    def your_function(self, milvus_data):
        self._current_milvus_data = milvus_data  # 전역 변수에 저장
        # state에는 넣지 않음
```

### 2. 컬렉션 이름 변경

**변경사항**: `concept_summary` → `concepts`

```python
# ❌ 기존 (더 이상 사용하지 않음)
collection_name="concept_summary"

# ✅ 새로운 방식
collection_name="concepts"
```

### 3. 동적 스키마 감지 사용

**자동 필드 감지**를 사용하여 하드코딩을 피하세요:

```python
# ❌ 하드코딩된 필드명 사용
vectorstore = Milvus(
    collection_name="concepts",
    text_field="content",  # 하드코딩 - 다른 컬렉션에서 오류 가능
    vector_field="embedding"
)

# ✅ 자동 스키마 감지 사용
vectorstore = milvus_manager.get_milvus_vectorstore("concepts")
# 자동으로 올바른 필드명과 메트릭 타입 사용
```

### 4. 연결 상태 확인

```python
# 항상 연결 상태를 확인하세요
if not milvus_data.get("connection_status", False):
    print("⚠️ MilvusDB 연결 안됨 - 기본 방식으로 진행")
    return state
```

### 5. 에러 처리

```python
try:
    documents = search_milvus_documents(
        milvus_data=milvus_data,
        collection_name="concepts",
        query=query,
        k=5
    )
except Exception as e:
    print(f"❌ MilvusDB 검색 실패: {e}")
    documents = []  # 빈 문서로 폴백
```

## 컨텍스트 생성 옵션

### 1. 기본 컨텍스트 생성

```python
# 기본 길이 제한 (2000자)
context = create_context_from_documents(documents, max_length=2000)
```

### 2. 상세한 컨텍스트 생성

```python
def create_detailed_context(documents, max_length=2000):
    """더 상세한 컨텍스트 생성"""
    if not documents:
        return ""
    
    context_parts = []
    current_length = 0
    
    for i, doc in enumerate(documents, 1):
        # 메타데이터에서 정보 추출
        title = doc.metadata.get("title", f"문서 {i}")
        source = doc.metadata.get("source", "")
        
        # 헤더 생성
        header = f"=== {title} ==="
        if source:
            header += f" (출처: {source})"
        
        # 내용
        content = doc.page_content
        
        # 전체 텍스트
        full_text = f"{header}\n{content}"
        
        # 길이 확인
        if current_length + len(full_text) <= max_length:
            context_parts.append(full_text)
            current_length += len(full_text)
        else:
            remaining = max_length - current_length
            if remaining > 100:
                context_parts.append(full_text[:remaining] + "...")
            break
    
    return "\n\n".join(context_parts)

# 사용
context = create_detailed_context(documents, max_length=2000)
```

## 주의사항

1. **연결 상태 확인**: 항상 `milvus_data.get("connection_status", False)`로 연결 상태를 확인하세요.

2. **에러 처리**: MilvusDB 검색 실패 시에도 노드가 계속 진행되도록 try-catch를 사용하세요.

3. **컨텍스트 길이**: 너무 긴 컨텍스트는 LLM 성능을 저하시킬 수 있으므로 적절한 길이로 제한하세요.

4. **메모리 사용량**: 많은 문서를 검색할 때는 메모리 사용량을 고려하세요.

5. **컬렉션 이름**: 정확한 컬렉션 이름을 사용하세요. 존재하지 않는 컬렉션은 빈 결과를 반환합니다.

## 요약

### 기본 사용법
- **기본 사용**: `search_milvus_documents()` + `create_context_from_documents()`
- **리트리버 사용**: `get_milvus_retriever()` + `retriever.invoke()`
- **연결 확인**: `milvus_data.get("connection_status", False)`
- **에러 처리**: try-catch로 안전하게 처리
- **컨텍스트 길이**: 적절한 `max_length` 설정

### 최신 기능 (2024.12)
- **✅ 동적 스키마 추론**: 컬렉션별 자동 필드 감지
- **✅ 메트릭 타입 자동 감지**: COSINE, L2 등 자동 감지
- **✅ 과목명 별칭 매핑**: 내부명과 DB명 자동 변환
- **✅ 컬렉션 이름 변경**: `concept_summary` → `concepts`
- **✅ 재시도 로직**: 연결 실패 시 자동 재시도

### 핵심 개선사항
1. **자동화**: 스키마와 메트릭 타입 자동 감지
2. **안정성**: 연결 재시도 및 에러 처리 강화
3. **유지보수성**: 하드코딩 제거로 유연성 향상
4. **호환성**: 다양한 컬렉션 구조 지원