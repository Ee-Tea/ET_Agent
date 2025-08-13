#!/usr/bin/env python3
"""
RAG 엔진 테스트 스크립트
"""

import os
from rag_engine import RAGEngine


def test_rag_engine():
    """RAG 엔진 기본 기능 테스트"""
    print("🧠 RAG 엔진 테스트 시작")
    
    try:
        # RAG 엔진 초기화
        data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        rag_engine = RAGEngine(data_folder=data_folder)
        
        print(f"✅ RAG 엔진 초기화 완료")
        print(f"📁 데이터 폴더: {rag_engine.data_folder}")
        
        # PDF 파일 목록 확인
        pdf_files = rag_engine.get_pdf_files()
        print(f"📚 발견된 PDF 파일 수: {len(pdf_files)}")
        
        if pdf_files:
            print("📋 PDF 파일 목록:")
            for i, file_path in enumerate(pdf_files[:5], 1):  # 처음 5개만 표시
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) / 1024
                print(f"  {i}. {filename} ({file_size:.1f} KB)")
            
            if len(pdf_files) > 5:
                print(f"  ... 외 {len(pdf_files)-5}개 파일")
            
            # 벡터 스토어 구축 테스트
            print("\n🔧 벡터 스토어 구축 중...")
            success = rag_engine.build_vectorstore_from_all_pdfs()
            
            if success:
                print("✅ 벡터 스토어 구축 완료")
                
                # 벡터 스토어 정보 확인
                info = rag_engine.get_vectorstore_info()
                print(f"📊 벡터 스토어 정보:")
                print(f"  - 초기화 상태: {info['is_initialized']}")
                print(f"  - 총 파일 수: {info['total_files']}")
                print(f"  - 임베딩 모델: {info['embeddings_model']}")
                
                # 간단한 검색 테스트
                print("\n🔍 간단한 검색 테스트...")
                test_query = "자료구조"
                result = rag_engine.retrieve_documents(
                    query=test_query,
                    subject_area="소프트웨어개발"
                )
                
                if "error" not in result:
                    print(f"✅ 검색 성공!")
                    print(f"  - 쿼리: {result['query']}")
                    print(f"  - 검색된 문서 수: {len(result['documents'])}")
                    print(f"  - 사용된 소스: {result['used_sources'][:3]}")  # 처음 3개만
                    
                    # 컨텍스트 준비 테스트
                    if result['documents']:
                        context = rag_engine.prepare_context(
                            documents=result['documents'][:3],  # 처음 3개 문서만
                            weakness_concepts=["자료구조", "스택", "큐"]
                        )
                        print(f"  - 컨텍스트 길이: {len(context)} 문자")
                        print(f"  - 컨텍스트 미리보기: {context[:200]}...")
                else:
                    print(f"❌ 검색 실패: {result['error']}")
            else:
                print("❌ 벡터 스토어 구축 실패")
        else:
            print("⚠️ PDF 파일이 발견되지 않았습니다.")
            print(f"폴더 경로를 확인해주세요: {data_folder}")
        
    except Exception as e:
        print(f"❌ RAG 엔진 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_rag_engine()
