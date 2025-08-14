#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 ID를 실제 파일 경로로 매핑하는 유틸리티
"""

import os
import glob
from typing import List, Dict, Optional
from pathlib import Path

class FilePathMapper:
    """파일 ID를 실제 파일 경로로 매핑하는 클래스"""
    
    def __init__(self, base_dirs: Optional[Dict[str, str]] = None):
        """
        Args:
            base_dirs: 디렉토리 타입별 기본 경로 매핑
                      예: {"pdf": "agents/TestGenerator/data", "image": "temp_images"}
        """
        self.base_dirs = base_dirs or {
            "pdf": "agents\\TestGenerator\\data",
            "image": "..\\temp_images",
            "document": "..\\data"
        }
        
        # 현재 스크립트 위치 기준으로 상대 경로를 절대 경로로 변환
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for key, path in self.base_dirs.items():
            if not os.path.isabs(path):
                # teacher 디렉토리에서 실행하므로 상위 디렉토리로 이동
                self.base_dirs[key] = os.path.join(script_dir, "..", path)
                print(f"   - {key}: {self.base_dirs[key]} (존재: {os.path.exists(self.base_dirs[key])})")
                
                # 경로가 존재하지 않으면 절대 경로로 시도
                if not os.path.exists(self.base_dirs[key]):
                    # 프로젝트 루트 디렉토리 찾기
                    project_root = os.path.abspath(os.path.join(script_dir, ".."))
                    alt_path = os.path.join(project_root, path)
                    if os.path.exists(alt_path):
                        self.base_dirs[key] = alt_path
                        print(f"     → 대체 경로 사용: {alt_path}")
                    else:
                        print(f"     → 경로를 찾을 수 없음: {path}")
                        # Windows 경로 구분자 문제일 수 있으므로 os.path.join 사용
                        normalized_path = os.path.join(*path.split('\\'))
                        final_path = os.path.join(project_root, normalized_path)
                        if os.path.exists(final_path):
                            self.base_dirs[key] = final_path
                            print(f"     → 정규화된 경로 사용: {final_path}")
                        else:
                            print(f"     → 모든 경로 시도 실패")
                            # 마지막 시도: 직접 경로 확인
                            direct_path = os.path.join(project_root, "agents", "TestGenerator", "data")
                            if os.path.exists(direct_path):
                                self.base_dirs[key] = direct_path
                                print(f"     → 직접 경로 사용: {direct_path}")
                            else:
                                print(f"     → 모든 경로 시도 실패")
    
    def find_file_by_id(self, file_id: str, file_type: str = "pdf") -> Optional[str]:
        """
        파일 ID로 실제 파일 경로를 찾습니다.
        
        Args:
            file_id: 파일 ID (파일명에 포함된 문자열)
            file_type: 파일 타입 ("pdf", "image", "document")
            
        Returns:
            파일 경로 또는 None (찾지 못한 경우)
        """
        base_dir = self.base_dirs.get(file_type)
        if not base_dir or not os.path.exists(base_dir):
            return None
        
        # 파일 타입별 확장자
        extensions = {
            "pdf": [".pdf"],
            "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
            "document": [".pdf", ".doc", ".docx", ".txt", ".md"]
        }
        
        ext_list = extensions.get(file_type, [".*"])
        
        # 파일 검색
        for ext in ext_list:
            if ext == ".*":
                # 모든 파일 검색
                pattern = os.path.join(base_dir, "*")
                files = glob.glob(pattern)
            else:
                # 특정 확장자만 검색
                pattern = os.path.join(base_dir, f"*{ext}")
                files = glob.glob(pattern)
            
            for file_path in files:
                if os.path.isfile(file_path) and file_id.lower() in os.path.basename(file_path).lower():
                    return file_path
        
        return None
    
    def find_files_by_ids(self, file_ids: List[str], file_type: str = "pdf") -> List[str]:
        """
        여러 파일 ID로 실제 파일 경로들을 찾습니다.
        
        Args:
            file_ids: 파일 ID 리스트
            file_type: 파일 타입
            
        Returns:
            찾은 파일 경로 리스트
        """
        found_files = []
        for file_id in file_ids:
            file_path = self.find_file_by_id(file_id, file_type)
            if file_path:
                found_files.append(file_path)
        
        return found_files
    
    def map_artifacts_to_paths(self, artifacts: Dict) -> List[str]:
        """
        artifacts 딕셔너리에서 파일 경로들을 추출합니다.
        
        Args:
            artifacts: {"pdf_ids": ["id1", "id2"], "image_ids": ["img1"]} 형태
            
        Returns:
            파일 경로 리스트
        """
        all_paths = []
        
        # PDF 파일들
        if "pdf_ids" in artifacts:
            pdf_paths = self.find_files_by_ids(artifacts["pdf_ids"], "pdf")
            all_paths.extend(pdf_paths)
        
        # 이미지 파일들
        if "image_ids" in artifacts:
            image_paths = self.find_files_by_ids(artifacts["image_ids"], "image")
            all_paths.extend(image_paths)
        
        # 문서 파일들
        if "document_ids" in artifacts:
            doc_paths = self.find_files_by_ids(artifacts["document_ids"], "document")
            all_paths.extend(doc_paths)
        
        return all_paths
    
    def list_available_files(self, file_type: str = "pdf") -> List[str]:
        """
        특정 타입의 사용 가능한 파일들을 나열합니다.
        
        Args:
            file_type: 파일 타입
            
        Returns:
            파일명 리스트
        """
        base_dir = self.base_dirs.get(file_type)
        if not base_dir or not os.path.exists(base_dir):
            return []
        
        extensions = {
            "pdf": [".pdf"],
            "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
            "document": [".pdf", ".doc", ".docx", ".txt", ".md"]
        }
        
        ext_list = extensions.get(file_type, [".*"])
        files = []
        
        for ext in ext_list:
            if ext == ".*":
                pattern = os.path.join(base_dir, "*")
                files.extend(glob.glob(pattern))
            else:
                pattern = os.path.join(base_dir, f"*{ext}")
                files.extend(glob.glob(pattern))
        
        # 파일만 필터링하고 정렬
        files = [f for f in files if os.path.isfile(f)]
        files.sort()
        
        return files

def create_default_mapper() -> FilePathMapper:
    """기본 설정으로 FilePathMapper를 생성합니다."""
    return FilePathMapper()

if __name__ == "__main__":
    # 테스트
    mapper = create_default_mapper()
    
    print("🔍 사용 가능한 파일들:")
    
    # PDF 파일들
    pdf_files = mapper.list_available_files("pdf")
    print(f"\n📚 PDF 파일들 ({len(pdf_files)}개):")
    for f in pdf_files[:5]:  # 처음 5개만 표시
        print(f"   - {os.path.basename(f)}")
    if len(pdf_files) > 5:
        print(f"   ... 및 {len(pdf_files) - 5}개 더")
    
    # 이미지 파일들
    image_files = mapper.list_available_files("image")
    print(f"\n🖼️ 이미지 파일들 ({len(image_files)}개):")
    for f in image_files[:5]:
        print(f"   - {os.path.basename(f)}")
    if len(image_files) > 5:
        print(f"   ... 및 {len(image_files) - 5}개 더")
    
    # 파일 ID로 검색 테스트
    print(f"\n🔍 파일 ID 검색 테스트:")
    test_ids = ["2024년3회", "정보처리기사"]
    for test_id in test_ids:
        found = mapper.find_file_by_id(test_id, "pdf")
        if found:
            print(f"   ✅ '{test_id}' → {os.path.basename(found)}")
        else:
            print(f"   ❌ '{test_id}' → 찾을 수 없음")
