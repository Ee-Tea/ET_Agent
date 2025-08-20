import pdfplumber
import os

def extract_text_from_pdf(pdf_path, output_path):
    """
    PDF 파일에서 텍스트를 추출하여 txt 파일로 저장합니다.
    좌우 2열 구조를 고려하여 텍스트를 처리합니다.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"페이지 {page_num} 처리 중...")
                
                # 페이지의 텍스트 추출
                text = page.extract_text()
                if text:
                    all_text.append(f"\n=== 페이지 {page_num} ===\n")
                    all_text.append(text)
                    all_text.append("\n" + "="*50 + "\n")
                
                # 좌우 2열 구조를 더 정확하게 처리하기 위해
                # 페이지를 좌우로 분할하여 처리
                width = page.width
                height = page.height
                
                # 좌측 영역 (페이지의 왼쪽 절반)
                left_bbox = (0, 0, width/2, height)
                left_page = page.within_bbox(left_bbox)
                left_text = left_page.extract_text()
                
                # 우측 영역 (페이지의 오른쪽 절반)
                right_bbox = (width/2, 0, width, height)
                right_page = page.within_bbox(right_bbox)
                right_text = right_page.extract_text()
                
                if left_text or right_text:
                    all_text.append(f"\n--- 페이지 {page_num} 좌우 분할 ---\n")
                    if left_text:
                        all_text.append("【좌측】\n")
                        all_text.append(left_text)
                        all_text.append("\n")
                    if right_text:
                        all_text.append("【우측】\n")
                        all_text.append(right_text)
                        all_text.append("\n")
                    all_text.append("-" * 50 + "\n")
            
            # 전체 텍스트를 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(''.join(all_text))
            
            print(f"텍스트 추출 완료: {output_path}")
            return True
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return False

def main():
    # PDF 파일 경로
    pdf_file = "1. 2024년3회_정보처리기사필기기출문제_cut.pdf"
    output_file = "extracted_pdf_text.txt"
    
    # 파일 존재 확인
    if not os.path.exists(pdf_file):
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_file}")
        return
    
    # 텍스트 추출 실행
    success = extract_text_from_pdf(pdf_file, output_file)
    
    if success:
        print(f"\n텍스트 추출이 성공적으로 완료되었습니다.")
        print(f"결과 파일: {output_file}")
        
        # 파일 크기 확인
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"생성된 파일 크기: {file_size:,} bytes")
    else:
        print("텍스트 추출에 실패했습니다.")

if __name__ == "__main__":
    main()
