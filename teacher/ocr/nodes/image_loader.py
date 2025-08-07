# image_loader.py
import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

def convert_pdf_to_images(pdf_path: str, output_dir: str = "temp_images"):
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"{Path(pdf_path).stem}_page{i+1}.png")
        img.save(path, "PNG")
        image_paths.append(path)
    return image_paths

def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")

if __name__ == "__main__":
    pdf_file = "example.pdf"  # 여기에 실제 PDF 경로 입력
    image_files = convert_pdf_to_images(pdf_file)
    print("🔹 변환된 이미지 파일들:", image_files)
