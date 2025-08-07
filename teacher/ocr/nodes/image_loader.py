# image_loader.py
import os
from PIL import Image
import fitz  # PyMuPDF

def convert_pdf_to_images(pdf_path: str, output_folder: str = "./converted_pages") -> list:
    """
    PDF 파일을 이미지로 변환 (Poppler 없이 PyMuPDF로 대체)
    """
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        output_path = os.path.join(output_folder, f"page_{i + 1}.png")
        pix.save(output_path)
        image_paths.append(output_path)

    doc.close()
    return image_paths

def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path)