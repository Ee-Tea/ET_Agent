# image_loader.py
import os
from typing import List
from PIL import Image
import pymupdf as fitz  # PyMuPDF

def convert_pdf_to_images(pdf_path: str,
                          output_dir: str = "./converted_pages",
                          dpi: int = 300) -> List[str]:
    """
    PyMuPDF로 PDF 각 페이지를 PNG로 저장해서 경로 리스트 반환.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths: List[str] = []
    doc = None
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            img_path = os.path.join(output_dir, f"page_{i+1:03}.png")
            pix.save(img_path)
            image_paths.append(img_path)
    finally:
        if doc is not None:
            doc.close()
    return image_paths

def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")
