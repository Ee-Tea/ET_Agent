# structure_parser.py
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from image_loader import convert_pdf_to_images, load_image
from openai import OpenAI
import numpy as np
import base64
import io
import easyocr
import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQAI_API_KEY")

# ✅ 모델과 프로세서 로드 (Donut은 그림이 있을 때만 사용)
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-docvqa"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())

# client = OpenAI(api_key=api_key)
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)

def ocr_extract_text(image: Image.Image) -> str:
    results = reader.readtext(np.array(image), detail=0)
    return "\n".join(results)

def extract_diagram_with_gpt(image: Image.Image) -> dict:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    prompt = """
    다음 이미지는 시험 문제에 포함된 그림입니다.
    그림의 구조를 다음 형식으로 분석해주세요 (가능한 한 자세히):
    {
        "diagram_type": "",
        "diagram_structure": "",
        "diagram_description": ""
    }
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "이미지를 기반으로 구조화된 분석을 도와주는 어시스턴트입니다."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}
        ],
        max_tokens=800
    )

    return eval(response.choices[0].message.content)

def extract_structure_json(image: Image.Image, has_diagram: bool) -> dict:
    ocr_text = ocr_extract_text(image)

    if has_diagram:
        diagram_info = extract_diagram_with_gpt(image)
    else:
        diagram_info = {"diagram_type": "", "diagram_structure": "", "diagram_description": ""}

    # ✨ 구조 정형화
    return {
        "문제": ocr_text,
        "그림": {
            "타입": diagram_info.get("diagram_type", ""),
            "내용": "",
            "구조": diagram_info.get("diagram_structure", ""),
            "설명": diagram_info.get("diagram_description", "")
        },
        "보기": []
    }

def detect_diagram(image: Image.Image) -> bool:
    # 간단한 기준: 이미지 면적 대비 채워진 영역 비율
    gray = image.convert("L")
    bw = gray.point(lambda x: 0 if x < 200 else 255, '1')
    non_white = bw.histogram()[0]
    ratio = non_white / (image.width * image.height)
    return ratio > 0.05  # 5% 이상이면 그림 있다고 판단

def parse_structure_from_pdf(pdf_path: str) -> list:
    image_paths = convert_pdf_to_images(pdf_path)
    results = []
    for img_path in image_paths:
        image = load_image(img_path)
        has_diagram = detect_diagram(image)
        structure = extract_structure_json(image, has_diagram)
        results.append({
            "page": img_path,
            "structure": structure
        })
    return results

def send_to_answer_agent(structured_qas: list):
    for qa in structured_qas:
        payload = {
            "question": qa["structure"]["문제"],
            "diagram": qa["structure"]["그림"],
            "choices": qa["structure"]["보기"]
        }
        print("\n🚀 에이전트로 전달할 데이터:", payload)
        # answer = answer_agent(payload)
        # print("답변:", answer)

if __name__ == "__main__":
    pdf_file = "example.pdf"  # 여기에 실제 PDF 경로 입력
    parsed = parse_structure_from_pdf(pdf_file)

    for page in parsed:
        print(f"\n📄 {page['page']} 구조:")
        print(page['structure'])

    send_to_answer_agent(parsed)
