import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
import io

# ✅ 1. Donut 모델 로드
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model.eval()

# ✅ 2. PDF → 이미지 변환 함수
def convert_pdf_to_image(pdf_path, page_num=0, dpi=200):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = io.BytesIO(pix.tobytes("png"))
    image = Image.open(img_bytes).convert("RGB")
    return image

# ✅ 3. Donut QA 함수
def ask_question_with_donut(image: Image, question: str):
    task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    pixel_values = processor(image, return_tensors="pt").pixel_values
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=512,
            early_stopping=True
        )

    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer.replace(task_prompt, "").strip()

# ✅ 4. 실행 예시
if __name__ == "__main__":
    pdf_file = "3. 2025년2회_정보처리기사필기기출문제.pdf"  # 파일명 수정 가능
    image = convert_pdf_to_image(pdf_file, page_num=0)  # 0 = 첫 페이지

    question = "1번 문제의 정답은 무엇인가요?"
    answer = ask_question_with_donut(image, question)

    print(f"📝 질문: {question}")
    print(f"✅ 정답 예측: {answer}")
