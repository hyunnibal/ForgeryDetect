import os
import cv2
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
import spacy

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 1. 이미지 불러오기 (cv2 사용)
image_path = './Data/Toy/invoice_scanned.jpg'  # 이미지 경로를 지정합니다.
cv2_image = cv2.imread(image_path)

# cv2 이미지를 PIL 이미지로 변환
pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

# 2. 이미지 설명 생성 (BLIP 모델 사용)
# BLIP 모델과 프로세서 로드
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 이미지 설명 생성
num_descriptions = 5  # 생성할 설명의 개수
descriptions = []

for i in range(num_descriptions):
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7 + 0.1 * i)
    description = processor.decode(out[0], skip_special_tokens=True)
    descriptions.append(description)

print("생성된 설명들:", descriptions)

src_lang = "en"
tgt_lang = "ko"
model_name = f"Helsinki-NLP/opus-mt-tc-big-en-ko"

tokenizer = MarianTokenizer.from_pretrained(model_name)
translator = MarianMTModel.from_pretrained(model_name)

translated_descriptions = []

for description in descriptions:
    inputs = tokenizer(description, return_tensors="pt", padding=True)
    translated = translator.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    translated_descriptions.append(translated_text)

print("번역된 설명들:", translated_descriptions)
# 3. 설명 분석 및 태그 생성 (spaCy 사용)
# NLP 모델 로드
nlp = spacy.load("en_core_web_sm")

# 텍스트 분석
doc = nlp(description)

# 명사 추출하여 태그 생성
tags = [chunk.text for chunk in doc.noun_chunks]

print("생성된 태그:", tags)

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 1. 이미지 불러오기 (cv2 사용)
image_path = './Data/Toy/invoice_scanned.jpg'  # 이미지 경로를 지정합니다.
cv2_image = cv2.imread(image_path)

# cv2 이미지를 PIL 이미지로 변환
pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

# 2. 이미지 설명 생성 (BLIP 모델 사용)
# BLIP 모델과 프로세서 로드
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 이미지 설명 생성
num_descriptions = 5  # 생성할 설명의 개수
descriptions = []

for i in range(num_descriptions):
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7 + 0.1 * i, top_p=0.9, num_return_sequences=1)
    description = processor.decode(out[0], skip_special_tokens=True)
    descriptions.append(description)

print("생성된 설명들:", descriptions)

# 3. 설명을 한국어로 번역 (MarianMT 모델 사용)
src_lang = "en"
tgt_lang = "ko"
model_name = "Helsinki-NLP/opus-mt-tc-big-en-ko"

tokenizer = MarianTokenizer.from_pretrained(model_name)
translator = MarianMTModel.from_pretrained(model_name)

translated_descriptions = []

for description in descriptions:
    inputs = tokenizer(description, return_tensors="pt", padding=True)
    translated = translator.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    translated_descriptions.append(translated_text)

print("번역된 설명들:", translated_descriptions)

# 4. 설명 분석 및 태그 생성 (spaCy 사용)
# NLP 모델 로드
nlp = spacy.load("en_core_web_sm")

# 설명에서 명사 추출 및 태그 생성
all_tags = []

for description in descriptions:
    doc = nlp(description)
    tags = [chunk.text for chunk in doc.noun_chunks]
    all_tags.append(tags)

# 태그를 한국어로 번역
translated_tags = []

for tags in all_tags:
    for tag in tags:
        inputs = tokenizer(tag, return_tensors="pt", padding=True)
        translated = translator.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_tags.append(translated_text)

print("생성된 태그들:", translated_tags)

# 한글로 출력
for i, (desc, tags) in enumerate(zip(translated_descriptions, translated_tags)):
    print(f"설명 {i + 1}: {desc}")
    print(f"태그 {i + 1}: {tags}")

import pytesseract
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# Tesseract 경로 설정 (설치된 경로에 맞게 변경)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이진화
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 커널 설정
    kernel = np.ones((1, 1), np.uint8)

    # 모폴로지 변환을 통한 노이즈 제거
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # 이미지 반전
    inverted_image = cv2.bitwise_not(cleaned_image)

    # 이미지 저장
    preprocessed_image_path = 'preprocessed_image.png'
    cv2.imwrite(preprocessed_image_path, inverted_image)

    return preprocessed_image_path


def extract_text_from_image(image_path):
    # 이미지 전처리
    preprocessed_image_path = preprocess_image(image_path)
    # 이미지에서 텍스트 추출 (한글 지원)
    image = Image.open(preprocessed_image_path)
    text = pytesseract.image_to_string(image, lang='kor')
    return text

def generate_image_caption(image_path):
    # 한글 텍스트 생성을 위한 모델 및 프로세서 로드
    # 적절한 한글 지원 모델을 사용합니다.
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # 이미지 로드 및 전처리
    image = Image.open(image_path)
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    # 캡션 생성
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# 이미지 파일 경로
image_path = './Data/receipt/generated/output_image_1.png'

# 이미지에서 텍스트 추출
extracted_text = extract_text_from_image(image_path)
print("Extracted Text:", extracted_text)

# 이미지 캡션 생성
image_caption = generate_image_caption(image_path)
print("Image Caption:", image_caption)