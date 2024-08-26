import cv2
import pytesseract
from transformers import pipeline
import re
import matplotlib.pyplot as plt
import os


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def extract_text_from_image(image_path):
    processed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(processed_image, lang='kor')
    cleaned_text = re.sub(r'[^가-힣0-9\s]', '', text)
    return cleaned_text


def translate_text(text, model_name):
    translator = pipeline("translation", model=model_name, tokenizer=model_name)
    translation = translator(text, max_length=512)[0]['translation_text']
    return translation


def summarize_text(text, model_name):
    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary


def process_image_for_summary(image_path):
    extracted_text = extract_text_from_image(image_path)

    # 한글 텍스트를 영어로 번역
    english_text = translate_text(extracted_text, "Helsinki-NLP/opus-mt-ko-en")

    # 영어 텍스트 요약
    summarized_text = summarize_text(english_text, "facebook/bart-large-cnn")

    # 요약된 영어 텍스트를 다시 한글로 번역
    korean_summary = translate_text(summarized_text, "Helsinki-NLP/opus-mt-tc-big-en-ko")

    return english_text, summarized_text, korean_summary



image_path = 'C:/Users/user/PycharmProjects/ForgeryDetect/Data/receipt/forgery/forgery_processed_output_image_1.png'

# 요약 결과 출력
summary_result = process_image_for_summary(image_path)
print(summary_result)
