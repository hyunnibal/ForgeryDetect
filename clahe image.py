import cv2
import numpy as np
import os


def apply_clahe(image):
    # Convert to grayscale and ensure the image is in uint8 format
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    # Convert CLAHE result back to 3 channels (for compatibility with the model)
    clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    return clahe_image


def save_image(image, output_path):
    # Save the image as a PNG file
    cv2.imwrite(output_path, image)


# 예시 이미지 경로 및 출력 경로
input_image_path = './Data/receipt/forgery/forgery_processed_output_image_1.png'  # 입력 이미지 경로
output_image_path = './Data/clahe/clahe.png'  # 저장할 PNG 파일 경로

# 이미지 로드
image = cv2.imread(input_image_path)
if image is not None:
    # CLAHE 적용
    processed_image = apply_clahe(image)

    # 이미지 저장
    save_image(processed_image, output_image_path)
    print(f"Image saved to {output_image_path}")
else:
    print(f"Failed to load image from {input_image_path}")
