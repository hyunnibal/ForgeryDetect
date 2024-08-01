import os
import glob
import cv2
import random

# copy_point와 paste_point 좌표 리스트
paste_point = [(817,214), (824, 214), (831, 214)]
copy_point = [(810,268), (817,268), (824, 268), (831, 268)]

def copymove(image, copy, paste):
    # 크롭할 영역의 크기 (height, width)
    crop_height, crop_width = 12, 7

    # 이미지의 복사할 부분을 크롭
    cropped_part = image[copy[1]:copy[1]+crop_height, copy[0]:copy[0]+crop_width]

    # 크롭된 부분을 붙여넣을 위치 지정
    paste_y, paste_x = paste[1], paste[0]
    end_y = paste_y + crop_height
    end_x = paste_x + crop_width

    # 이미지의 붙여넣을 위치에 크롭된 부분 복사
    image[paste_y:end_y, paste_x:end_x] = cropped_part

    return image

# 문서 후처리
input_folder_path = './Data/receipt/processed/'
output_folder_path = './Data/receipt/forgery/'

# 결과 저장 폴더가 존재하지 않으면 생성
os.makedirs(output_folder_path, exist_ok=True)

# 폴더 내 모든 이미지 파일 경로 불러오기
image_paths = glob.glob(os.path.join(input_folder_path, '*.png'))

# 각 이미지에 대해 함수 적용
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is not None:
        final_image = copymove(image, random.choice(copy_point), random.choice(paste_point))

        # 결과 이미지 저장
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder_path, f'forgery_{base_name}')
        cv2.imwrite(output_path, final_image)

print("모든 이미지 처리가 완료되었습니다.")
