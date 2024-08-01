from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pytesseract
import cv2
from pytesseract import Output
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#이미지 생성
original_img = Image.open('./Data/receipt/receipt_orginal.png')
patient_info_path = './Data/patient_info.xlsx'
patient_info = pd.read_excel(patient_info_path)

I_origin = ImageDraw.Draw(original_img)
font = ImageFont.truetype('NanumGothic.ttf', 12)

# 이미지에 텍스트 추가하는 함수
def add_text_to_image(image, text, position, font, color=(0, 0, 0)):
    draw = ImageDraw.Draw(image)
    draw.text(position, text, font=font, fill=color)
    return image

# 각 이름을 이미지에 추가
image_list = []
for index, row in patient_info.iterrows():
    img = original_img.copy()
    img = add_text_to_image(img, f"{row['Full Name']}",(340, 110) , font)
    img = add_text_to_image(img, f"{row['Patient Number']}", (142, 110), font)
    img = add_text_to_image(img, f"{row['Visit Date']}", (498, 110), font)
    img = add_text_to_image(img, f"{row['Consultation Fee']}", (275, 237), font)
    img = add_text_to_image(img, f"{row['Hospitalization Fee']}", (277, 259), font)
    img = add_text_to_image(img, f"{row['Injection Fee']}", (277, 405), font)
    img = add_text_to_image(img, f"{row['Test Fee']}", (277, 468), font)
    img = add_text_to_image(img, f"{row['Total Cost']}", (810,211), font)
    img = add_text_to_image(img, f"{row['Patient Responsibility']}", (810, 265), font)
    image_list.append(img)

# 이미지를 저장
for i, img in enumerate(image_list):
    img.save(f"./Data/receipt/generated/output_image_{i+1}.png")

for i in range(2000):

img_toy = original_img.copy()
img_toy_gen = add_text_to_image(img_toy, patient_info(334, 110), "최선호", font=myFont, fill =(0, 0, 0))

# Display edited image
img.show()

# Save the edited image
img.save("./Data/Toy/Invoice_Toy.png")

#이미지 편집
img2 = Image.open('./Data/Toy/invoice_scanned.jpg')
I2 = img2.crop((345, 95, 386, 113))
I2.save("./Data/Toy/cropped.jpg")
#텍스트 추출
reader = easyocr.Reader(['ko'])
result = reader.readtext("./Data/Toy/cropped.jpg")
#copy and move
name2 = I2.crop((result[0][0][0][0]+12,result[0][0][0][1],result[0][0][0][0]+22,result[0][0][0][0]+10))
name3 = I2.crop((result[0][0][0][0]+24,result[0][0][0][1],result[0][0][0][0]+34,result[0][0][0][0]+10))

I3 = I2
Image.Image.paste(I3, name3, (result[0][0][0][0]+12,result[0][0][0][1]))
Image.Image.paste(I3, name2, (result[0][0][0][0]+24,result[0][0][0][1]))
Image.Image.paste(img2, I3, (345, 95))
img2.save('./Data/Toy/fake.jpg')

######################
image_path = './Data/Toy/invoice_scanned.jpg'
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

custom_config = '--oem 3 --psm 6 -l kor'
data = pytesseract.image_to_data(image_rgb, config=custom_config, output_type=Output.DICT)

# 글자 단위로 결과 출력 및 좌표 시각화
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image_rgb)

for i in range(len(data['text'])):
    if int(data['conf'][i]) > 0:  # 신뢰도가 0보다 큰 경우에만 출력
        x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        text = data['text'][i]
        conf = data['conf'][i]

        # 글자와 좌표 출력
        print(f"글자: {text} - 좌표: ({x}, {y}, {w}, {h}) - 신뢰도: {conf}")

        # 이미지에 경계 상자 추가
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y - 10, text, color='red', fontsize=12, backgroundcolor='yellow')

plt.show()

####################################