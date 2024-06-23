from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pytesseract
import cv2
from pytesseract import Output
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#이미지 생성
img = Image.open('./Data/Toy/Invoice_original.png')

I1 = ImageDraw.Draw(img)
myFont = ImageFont.truetype('NanumGothic.ttf', 12)

I1.text((334, 110), "최선호", font=myFont, fill =(0, 0, 0))

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