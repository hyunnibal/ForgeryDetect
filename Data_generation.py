from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pandas as pd
import cv2
import numpy as np

# Load the original image and patient info
original_img = Image.open('./Data/receipt/receipt_orginal.png')
patient_info_path = './Data/patient_info.xlsx'
patient_info = pd.read_excel(patient_info_path)

font = ImageFont.truetype('NanumGothic.ttf', 12)


# Function to add text to an image
def add_text_to_image(image, text, position, font, color=(0, 0, 0)):
    draw = ImageDraw.Draw(image)
    draw.text(position, text, font=font, fill=color)
    return image


def add_lighting(image, light_strength=0.5, light_radius_factor=0.5):
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rows, cols, ch = image_cv.shape
    center_x, center_y = int(cols / 2), int(rows / 2)

    # Create a lighting mask
    mask = np.zeros((rows, cols), np.uint8)
    light_radius = int(min(center_x, center_y) * light_radius_factor)
    cv2.circle(mask, (center_x, center_y), light_radius, (255), -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 11)

    # Mix the original image and grayscale image
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.merge([gray_image, gray_image, gray_image])

    # Adjust lighting effect
    lighting = cv2.addWeighted(image_cv, 1 - light_strength, gray_image, light_strength, 0)

    # Apply the lighting mask
    mask = mask.astype(float) / 255
    mask = cv2.merge([mask, mask, mask])
    lighting = image_cv * (1 - mask) + lighting * mask
    lighting = np.clip(lighting, 0, 255).astype(np.uint8)

    # Convert back to PIL format
    lighting = cv2.cvtColor(lighting, cv2.COLOR_BGR2RGB)
    return Image.fromarray(lighting)


def add_gaussian_noise(image, mean=0, sigma=25):
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv = image_cv.astype(np.float32)

    # Create Gaussian noise
    noise = np.random.normal(mean, sigma, image_cv.shape).astype(np.float32)

    # Add noise to the image
    noisy_image = cv2.addWeighted(image_cv, 1.0, noise, 1.0, 0)

    # Clip the values to [0, 255] and convert to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    # Convert back to PIL format
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(noisy_image)


# Add text and effects to each image
image_list = []
for index, row in patient_info.iterrows():
    img = original_img.copy()
    img = add_text_to_image(img, f"{row['Full Name']}", (340, 110), font)
    img = add_text_to_image(img, f"{row['Patient Number']}", (142, 110), font)
    img = add_text_to_image(img, f"{row['Visit Date']}", (498, 110), font)
    img = add_text_to_image(img, f"{row['Consultation Fee']}", (275, 237), font)
    img = add_text_to_image(img, f"{row['Hospitalization Fee']}", (277, 259), font)
    img = add_text_to_image(img, f"{row['Injection Fee']}", (277, 405), font)
    img = add_text_to_image(img, f"{row['Test Fee']}", (277, 468), font)
    img = add_text_to_image(img, f"{row['Total Cost']}", (810, 211), font)
    img = add_text_to_image(img, f"{row['Patient Responsibility']}", (810, 265), font)
    image_list.append(img)

# Save the images
for i, img in enumerate(image_list):
    img.save(f"./Data/receipt/generated/output_image_{i + 1}.png")
