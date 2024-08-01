import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 경로 설정
original_data_dir = './Data/receipt/processed/'
forgery_data_dir = './Data/receipt/forgery/'


# 이미지 파일 로드 및 비율 유지하며 크기 조정
def load_images_from_folder(folder, label, target_size=(256, 256)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))

        # 비율 유지하면서 이미지 크기 조정
        img.thumbnail(target_size, Image.LANCZOS)

        # 빈 공간을 채우기 위해 배경을 흰색으로 설정
        background = Image.new('RGB', target_size, (255, 255, 255))
        offset = ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
        background.paste(img, offset)

        img = img_to_array(background)
        images.append(img)
        labels.append(label)
    return images, labels


# 이미지 로드
original_images, original_labels = load_images_from_folder(original_data_dir, 0)
forgery_images, forgery_labels = load_images_from_folder(forgery_data_dir, 1)

# 데이터 결합 및 전처리
images = np.array(original_images + forgery_images)
labels = np.array(original_labels + forgery_labels)

# 데이터셋 나누기
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 데이터 증강
train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
validation_generator = test_datagen.flow(X_test, y_test, batch_size=32)

# 모델 설계
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    validation_data=validation_generator,
    validation_steps=len(X_test) // 32,
    epochs=30
)

# 모델 평가
test_loss, test_acc = model.evaluate(validation_generator)
print('테스트 정확도:', test_acc)

# 성능 측정 (accuracy, loss 그래프 그리기 등)
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training los
