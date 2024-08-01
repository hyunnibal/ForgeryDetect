import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 경로 설정
original_data_dir = './Data/receipt/processed/'
forgery_data_dir = './Data/receipt/forgery/'

# ImageDataGenerator 설정
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 학습 데이터 제너레이터
train_generator = datagen.flow_from_directory(
    './Data/receipt',
    target_size=(992, 1403),  # 원본 이미지 크기로 설정
    batch_size=10,
    class_mode='binary',
    subset='training'
)

# 검증 데이터 제너레이터
validation_generator = datagen.flow_from_directory(
    './Data/receipt',
    target_size=(992, 1403),  # 원본 이미지 크기로 설정
    batch_size=10,
    class_mode='binary',
    subset='validation'
)

# 모델 설계
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(992, 1403, 3)),
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
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=30
)

# 모델 평가
test_loss, test_acc = model.evaluate(validation_generator)
print('테스트 정확도:', test_acc)

# 성능 측정 (accuracy, loss 그래프 그리기 등)
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

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
