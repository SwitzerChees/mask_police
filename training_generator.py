
from operator import mod
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf
import numpy as np
from shutil import copyfile
import os


def create_model():
    model = Sequential([
        Conv2D(100, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),

        Conv2D(100, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['acc'])
    return model


def train_model(training_path, validation_path):
    model = create_model()

    train_datagen = ImageDataGenerator(rescale=1.0/255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(training_path,
                                                        batch_size=10,
                                                        target_size=(150, 150))

    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    validation_generator = validation_datagen.flow_from_directory(validation_path,
                                                                  batch_size=10,
                                                                  target_size=(150, 150))

    checkpoint = ModelCheckpoint(
        'checkpoints/{epoch:03d}.ckpt', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    model.fit(
        train_generator, epochs=30, validation_data=validation_generator, callbacks=[checkpoint])
    model.save('model/my_model')


def split_dataset(face_path, mask_path, training_path, validation_path):
    face_images = os.listdir(face_path)
    split_face = int(len(face_images) * 0.75)
    for face_image in face_images[:split_face]:
        copyfile(os.path.join(face_path, face_image),
                 os.path.join(training_path, 'face', face_image))
    for face_image in face_images[split_face:]:
        copyfile(os.path.join(face_path, face_image),
                 os.path.join(validation_path, 'face', face_image))

    mask_images = os.listdir(mask_path)
    split_mask = int(len(mask_images) * 0.75)
    for mask_image in mask_images[:split_mask]:
        copyfile(os.path.join(mask_path, mask_image),
                 os.path.join(training_path, 'mask', mask_image))
    for mask_image in mask_images[split_mask:]:
        copyfile(os.path.join(mask_path, mask_image),
                 os.path.join(validation_path, 'mask', mask_image))


def test_model(model_path, test_path):
    model = load_model(model_path)
    cases = []
    for face_image in os.listdir(os.path.join(test_path, 'face')):
        image = tf.keras.utils.load_img(os.path.join(test_path, 'face', face_image), target_size=(150, 150))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.expand_dims(image, 0)
        cases.append({ 'label': 'face', 'image': image})
    for mask_image in os.listdir(os.path.join(test_path, 'mask')):
        image = tf.keras.utils.load_img(os.path.join(test_path, 'mask', mask_image), target_size=(150, 150))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.expand_dims(image, 0)
        cases.append({ 'label': 'mask', 'image': image})
    for case in cases:
        result = model.predict(case['image'])
        prediction = 'mask' if result[0][0] > result[0][1] else 'face'
        print('Label:', case['label'], 'Prediction:', prediction)
    # loss, acc = model.evaluate(test_images, test_labels, verbose=2)


if __name__ == '__main__':
    training_path = 'model_data/training'
    validation_path = 'model_data/validation'
    test_path = 'model_data/test'
    # split_dataset('images/without_mask', 'images/with_mask',
    #               training_path, validation_path)
    # train_model(training_path, validation_path)
    test_model('model/my_model', test_path)
