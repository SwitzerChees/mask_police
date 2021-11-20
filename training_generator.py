
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import VGG16, Xception, VGG19, MobileNetV2, InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from PIL import Image
import face_recognition
import os
import shutil

from tensorflow.python.keras.callbacks import EarlyStopping


def create_model(size=125):
    model = VGG16(include_top=False, input_shape=(size, size, 3))
    # model = VGG19(include_top=False, input_shape=(size, size, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(256, activation='relu')(flat1)
    class2 = Dense(128, activation='relu')(class1)
    output = Dense(2, activation='softmax')(class2)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    
    # model = Sequential([
    #     Conv2D(50, (3, 3), activation='relu', input_shape=(size, size, 3)),
    #     MaxPooling2D(2, 2),

    #     Conv2D(50, (3, 3), activation='relu'),
    #     MaxPooling2D(2, 2),

    #     Flatten(),
    #     Dropout(0.3),
    #     Dense(50, activation='relu'),
    #     Dense(2, activation='softmax')
    # ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['acc'])
    return model


def train_model(training_path, validation_path, model_path, size=(125, 125)):
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
                                                        target_size=size)

    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    validation_generator = validation_datagen.flow_from_directory(validation_path,
                                                                  batch_size=10,
                                                                  target_size=size)

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(os.path.join(model_path, 'checkpoints'), exist_ok=True)
    # checkpoint_filepath = os.path.join(model_path, 'checkpoints')
    # checkpoint = ModelCheckpoint(
    #     checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=1, mode="auto")
    model.fit(
        train_generator, epochs=100, validation_data=validation_generator, callbacks=[early_stopping])
    # model.load_weights(checkpoint_filepath)
    model.save('model/my_model')


def split_dataset(face_path, mask_path, training_path, validation_path):
    recreate_dir(training_path)
    recreate_dir(validation_path)
    face_images = os.listdir(face_path)
    split_face = int(len(face_images) * 0.7)
    for face_image in face_images[:split_face]:
        get_face(os.path.join(face_path, face_image),
                 os.path.join(training_path, 'face', face_image))
    for face_image in face_images[split_face:]:
        get_face(os.path.join(face_path, face_image),
                 os.path.join(validation_path, 'face', face_image))

    mask_images = os.listdir(mask_path)
    split_mask = int(len(mask_images) * 0.7)
    for mask_image in mask_images[:split_mask]:
        get_face(os.path.join(mask_path, mask_image),
                 os.path.join(training_path, 'mask', mask_image))
    for mask_image in mask_images[split_mask:]:
        get_face(os.path.join(mask_path, mask_image),
                 os.path.join(validation_path, 'mask', mask_image))


def recreate_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir, ignore_errors=False, onerror=None)
    os.makedirs(os.path.join(dir, 'face'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'mask'), exist_ok=True)


def get_face(orig, target, size=(125, 125)):
    face_image_np = face_recognition.load_image_file(orig)
    faces = face_recognition.face_locations(
        face_image_np, model='hog')
    for (top, right, bottom, left) in faces[:1]:
        face_image_np = face_image_np[top:bottom, left:right]
    if len(faces) == 0:
        return
    face_img = Image.fromarray(face_image_np)
    face_img = face_img.resize(size)
    face_img.save(target)


if __name__ == '__main__':
    training_path = 'model/data/training'
    validation_path = 'model/data/validation'
    model_path = 'model'
    # split_dataset('images/without_mask', 'images/with_mask',
    #               training_path, validation_path)
    train_model(training_path, validation_path, model_path)
