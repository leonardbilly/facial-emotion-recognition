from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Dropout, Flatten
from keras.optimizers import Adam


test_data_path = 'dataset/train/'
validation_data_path = 'dataset/validation/'

# Image preprocessing
test_data_generator = ImageDataGenerator(rescale=1.0/255.0,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         rotation_range=20,
                                         horizontal_flip=True
                                         )
validation_data_generator = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_data_generator.flow_from_directory(
    test_data_path,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_data_generator.flow_from_directory(
    validation_data_path,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
)

# FER CNN
facial_emotion_recognition_model = Sequential()
input_shape = (48, 48, 1)

# Layer 1
facial_emotion_recognition_model.add(Conv2D(32, kernel_size=(3, 3),
                                            activation='relu', input_shape=input_shape))
facial_emotion_recognition_model.add(BatchNormalization())
facial_emotion_recognition_model.add(
    Conv2D(64, kernel_size=(3, 3), activation='relu'))
facial_emotion_recognition_model.add(BatchNormalization())
facial_emotion_recognition_model.add(MaxPool2D(pool_size=(2, 2)))
facial_emotion_recognition_model.add(Dropout(0.25))

# Layer 2
facial_emotion_recognition_model.add(Conv2D(128, kernel_size=(3, 3),
                                            activation='relu'))
facial_emotion_recognition_model.add(BatchNormalization())
facial_emotion_recognition_model.add(MaxPool2D(pool_size=(2, 2)))
facial_emotion_recognition_model.add(
    Conv2D(128, kernel_size=(3, 3), activation='relu'))
facial_emotion_recognition_model.add(BatchNormalization())
facial_emotion_recognition_model.add(MaxPool2D(pool_size=(2, 2)))
facial_emotion_recognition_model.add(Dropout(0.25))

# Last layer
facial_emotion_recognition_model.add(Flatten())
facial_emotion_recognition_model.add(Dense(1024, activation='relu'))
facial_emotion_recognition_model.add(BatchNormalization())
facial_emotion_recognition_model.add(Dropout(0.5))
facial_emotion_recognition_model.add(Dense(7, activation='softmax'))

facial_emotion_recognition_model.compile(loss='categorical_crossentropy', optimizer=Adam(
    learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

# Training and saving model
fer_model_info = facial_emotion_recognition_model.fit_generator(
    test_generator,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=7178 // 64
)

fer_model_json = facial_emotion_recognition_model.to_json()
with open('facial_emotion_recognition_model.json', 'w') as json_file:
    json_file.write(fer_model_json)

facial_emotion_recognition_model.save_weights(
    'facial_emotion_recognition_model.h5')
