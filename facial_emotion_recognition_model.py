from keras.preprocessing import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.optimizers import Adam


test_data_path = 'dataset/train/'
validation_data_path = 'dataset/validation/'

# Image preprocessing
test_data_generator = ImageDataGenerator(rescale=1.0/255.0)
validation_data_generator = ImageDataGenerator(rescale=1.0/255.0)

test_generator = train_data_generator.flow_from_directory(
    test_data_path,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

validation_generator = validation_data_generator.flow_from_directory(
    validation_data_path,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)
