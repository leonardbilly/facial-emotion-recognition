from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical

train_data_dir = './dataset/train'
validation_data_dir = './dataset/validation'


def load_images(folder):
    image_files = []
    emotion_labels = []

    for image_folder in os.listdir(folder):
        for image_file_name in os.listdir(folder + '/' + image_folder):
            image_file = os.path.join(folder, image_folder, image_file_name)
            image_files.append(image_file)
            emotion_labels.append(image_folder)
        print(image_folder, "Complete")
    return image_files, emotion_labels


# Create train and validation dataframes
train_data_frame = pd.DataFrame()
train_data_frame['image_file'], train_data_frame['emotion_label'] = load_images(
    train_data_dir)
train_data_frame = train_data_frame.sample(
    frac=1).reset_index(drop=True)  # Shuffle the images
train_data_frame.head()

'''image_files = train_data_frame.iloc[0:25]

for image_index, image_file, emotion_label in image_files.itertuples():
    img = load_img(image_file)
    img = np.array(img)'''


# Get gray scale pixel values as features
def extract_image_features(images):
    features = []
    for image_index, image_file, emotion_label in train_data_frame.itertuples():
        image = Image.open(image_file)
        feature = np.reshape(image, (48*48))
        features.append(feature)
    return features


def calculate_emotion_feature_averages(folder):
    emotion_features = []
    emotion_features_averages = {}

    for image_folder in os.listdir(folder):
        for image_file_name in os.listdir(folder + '/' + image_folder):
            image_file = os.path.join(folder, image_folder, image_file_name)
            image = Image.open(image_file)
            image_feature = np.reshape(image, (48*48))
            emotion_features.append(image_feature)
            print(f'File: {image_file} --- Feature extraction complete')
        emotion_features_averages[image_folder] = np.mean(emotion_features)
        print(
            f'Folder: {image_folder.upper()} --- Emotion features average calculated.\n\n')
        emotion_features = []
    print('Finished calculating emotion features averages.')
    return emotion_features_averages


emotion_features_averages = calculate_emotion_feature_averages(train_data_dir)
print(emotion_features_averages)
