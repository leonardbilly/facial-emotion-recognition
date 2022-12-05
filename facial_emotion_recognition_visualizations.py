from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from emotion_features_average import calculate_emotion_feature_averages
from emotion_correlation_matrices import generate_emotion_correlation_matrices


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

emotion_features_averages = calculate_emotion_feature_averages(train_data_dir)


emotion_correlation_matrices, plottable_correlations = generate_emotion_correlation_matrices(
    emotion_features_averages)
print(emotion_features_averages)
print(emotion_correlation_matrices)
print(plottable_correlations)
