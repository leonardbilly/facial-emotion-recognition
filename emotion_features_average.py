"""
Emotion feature average calculator that returns a dictionary containing an average of the gray scale pixel features extracted from the images in a particular emotion folder in the dataset.
"""
from PIL import Image
import os
import numpy as np


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
