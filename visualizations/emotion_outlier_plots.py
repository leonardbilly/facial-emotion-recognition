'''
Generate outlier plots using features extracted from individual images in groups. The figures are saved in a figures directory.
Dependencies: os, PIL, matplotlib, numpy
'''
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def identify_outliers(folder):
    emotion_group_features = {}
    image_features_averages = []

    for emotion_group in os.listdir(folder):
        print(f'Flattening {emotion_group.upper()} images:')
        emotion_group_features[emotion_group] = {}
        for image_file in os.listdir(folder + '/' + emotion_group):
            image_path = os.path.join(folder, emotion_group, image_file)
            image = Image.open(image_path)
            image_features = np.reshape(image, (48*48))
            image_features_average = np.mean(image_features)
            image_features_averages.append(image_features_average)
            print(f'\t*File: {image_path} --- flattened.')
        print(f'Group: {emotion_group.upper()} --- flattened.')
        emotion_group_features[emotion_group] = image_features_averages
        emotion_group_features_average = np.mean(image_features_averages)
        print(
            f'Group: {emotion_group.upper()} Image features average: {emotion_group_features_average}')

        image_features_averages = []
        sorted_features = sorted(emotion_group_features)
        features = {
            key: emotion_group_features[key] for key in sorted_features}
    return features


def generate_outlier_plots(features):
    x = []
    y_features = []
    figure_index = 1

    for emotion_group, image_feature_values in features.items():
        figure = plt.figure(figure_index)
        x = np.array([(index+1) for index in range(len(image_feature_values))])
        y_features = np.array(image_feature_values)
        # Get line of best fit from features
        a, b = np.polyfit(x, y_features, 1)
        plt.plot(x, a*x+b, linewidth=3, color='red')
        plt.title(f'Outliers in {emotion_group.upper()} emotion group:')
        plt.scatter(x, y_features, color='green', marker='*', s=8)
        plt.savefig(f'figures/{emotion_group}_outlier_plots.png')
        x = []
        y_features = []
        figure_index += 1

    plt.show()
