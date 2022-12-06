from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from visualizations.emotion_features_average import calculate_emotion_feature_averages
from visualizations.emotion_correlation_matrices import generate_emotion_correlation_matrices
from visualizations.emotion_heat_map import generate_emotion_heat_map
from visualizations.emotion_outlier_plots import identify_outliers, generate_outlier_plots

train_data_dir = './dataset/train'
validation_data_dir = './dataset/validation'

emotion_features_averages = calculate_emotion_feature_averages(train_data_dir)
emotion_correlation_matrices, plottable_correlations = generate_emotion_correlation_matrices(
    emotion_features_averages)
generate_emotion_heat_map(
    emotion_correlation_matrices, plottable_correlations)
features = identify_outliers(train_data_dir)
generate_outlier_plots(features)
