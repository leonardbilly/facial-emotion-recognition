'''
Emotion correlation matrices calculator that generates the Pearson's correlation between emotions using emotion feature averages calculated from images grouped under different emotions in the dataset.
Dependencies: numpy
'''
import numpy as np


def generate_emotion_correlation_matrices(emotion_features_averages):
    emotion_correlation_matrices = {}

    for emotion_name, emotion_features_average in emotion_features_averages.items():
        print(f'Calcuating correlation matrices for {emotion_name}.')
        emotion_correlation_matrices[emotion_name] = {}

        for other_emotion_name, other_emotion_features_average in emotion_features_averages.items():
            # {emotion_name} : {other_emotion_name} : {emotion_features_average} : {other_emotion_features_average}
            emotion_correlation_matrix = np.corrcoef([emotion_features_average, 0], [
                                                     other_emotion_features_average, 0])
            emotion_correlation_matrices[emotion_name][other_emotion_name] = emotion_correlation_matrix
            print(
                f'\tFinished calculating {emotion_name}-{other_emotion_name} correlation matrix')
        print(
            f'Finished calculating {emotion_name} emotion correlation matrices.\n')
    print('Done calculating all emotion correlation matrices')
    return emotion_correlation_matrices
