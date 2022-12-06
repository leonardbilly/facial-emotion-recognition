'''
Generates figure showing Pearson's correlation of different human emotions.
Positive(+) correlation means direct proportionality, Negative(-) correlation means inverse proportionality.
Dependencies: matplotlib.pyplot, numpy
'''
import matplotlib.pyplot as plt
import numpy as np


def generate_emotion_heat_map(emotion_correlation_matrices, plottable_correlations):
    emotions = [emotion for emotion in emotion_correlation_matrices.keys()]
    correlations = np.array(plottable_correlations)

    figure, axes = plt.subplots()
    visual = axes.imshow(correlations)

    axes.set_xticks(np.arange(len(emotions)), labels=emotions)
    axes.set_yticks(np.arange(len(emotions)), labels=emotions)

    plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(emotions)):
        for j in range(len(emotions)):
            text = axes.text(j, i, correlations[i, j],
                             ha="center", va="center", color="w")

    axes.set_title("PEARSON'S CORRELATION BETWEEN HUMAN EMOTIONS.")
    figure.tight_layout()
    plt.savefig('figures/emotion_heat_map.png')
    plt.show()
