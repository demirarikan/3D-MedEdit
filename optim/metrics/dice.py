import numpy as np


def dice_coefficient_batch(a, b):
    intersection = np.sum(a * b, axis=(1, 2, 3))
    union = np.sum(a, axis=(1, 2, 3)) + np.sum(b, axis=(1, 2, 3))
    dice = (2.0 * intersection) / union
    mean_dice = np.mean(dice)
    return mean_dice
