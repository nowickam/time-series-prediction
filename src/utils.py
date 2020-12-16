import pandas as pd
from scipy.stats import logistic
import random

from src.config import FACE_FILE


def read_csv(nodes, stride):
    """
    Reads the csv and outputs the first row with the time series prediction

    """
    csv = (pd.read_csv(FACE_FILE + 'FaceFour_3_Dimensions/FaceFourDimension1_TRAIN.csv', sep=';',
                       decimal=','))
    input = csv.iloc[0][1:].tolist()
    return input


def f(weights, input):
    return logistic.cdf(weights.dot(input))


def random_step_single(weight):
    return weight + random.uniform(-1 - weight, 1 - weight)


def square(l):
    return list(map(lambda x: x ** 2, l))
