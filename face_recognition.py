import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler


class Face_Recognition(object):

    def __init__(self, scaler):
        self.labels = None
        self.features = None
        self.scaler = scaler

    def set(self, features, labels):
        self.features = features
        self.labels = labels

    def evaluate(self, input_data):
        if input_data is None:
            print('empty')
        else:
            features = np.concatenate((self.features, input_data), axis=0)

            sum_of_squares = np.sum(features ** 2.0, axis=1, keepdims=True)
            d_mat = sum_of_squares + sum_of_squares.transpose() - (2.0 * np.dot(features, features.transpose()))

            output_idx = np.argmin(d_mat[-1][:-1])
            distance = np.min(d_mat[-1][:-1])
            print(output_idx, distance, d_mat[-1][:-1])

            new_d_mat = d_mat[-1][:-1]
            mean_d_mat = np.mean(new_d_mat)
            print(mean_d_mat, mean_d_mat - distance)


face_recognition = Face_Recognition(StandardScaler())
