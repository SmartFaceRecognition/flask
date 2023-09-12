import numpy as np


class Face_Recognition(object):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def evaluate(self, input_data):
        if input_data is None:
            print('empty')
        else:
            features = self.features
            test1 = np.array(features)
            test2 = np.array(input_data)
            features = np.concatenate((features, input_data), axis=0)

            sum_of_squares = np.sum(features ** 2.0, axis=1, keepdims=True)
            d_mat = sum_of_squares + sum_of_squares.transpose() - (2.0 * np.dot(features, features.transpose()))

            output_idx = np.argmin(d_mat[-1][:-1])
            distance = np.min(d_mat[-1][:-1])
            # name = ['오민택', '이현준', '조재호']
            print(output_idx, distance, d_mat[-1][:-1])
