import os
import cv2
import dlib
import shutil

import numpy as np
import mxnet as mx

from mxnet.gluon.data.vision import transforms

from landmark import face_alignment
from feature_extractor import Model, input_feature_map

opt = parse_opt()


def image_face_detect(image_path):
    opt.source = image_path
    opt.save_crop = True
    run(**vars(opt))
    crop_face = cv2.imread("yolov5/runs/detect/exp/crops/person/image.jpg")
    shutil.rmtree("yolov5/runs")
    return crop_face


def cam_face_detect(image_path):
    opt.source = image_path
    opt.save_crop = True
    run(**vars(opt))
    print("a")
    crop_face = cv2.imread("yolov5/runs/detect/exp/crops/person/_action_stream.jpg")
    shutil.rmtree("yolov5/runs")
    return crop_face


def db_feature_map_extractor(detector, predictor, input_feature_map_extractor, crop_face):
    feature_maps = []
    db_labels = []
    db_path = "DB/people"
    labels = os.listdir(db_path)
    for label in labels:
        image_path = os.path.join(db_path, label)

        crop_face = image_face_detect(image_path)
        aligned_face = face_alignment(detector, predictor, crop_face)

        feature_map = input_feature_map_extractor.get_feature_map(aligned_face)

        feature_maps.append(feature_map)
        db_labels.append(label)

    return feature_maps, db_labels


class Face_Recognition(object):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def evaluate(self, input_data):
        features = self.features
        features = np.concatenate((features, input_data), axis=0)

        sum_of_squares = np.sum(features ** 2.0, axis=1, keepdims=True)
        d_mat = sum_of_squares + sum_of_squares.transpose() - (2.0 * np.dot(features, features.transpose()))

        output_idx = np.argmin(d_mat[-1][:-1]) + 1
        name = ['오민택', '이현준', '조재호']
        print(name)


def main():
    # 모델 불러오기, 셋팅
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('DB/landmark_detector.dat')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = Model()
    input_feature_map_extractor = input_feature_map(transform, model, mx.cpu())

    # DB feature map 생성
    db_feature_maps, db_labels = db_feature_map_extractor(detector, predictor, input_feature_map_extractor)

    face_recognitor = Face_Recognition(db_feature_maps, db_labels)

    # 웹캠 받아오기 루프
    STREAM_URL = 'http://192.168.73.158:8081/?action=stream'

    cap = cv2.VideoCapture(STREAM_URL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)

        crop_face = cam_face_detect(buffer)
        aligned_face = face_alignment(detector, predictor, crop_face)

        feature_map = input_feature_map_extractor.get_feature_map(aligned_face)

        face_recognitor.evaluate(feature_map)


if __name__ == "__main__":
    main()
