import os
import pickle

import cv2
import dlib
import shutil
import numpy as np
import mxnet as mx

from mxnet.gluon.data.vision import transforms

from landmark import face_alignment
from yolov5.detect import parse_opt, run
from feature_extractor import Model, input_feature_map


opt = parse_opt()

def image_face_detect(image_path):
    opt.source = image_path
    opt.save_crop = True
    run(**vars(opt))
    crop_face = cv2.imread("yolov5/runs/detect/exp/crops/person/image.jpg")
    shutil.rmtree("yolov5/runs")
    return crop_face


def db_feature_map_extractor(detector, predictor, input_feature_map_extractor):
    feature_maps = []
    db_labels = []
    db_path = "DB/people"
    labels = os.listdir(db_path)
    for label in labels:
        image_path = os.path.join(db_path, label)

        crop_face = image_face_detect(image_path)
        aligned_face = face_alignment(detector, predictor, crop_face)
        if aligned_face is None:
            print(None)
        else:
            feature_map = input_feature_map_extractor.get_feature_map(aligned_face)

        feature_maps.append(feature_map)
        db_labels.append(int(label))

    print((np.array(feature_maps)).shape)
    print(db_labels)

    feature_maps = np.concatenate(feature_maps)
    return feature_maps, db_labels


def main():
    if os.path.isdir("yolov5/runs"):
        shutil.rmtree("yolov5/runs")
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

    feature_map_path = 'DB/featuremap/feature_maps.pkl'
    label_path = 'DB/featuremap/labels.pkl'

    with open(feature_map_path, 'wb') as f:
        pickle.dump(db_feature_maps, f)

    with open(label_path, 'wb') as f:
        pickle.dump(label_path, f)


if __name__ == "__main__":
    main()
