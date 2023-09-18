import os
import shutil
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image
from flask import jsonify

from landmark import face_alignment
from yolov5.detect import parse_opt, run

opt = parse_opt()


def image_face_detect(faceUrl):
    if os.path.isdir("yolov5/runs"):
        shutil.rmtree("yolov5/runs")

    response = requests.get(faceUrl)
    image = Image.open(BytesIO(response.content))

    # 이미지 객체를 바이너리 데이터로 변환합니다.
    local_image_path = 'image.jpg'
    image.save(local_image_path, format="JPEG")

    opt.source = local_image_path
    opt.save_crop = True

    run(**vars(opt))
    os.remove(local_image_path)

    crop_face = cv2.imread("yolov5/runs/detect/exp/crops/person/image.jpg")
    shutil.rmtree("yolov5/runs")
    return crop_face


def db_feature_map_extractor(detector, predictor, input_feature_map_extractor, index, faceUrl):
    feature_maps = []
    db_labels = []

    crop_face = image_face_detect(faceUrl)
    aligned_face = face_alignment(detector, predictor, crop_face)

    # 얼굴 인식 불가능시 400 에러 처리
    if aligned_face is None:
        return jsonify({'error': 'aligned_face is None'}), 400
    else:
        feature_map = input_feature_map_extractor.get_feature_map(aligned_face)

    feature_maps.append(feature_map)
    db_labels.append(index)

    print((np.array(feature_maps)).shape)
    print(db_labels)

    feature_maps = np.concatenate(feature_maps)
    return feature_maps, db_labels
