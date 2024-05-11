import json

import dlib
import mxnet as mx
import numpy as np
import redis
from flask import Flask, request, render_template
from flask_cors import CORS
from mxnet.gluon.data.vision import transforms

from db_featuremap import db_feature_map_extractor
from face_recognition import face_recognition
from feature_extractor import Model, input_feature_map
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size
from yolov5.utils.torch_utils import select_device

app = Flask(__name__)
CORS(app)
# 라즈베리파이 영상 스트리밍 주소
# STREAM_URL = 'http://192.168.73.158:8081/?action=stream'
r = redis.StrictRedis(host='localhost', port=6379, db=1)

feature_maps = []
labels = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('DB/landmark_detector.dat')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = Model()
input_feature_map_extractor = input_feature_map(transform, model, mx.cpu())

device = select_device("cpu")
modelStreaming = DetectMultiBackend("yolov5/best_50.pt", device=device, dnn=False, data='yolov5/data/coco128.yaml',
                                    fp16=False)
stride, names, pt = modelStreaming.stride, modelStreaming.names, modelStreaming.pt
imgsz = check_img_size((640, 640), s=stride)  # check image size


@app.route('/streaming', methods=['GET'])
def video():
    from yolov5.recognize import run
    text = run(model=modelStreaming, stride=stride, names=names, pt=pt)
    return text


@app.route('/face', methods=['POST'])
def process_face_data():
    global feature_maps
    global labels

    data = request.get_json()  # 받은 JSON 데이터를 파싱
    faceUrl = data['faceUrl']
    personId = data['personId']

    # feature map 생성
    new_feature_map, new_labels = db_feature_map_extractor(detector, predictor, input_feature_map_extractor, personId,
                                                           faceUrl)
    r.set(personId, json.dumps(new_feature_map.tolist()))

    labels = [key.decode('utf-8') for key in r.keys('*')]

    feature_maps = []

    for key in r.keys('*'):
        value = r.get(key)
        if value is not None:
            value = np.array(eval(value.decode('utf-8')))
            feature_maps.append(value)
    feature_maps = np.stack(feature_maps)
    feature_maps = np.concatenate(feature_maps)

    face_recognition.set(feature_maps, labels)

    return '', 200  # 상태 코드 200 OK 반환


@app.route('/')
def video_show():
    return render_template('video_stream.html')


if __name__ == '__main__':

    labels = [key.decode('utf-8') for key in r.keys('*')]

    feature_maps = []

    for key in r.keys('*'):
        value = r.get(key)
        if value is not None:
            value = np.array(eval(value.decode('utf-8')))
            feature_maps.append(value)
    if len(feature_maps) != 0:
        feature_maps = np.stack(feature_maps)
        feature_maps = np.concatenate(feature_maps)

    face_recognition.set(feature_maps, labels)
    app.run(host='0.0.0.0', port=5000)
