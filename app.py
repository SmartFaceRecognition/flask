import requests
from flask import Flask, Response, request, render_template
import cv2
import json
import numpy as np
import feature_excluded
import redis
from PIL import Image
from io import BytesIO

# from landmark import face_alignment
from yolov5.recognize import parse_opt, run

app = Flask(__name__)

# 라즈베리파이 영상 스트리밍 주소
STREAM_URL = 'http://192.168.73.158:8081/?action=stream'
redis_client = redis.StrictRedis(host='localhost', port=6379, db=1)

opt = parse_opt()
opt.source = STREAM_URL
opt.save_crop = True

cap = cv2.VideoCapture(STREAM_URL)


# def get_stream():
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

        # face_detector.predict(source=frame, device="cpu", save_crop=True)


        # 캡처한 프레임을 인코딩하여 스트리밍 프레임으로 전송합니다.
        # ret, buffer = cv2.imencode('.jpg', frame)
        # frame_bytes = buffer.tobytes()
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# @app.route('/face', methods=['POST'])
# def process_face_data():
#     data = request.get_json()  # 받은 JSON 데이터를 파싱
#     url = data['url']  # 'url' 키에 해당하는 값 추출
#     id = data['id']  # 'url' 키에 해당하는 값 추출
#
#     response = requests.get(url)
#     image_data = response.content  # 이미지 데이터 변수에 저장
#
#     image_byte = Image.open(BytesIO(image_data))
#
#     image_np = np.array(image_byte)
#
# #    aligned_face = face_alignment(image_np)
#
#     aligned_face_pil = Image.fromarray(aligned_face)
#
#     feature = feature_excluded.get_vector(aligned_face_pil)
#
#     key = id
#     value = {'feature': feature.tolist()}
#
#     redis_client.hset(key, 'feature', json.dumps(value))  # feature 값을 문자열로 변환하여 저장
#
#     return '', 200  # 상태 코드 200 OK 반환


# 아두이노 에서 지문인식에 해당하는 id를 받고/얼굴인식 후 추출값과 redis에 아두이노한테 받은 id값의 feature와 비교
@app.route('/face-verify', methods=['GET'])
def face_verify():
    data = request.get_json()
    id = data['id']

    # 얼굴인식 진행

    # redis에서 해당 id의 얼굴 추출값 데이터
    feature = redis_client.hget(id, 'feature')

    return '', 200  # 상태 코드 200 OK 반환


@app.route('/video')
def video():
    run(**vars(opt))
  #  return Response(get_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def video_show():
    return render_template('video_stream.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
