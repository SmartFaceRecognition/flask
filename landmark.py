import cv2
import numpy as np
import matplotlib.pyplot as plt


# Landmark predict function
def landmark_predictor(detector, predictor, image):
    detections = detector(image)
    if not detections:
        return None
    landmarks = predictor(image, detections[0])  # 첫 번째 얼굴에 대해서만 랜드마크 추출
    landmarks_array = np.array([[point.x, point.y] for point in landmarks.parts()])

    return landmarks_array


# Founding eye center function
def eye_center(eye_landmark):
    x, y, w, h = cv2.boundingRect(eye_landmark)
    return (x+w/2, y+h/2)


# Face alignment function
def face_alignment(detector, predictor, image):
    landmark = landmark_predictor(detector, predictor, image)
    if landmark is None:
        return None

    for i in range(68):
        if landmark[i][0] < 0:
            landmark[i][0] = 0
        if landmark[i][1] < 0:
            landmark[i][1] = 0

    left_eye = eye_center(landmark[36:42])
    right_eye = eye_center(landmark[42:48])
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    eyes_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    rotated_landmark = cv2.transform(landmark.reshape(-1, 1, 2), M).reshape(-1, 2)

    face_contour = rotated_landmark[0:17]

    for i in range(17):
        if face_contour[i][0] < 0:
            face_contour[i][0] = 0
        if face_contour[i][1] < 0:
            face_contour[i][1] = 0
    face_contour = np.vstack((face_contour, np.flip(rotated_landmark[17:27], axis=0)))

    mask = np.zeros_like(rotated_image, dtype=np.uint8)
    pts = np.array(face_contour, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], (255, 255, 255))

    face = cv2.bitwise_and(rotated_image, mask)

    x, y, w, h = cv2.boundingRect(rotated_landmark)
    face = face[y:y + h, x:x + w]

    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    return face


