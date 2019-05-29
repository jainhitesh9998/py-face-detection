from threading import Thread

import cv2
import imutils
from py_tensorflow_runner.session_utils import SessionRunner
from py_face_detection.mtcnn_api.face_detector_api import FaceDetectorMTCNN

# from proj_data.py_face_detection.images import path as image_path

session_runner = SessionRunner()
detection = FaceDetectorMTCNN()
detection.use_threading()
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)
session_runner.start()
detection.run()

cap = cv2.VideoCapture(-1)


def read():
    while True:
        detector_ip.push_wait()
        ret, image = cap.read()
        if not ret:
            continue
        image = imutils.resize(image, width=1080)
        inference = FaceDetectorMTCNN.Inference(image)
        detector_ip.push(inference)


# def run():
#     while True:
#         detector_op.pull_wait()
#         ret, inference = detector_op.pull(True)
#         if ret:
#             faces = inference.get_result()
#             image = inference.get_input()
#             for face in faces:
#                 # cv2.rectangle(image, (face['rect'][0], face['rect'][1]), (face['rect'][2], face['rect'][3]),
#                 #               (0, 255, 0), 2)
#                 cv2.imshow("faces", face['face'])
#                 cv2.waitKey(1)
#
#             # cv2.imshow("faces", image)
#             # cv2.waitKey(1)

def annotate(image, bboxs):
    for bbox in bboxs:
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2)
    return image


def run():
    while True:
        detector_op.pull_wait()
        ret, inference = detector_op.pull(True)
        input_image = inference.get_input()
        if ret:
            detections = inference.get_result()
            for det in detections:
                input_image = annotate(input_image, [det['rect']])

        cv2.imshow("face", input_image)
        cv2.waitKey(1)


Thread(target=run).start()
Thread(target=read).start()
