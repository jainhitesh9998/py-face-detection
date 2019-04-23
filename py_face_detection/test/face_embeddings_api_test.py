from py_data.data import Data
DATA = Data()
DATA.create_dir('images')

from threading import Thread
import cv2
from py_tensorflow_runner.session_utils import SessionRunner

from py_face_detection.facenet_api.face_embeddings_api import FNEmbeddingsGenerator


cap = cv2.VideoCapture(-1)

session_runner = SessionRunner()
detection = FNEmbeddingsGenerator()
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)
session_runner.start()
detection.run()


def read():
    while True:
        detector_ip.push_wait()
        ret, image = cap.read()
        if not ret:
            continue
        inference = FNEmbeddingsGenerator.Inference(image)
        detector_ip.push(inference)


def run():
    while True:
        detector_op.pull_wait()
        ret, inference = detector_op.pull(True)
        if ret:
            i_dets = inference.get_result()
            # cv2.imshow("annotated", i_dets.get_annotated())
            # cv2.waitKey(1)
            print(i_dets)


Thread(target=run).start()
Thread(target=read).start()
