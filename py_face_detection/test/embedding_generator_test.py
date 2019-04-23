from threading import Thread

import cv2
from py_tensorflow_runner.session_utils import SessionRunner

from py_face_detection.comparator_api.embedding_generator import EmbeddingGenerator

session_runner = SessionRunner()

generator = EmbeddingGenerator()
generator_ip = generator.get_in_pipe()
generator_op = generator.get_out_pipe()
generator.use_session_runner(session_runner)
generator.run()

cap = cv2.VideoCapture(-1)


def read():
    while True:
        generator_ip.push_wait()
        ret, image = cap.read()
        if not ret:
            continue
        inference = EmbeddingGenerator.Inference(image)
        generator_ip.push(inference)


def run():
    while True:
        generator_op.pull_wait()
        ret, inference = generator_op.pull(True)
        if ret:
            embedding = inference.get_result()
            image = inference.get_meta('face_image')
            cv2.imshow("face", image)
            if cv2.waitKey(1) == ord('q'):
                print(str(embedding))


Thread(target=run).start()
Thread(target=read).start()