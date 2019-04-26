import operator
from threading import Thread
import numpy as np
import cv2
from py_tensorflow_runner.session_utils import SessionRunner

from py_face_detection.comparator_api.embedding_generator import EmbeddingGenerator
from proj_data.py_face_detection.embeddings.embeddings import emp

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
    last = 0
    while True:
        generator_op.pull_wait()
        ret, inference = generator_op.pull(True)
        if ret:
            embedding = inference.get_result()
            image = inference.get_meta('face_image')
            cv2.imshow("face", image)
            cv2.waitKey(1)
            dist = {}
            for key in emp.keys():
                d = np.sqrt(np.sum(np.square(np.subtract(embedding, emp[key]))))
                if d < 0.85:
                    dist[key] = d
            dist = sorted(dist.items(), key=lambda kv: kv[1])
            if len(dist) > 0:
                if last == dist[0][0]:
                    continue
                print(dist[0][0])
                last = dist[0][0]


Thread(target=run).start()
Thread(target=read).start()
