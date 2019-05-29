from threading import Thread

import cv2
from py_tensorflow_runner.session_utils import SessionRunner
from proj_data.py_face_detection.embeddings import path as emb_path
import numpy as np
from py_face_detection.comparator_api.embedding_generator import EmbeddingGenerator
# import keyboard
session_runner = SessionRunner()

generator = EmbeddingGenerator()
generator_ip = generator.get_in_pipe()
generator_op = generator.get_out_pipe()
generator.use_session_runner(session_runner)
generator.run()

cap = cv2.VideoCapture(0)


def read():
    while True:
        generator_ip.push_wait()
        ret, image = cap.read()
        # ret, image = True, cv2.imread("/home/developer/PycharmProjects/facematch/images/ab_1.jpg")
        if not ret:
            continue
        # image = cv2.resize(image, (160, 160))
        inference = EmbeddingGenerator.Inference(image)
        generator_ip.push(inference)

def write_to_a_file(embedding, length = 512):
    # assert len(embedding[0] == length)
    try:
        emp_number = int(input("enter employee number: "))
    except:
        print("try again. Emp Id not correct")
    print("empid is : ", emp_number)
    np.save(emb_path.get() + "/" + str(emp_number), embedding)
    np.savetxt(emb_path.get() + "/" + str(emp_number), embedding,delimiter=",")

def run():
    while True:
        generator_op.pull_wait()
        ret, inference = generator_op.pull(True)
        if ret:
            embedding = inference.get_result()
            image = inference.get_meta('face_image')
            cv2.imshow("face", image)
            # if ()
            if cv2.waitKey(1) == ord('q'):
                print(type(embedding))
                # print(str(embedding))
                write_to_a_file(embedding)


Thread(target=run).start()
Thread(target=read).start()