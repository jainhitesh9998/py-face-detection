from threading import Thread
from time import sleep
import cv2
import imutils
from py_tensorflow_runner.session_utils import SessionRunner
from py_face_detection.comparator_api.embedding_generator import EmbeddingGenerator
from proj_data.py_face_detection.embeddings import path as emb_path
import pickle
import numpy as np

from py_face_detection.faiss_datastore.embeddings import Embeddings

emb_dict = pickle.load(open(emb_path.get("infy_v2.pickle"), "rb"))
id_name_dict = dict()
name_id_dict = dict()
for i, key in enumerate(emb_dict.keys()):
    id_name_dict[i] = key
    name_id_dict[key] = i
print(id_name_dict)

added_order = []
data = []

for k, items in emb_dict.items():
    for item in items:
        data.append(item[0])
        added_order.append(name_id_dict[k])

data = np.asarray(data)
added_order = np.asarray(added_order)
print(data.shape)
print(added_order.shape)
store = Embeddings(data, added_order, gpu=True, inbuilt_index=False)

session_runner = SessionRunner()
generator = EmbeddingGenerator()
generator_ip = generator.get_in_pipe()
generator_op = generator.get_out_pipe()
generator.use_session_runner(session_runner)
generator.run()

# cap = cv2.VideoCapture('/home/developer/Downloads/video.mp4')
cap = cv2.VideoCapture(0)

def read():
    while True:
        generator_ip.push_wait()
        ret, image = cap.read()
        if not ret:
            continue
        inference = EmbeddingGenerator.Inference(imutils.resize(image, width=1080))
        generator_ip.push(inference)
        sleep(0.03)

def annotate(image, bboxs, embeddings):
    for bbox, embedding in zip(bboxs,embeddings):
        identity_distance_dict = store.search(np.asarray([embedding]), len=5)
        if identity_distance_dict["distance"][0] > 0.8:
            name = "Unknown"
        else:
            name =id_name_dict[identity_distance_dict["identity"][0]]
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,0), 2)
        cv2.putText(image,name, (int(bbox[0]), int(bbox[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3,cv2.LINE_AA)
    return image

def run():
    while True:
        generator_op.pull_wait()
        ret, inference = generator_op.pull(True)
        input_image = inference.get_input()
        if ret and inference.get_result() is  not None:
            embeddings = inference.get_result()
            bbox = inference.get_meta('bbox')
            output = annotate(input_image, bbox, embeddings)
            # print(inference.get_result().shape)
        else:
            pass
        cv2.imshow("face", input_image)
        cv2.waitKey(1)



Thread(target=run).start()
Thread(target=read).start()