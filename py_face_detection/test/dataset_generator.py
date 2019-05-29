from threading import Thread
from time import sleep

import cv2
import imutils
from py_tensorflow_runner.session_utils import SessionRunner

from py_face_detection.comparator_api.embedding_generator import EmbeddingGenerator
import  glob
from proj_data.py_face_detection.infy_images import path as images_path
from proj_data.py_face_detection.embeddings import path as emb_path
import pickle

stop_flag = True

session_runner = SessionRunner()
generator = EmbeddingGenerator()
generator_ip = generator.get_in_pipe()
generator_op = generator.get_out_pipe()
generator.use_session_runner(session_runner)
generator.run()
#
# cap = cv2.VideoCapture('/home/developer/Downloads/video.mp4')
emb_dict = dict()
images_list = glob.glob(images_path.get()+"/**/*.JPG")
print(images_list)
image_dict = dict()
for images in images_list:
    person_name = images.split("/")[-2]
    if person_name not in image_dict.keys():
        image_dict[person_name] = list()
    image_dict[person_name].append(images)

def read():
    for k, image_list in image_dict.items():
        for image_path in image_list:
            generator_ip.push_wait()
            ret, image = True, cv2.imread(image_path)
            if not ret:
                continue
            inference = EmbeddingGenerator.Inference(imutils.resize(image, width=1080))
            inference.set_meta("person_name", k)
            generator_ip.push(inference)


def annotate(image, bboxs):
    for bbox in bboxs:
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,0), 2)
    return image

def run():
    while True:
        generator_op.pull_wait()
        ret, inference = generator_op.pull(True)
        input_image = inference.get_input()
        person_name = inference.get_meta("person_name")
        if ret and inference.get_result() is not None:
            if person_name not in emb_dict.keys():
                emb_dict[person_name]=list()
            embedding = inference.get_result()
            emb_dict[person_name].append(embedding)
            print(person_name,type(embedding))
        else:
            pass
        cv2.imshow("face", input_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        Thread(target=read).start()
        run()
    except:
        pass
    finally:
        pickle.dump(emb_dict, open(emb_path.get() + "/infy_v2.pickle", "wb"))
        print("writing pickle file")