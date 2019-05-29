from flask import Flask, jsonify
from py_flask_movie.flask_movie import FlaskMovie
from py_pipe.pipe import Pipe

app = Flask(__name__)

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

pipe = Pipe(limit=1)
fs = FlaskMovie(app=app)
fs.create('face_image', pipe, np.zeros((160, 160, 3)), timeout=1)

import os


def get_cam_id():
    dict = {}
    for file in os.listdir("/sys/class/video4linux"):
        if not file.startswith("video"):
            continue
        try:
            real_file = os.path.realpath("/sys/class/video4linux/" + file)
            dir = os.path.abspath(real_file + "../../../../")
            serial = open(dir + "/serial").readline().strip()
            index = file.replace("video", "")
            dict.update({serial: int(index)})

        except:
            continue
    return dict


cam_dict = get_cam_id()

caps = {
    'BUILDING_IN': cam_dict['D181B16F'],
    'BUILDING_OUT': cam_dict['811BF06F'],
}

def capture(direction):
    cap = cv2.VideoCapture(caps[direction])
    ret, image = cap.read()
    cap.release()
    return ret, image

@app.route('/<emp_id>/<direction>')
def processReq(emp_id, direction):
    if emp_id not in emp.keys():
        return jsonify({"result":"not-found"})
    ret, image = capture(direction)
    if not ret:
        return jsonify({"result":"none"})

    inference = EmbeddingGenerator.Inference(image)
    generator_ip.push(inference)
    generator_op.pull_wait()
    ret, inference = generator_op.pull(True)
    if ret:
        embedding = inference.get_result()
        face_image = inference.get_meta('face_image')
        pipe.push(face_image)
        d = np.sqrt(np.sum(np.square(np.subtract(embedding, emp[emp_id]))))
        print(emp_id, d)
        if d < 0.85:
            # requests.get("http://192.168.0.7:5000/turnstile", {'direction': direction})
            return jsonify({"result":"valid"})
        return jsonify({"result":"invalid"})
    return jsonify({"result":"unknown"})


if __name__ == '__main__':
    fs.start("0.0.0.0", 4500)
