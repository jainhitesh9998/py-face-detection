from flask import Flask, json, Response
from flask_cors import CORS
import cv2
from py_pipe.pipe import Pipe

from py_flask_movie.flask_movie import FlaskMovie

app = Flask(__name__)

from threading import Thread
import numpy as np

import requests

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


@app.route('/<emp_id>/<direction>')
def processReq(emp_id, direction):
    cap = cv2.VideoCapture(-1)
    ret, image = cap.read()
    if not ret:
        return Response(json.dumps("{'none': none }"), status=200, mimetype='application/json')
    cap.release()

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
            requests.get("http://192.168.254.1:5000/turnstile", {'direction': direction})
            return Response(json.dumps("{'result': valid }"), status=200, mimetype='application/json')
        return Response(json.dumps("{'result': invalid }"), status=200, mimetype='application/json')
    return Response(json.dumps("{'result': unknown }"), status=200, mimetype='application/json')



if __name__ == '__main__':
    fs.start("0.0.0.0", 5000)
