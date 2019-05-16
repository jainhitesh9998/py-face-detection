from threading import Thread
import cv2
import imutils
import numpy as np
from py_pipe.pipe import Pipe

from py_tensorflow_runner.session_utils import SessionRunner, Inference
from py_face_detection.facenet_api.face_embeddings_api import FNEmbeddingsGenerator
from py_face_detection.mtcnn_api.face_detector_api import FaceDetectorMTCNN

class EmbeddingGenerator:
    class Inference(Inference):

        def __init__(self, input, return_pipe=None, meta_dict=None):
            super().__init__(input, return_pipe, meta_dict)

    def __init__(self, face_size=160):

        self.generator = FNEmbeddingsGenerator()
        self.generator.use_threading()
        self.generator_ip = self.generator.get_in_pipe()
        self.generator_op = self.generator.get_out_pipe()

        self.detector = FaceDetectorMTCNN()
        self.detector.use_threading()
        self.detector_ip = self.detector.get_in_pipe()
        self.detector_op = self.detector.get_out_pipe()

        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

        self.__run_session_on_thread = False
        self.__face_size=face_size


    def __in_pipe_process(self, inference):
        return inference

    def __out_pipe_process(self, result):
        result, inference = result
        inference.set_result(result)
        if inference.get_return_pipe():
            return '\0'

        return inference


    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def use_session_runner(self, session_runner):
        self.session_runner = session_runner
        self.generator.use_session_runner(session_runner)
        self.detector.use_session_runner(session_runner)


    def step_1(self):
        while self.__thread:
            self.detector_ip.push_wait()
            self.__in_pipe.pull_wait()
            ret, inf = self.__in_pipe.pull()
            if not ret:
                continue
            image = inf.get_input()
            image = imutils.resize(image, width=None)
            inference = FaceDetectorMTCNN.Inference(image)
            inference.set_meta('EmbeddingGenerator.Inference', inf)
            self.detector_ip.push(inference)
        self.detector.stop()

    def step_2(self):
        while self.__thread:
            self.generator_ip.push_wait()
            self.detector_op.pull_wait()
            ret, inference = self.detector_op.pull(True)
            if ret:
                faces = inference.get_result()
                # print("faces: ", faces)
                inf = inference.get_meta('EmbeddingGenerator.Inference')
                inf.set_meta('bbox', list())
                if faces:
                    face_imgs = np.empty((len(faces), self.__face_size, self.__face_size, 3))
                    for i in range(len(faces)):
                        inf.get_meta('bbox').append(faces[i]['rect'])
                        face_imgs[i,:,:,:] = faces[i]['face']
                    inf.set_meta('face_image', face_imgs[0])

                    inference = FNEmbeddingsGenerator.Inference(input=face_imgs)
                    inference.set_meta('EmbeddingGenerator.Inference', inf)
                    self.generator_ip.push(inference)
                else:
                    self.__out_pipe.push((None, inf))
        self.detector.stop()
        self.generator.stop()


    def step_3(self):
        while self.__thread:
            self.generator_op.pull_wait()
            ret, inference = self.generator_op.pull(True)
            embedding = None
            if ret:
                embedding = inference.get_result()
                inference = inference.get_meta('EmbeddingGenerator.Inference')
                # print("shape emb",embedding.shape)
            self.__out_pipe.push((embedding, inference))
        self.generator.stop()

    def __run(self):
        self.session_runner.start()
        self.generator.run()
        self.detector.run()
        Thread(target=self.step_1).start()
        Thread(target=self.step_2).start()
        Thread(target=self.step_3).start()

    def run(self):
        self.__thread = Thread(target=self.__run)
        self.__thread.start()

    def stop(self):
        self.__thread = None
