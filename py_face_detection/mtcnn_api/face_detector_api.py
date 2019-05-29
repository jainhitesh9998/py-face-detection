from py_data.data import Data

DATA = Data()
DATA.create_dir('pretrained/align')

import os
import tensorflow as tf
from threading import Thread
import cv2
import numpy as np
from py_pipe.pipe import Pipe
from py_tensorflow_runner.session_utils import Inference
from py_face_detection.mtcnn_api import detect_face
from proj_data.py_face_detection.pretrained.align import path as align_path

# some constants kept as default from facenet
minsize = 30
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44



class FaceDetectorMTCNN():
    class Inference(Inference):

        def __init__(self, input, return_pipe=None, meta_dict=None):
            super().__init__(input, return_pipe, meta_dict)

    def __init__(self, graph_prefix=None, flush_pipe_on_read=False, face_prob_score = 0.835, face_resize = 160):
        self.__face_prob_score = face_prob_score
        self.__flush_pipe_on_read = flush_pipe_on_read

        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)
        self.__face_resize = face_resize

        self.__run_session_on_thread = False

        if not graph_prefix:
            self.__graph_prefix = ''
        else:
            self.__graph_prefix = graph_prefix + '/'

    def __in_pipe_process(self, inference):
        img = inference.get_input()
        inference.set_data(img)
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

    def use_threading(self, run_on_thread=True):
        self.__run_session_on_thread = run_on_thread

    def use_session_runner(self, session_runner):
        self.__session_runner = session_runner
        self.__tf_sess = session_runner.get_session()
        self.__pnet, self.__rnet, self.__onet = detect_face.create_mtcnn(self.__tf_sess, align_path.get())

    def run(self):
        if self.__thread is None:
            self.__thread = Thread(target=self.__run)
            self.__thread.start()

    def __run(self):
        while self.__thread:

            if self.__in_pipe.is_closed():
                self.__out_pipe.close()
                return

            self.__in_pipe.pull_wait()
            ret, inference = self.__in_pipe.pull(self.__flush_pipe_on_read)
            if ret:
                self.__job(inference)

    def __job(self, inference):
        img = inference.get_data()
        faces = []
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, self.__pnet, self.__rnet, self.__onet, threshold,
                                                    factor)
        if not len(bounding_boxes) == 0:
            for face in bounding_boxes:
                if face[4] > self.__face_prob_score:
                    det = np.squeeze(face[0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    resized = cv2.resize(cropped, (self.__face_resize, self.__face_resize), interpolation=cv2.INTER_CUBIC)
                    faces.append({'face': resized, 'rect': [bb[0], bb[1], bb[2], bb[3]]})
        # print("len of faces", len(faces))
        self.__out_pipe.push((faces, inference))

    def stop(self):
        self.__thread = None

    def save_for_serving(self, save_path, model_version):
        path = os.path.join(save_path, str(model_version))
        builder = tf.saved_model.builder.SavedModelBuilder(path)

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tf.saved_model.utils.build_tensor_info(self.__images_placeholder),
                    'phase': tf.saved_model.utils.build_tensor_info(self.__phase_train_placeholder)},
            outputs={
                'embeddings': tf.saved_model.utils.build_tensor_info(self.__embeddings)
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')

        builder.add_meta_graph_and_variables(
            self.__tf_sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'calculate_embeddings': prediction_signature,
            })

        builder.save()
        print("model saved successfullt")
