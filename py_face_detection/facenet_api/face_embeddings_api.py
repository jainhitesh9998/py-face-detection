from py_data.data import Data

DATA = Data()
DATA.create_dir('pretrained/facenet')

import os
import tensorflow as tf

from threading import Thread
from py_pipe.pipe import Pipe
from py_tensorflow_runner.session_utils import SessionRunnable, Inference
from proj_data.py_face_detection.pretrained.facenet import path as facenet_path
from py_face_detection.facenet_api import facenet, detect_face

PRETRAINED_20170512_110547 = "20170512-110547.pb"
PRETRAINED_20180408_102900 = "20180408-102900.pb"
# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44



class FNEmbeddingsGenerator:
    class Inference(Inference):

        def __init__(self, input, return_pipe=None, meta_dict=None):
            super().__init__(input, return_pipe, meta_dict)

    def __init__(self, model_name=PRETRAINED_20180408_102900, graph_prefix=None, flush_pipe_on_read=False, face_resize = 160):

        facenet.load_model(facenet_path.get(model_name))
        self.__flush_pipe_on_read = flush_pipe_on_read

        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

        self.__run_session_on_thread = False
        self.__face_resize = face_resize
        if not graph_prefix:
            self.__graph_prefix = ''
        else:
            self.__graph_prefix = graph_prefix + '/'

    def __in_pipe_process(self, inference):
        resized = inference.get_input()
        prewhitened = facenet.prewhiten(resized)
        reshaped = prewhitened.reshape(-1, self.__face_resize, self.__face_resize, 3)
        # print(reshaped.shape)
        inference.set_data(reshaped)
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

        self.__images_placeholder = self.__tf_sess.graph.get_tensor_by_name(self.__graph_prefix + "input:0")
        self.__embeddings = self.__tf_sess.graph.get_tensor_by_name(self.__graph_prefix + "embeddings:0")
        self.__phase_train_placeholder = self.__tf_sess.graph.get_tensor_by_name(self.__graph_prefix + "phase_train:0")
        self.__embedding_size = self.__embeddings.get_shape()[1]

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
                self.__session_runner.get_in_pipe().push(
                    SessionRunnable(self.__job, inference, run_on_thread=self.__run_session_on_thread))

    def __job(self, inference):
        self.__out_pipe.push(
            (self.__tf_sess.run(self.__embeddings,
                                feed_dict={self.__images_placeholder: inference.get_data(),
                                           self.__phase_train_placeholder: False}), inference))

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
