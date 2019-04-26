#!/usr/bin/env python2.7

# sudo docker run --runtime=nvidia -p 8500:8500   --mount type=bind,source=/home/developer/serving/models/facenet,target=/models/facenet -e MODEL_NAME=facenet -t tensorflow/serving:latest-gpu

# This is a placeholder for a Google-internal import.
import time

from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import cv2
import os
import numpy as np
from scipy import misc

# tf.app.flags.DEFINE_string('server', 'localhost:8500',
#                            'PredictionService host:port')
# tf.app.flags.DEFINE_string('image', '/home/vivek/serving-1.4.0/test.jpeg', 'path to image in JPEG format')
# FLAGS = tf.app.flags.FLAGS

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def main(_):
  # host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel('localhost', 8500)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'facenet'
  request.model_spec.signature_name = 'calculate_embeddings'
  im = cv2.imread("/home/developer/PycharmProjects/facematch/images/ab_1.jpg")
  im_resized=None
  im = cv2.resize(im, (160,160))
  request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(im, shape=[1, im.shape[0], im.shape[1], im.shape[2]], dtype=tf.float32))
  request.inputs['phase'].CopyFrom(tf.contrib.util.make_tensor_proto(False))
  while True:
      start = time.time()
      result = stub.Predict(request, 10.0)  # 10 secs timeout
      response = np.array(result.outputs['embeddings'].float_val)
      # print(response)
      print(response.shape)
      print(type(response))
      print(time.time()-start)

if __name__ == '__main__':
  tf.app.run()

