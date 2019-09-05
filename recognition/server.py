import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # must import this although we will not use it explicitly
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import time
from keras.preprocessing.image import load_img, save_img, img_to_array
from flask import Flask, request, jsonify
import os
import argparse
from PIL import Image

app = Flask(__name__)
args = argparse.ArgumentParser()

PATH_IMAGES = os.getenv('PATH_IMAGES', None)
args.PATH_IMAGES = PATH_IMAGES

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def preprocess_image(image_path):
  img = load_img(image_path, target_size=(160, 160))
  img = img_to_array(img)
  img = np.expand_dims(img, axis=0).astype(float)
  img = (img - 127.5) / 128.0
  return img

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def, name="")
    return graph

def get_embedding(image_path, input, phase_train_placeholder, embedding, sess):
    img = preprocess_image(image_path)
    emb = sess.run(embedding, feed_dict={input: img, phase_train_placeholder: False})
    return emb.squeeze()

TENSOR_FROZEN_MODEL_PATH = './models/frozen_model.pb'

graph = load_graph(TENSOR_FROZEN_MODEL_PATH)
sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.60)))

output = sess.graph.get_tensor_by_name('model/embeddings:0')
input = sess.graph.get_tensor_by_name('model/input:0')
phase_train_placeholder = sess.graph.get_tensor_by_name('model/phase_train:0')

@app.route('/', methods=['POST'])
def post():
    data = request.get_json()
    path = os.path.join(args.PATH_IMAGES, data['file_path'])
    embedding = get_embedding(path, input, phase_train_placeholder, output, sess)
    return {
      "success": True,
      "embedding": embedding.shape
    }

if __name__ == "__main__":
  if args.PATH_IMAGES is None:
    print("PATH_IMAGES is not None")
    return
  app.run()