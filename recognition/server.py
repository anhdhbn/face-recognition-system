import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import time

from flask import Flask, request, jsonify
import os
import argparse
from PIL import Image
import utilities

from utilities import load_graph, load_obj, get_embedding, find_min

app = Flask(__name__)
args = argparse.ArgumentParser()

PATH_IMAGES = os.getenv('PATH_IMAGES', None)
args.PATH_IMAGES = PATH_IMAGES

TENSOR_FROZEN_MODEL_PATH = './models/frozen_model.pb'

graph = load_graph(TENSOR_FROZEN_MODEL_PATH)
sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.60)))

output = sess.graph.get_tensor_by_name('model/embeddings:0')
input = sess.graph.get_tensor_by_name('model/input:0')
phase_train_placeholder = sess.graph.get_tensor_by_name('model/phase_train:0')

data_pkl = load_obj("data")

@app.route('/', methods=['POST'])
def post():
    data = request.get_json()
    path = os.path.join(args.PATH_IMAGES, "cropped", data['file_path'])
    embedding = get_embedding(path, input, phase_train_placeholder, output, sess)   
    person_id = find_min(embedding, data_pkl)
    result =  {
      "success": True,
      "embedding": person_id
    }
    print(result)
    return result

@app.route('/', methods=['GET'])
def test():
  return "Okiela"
  
if __name__ == "__main__":
  if args.PATH_IMAGES is None:
    print("PATH_IMAGES is not None")
  else:
    app.run(debug=True,host='0.0.0.0',port=5000)