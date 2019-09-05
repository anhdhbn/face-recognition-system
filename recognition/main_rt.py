import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # must import this although we will not use it explicitly
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import time
from keras.preprocessing.image import load_img, save_img, img_to_array
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def preprocess_image(image_path):
    ''' Load the image, resize and normalized it
    Note: The way we normalize the image here should be
    consistant with the way we nomalize the images while training
    '''

    img = load_img(image_path, target_size=(160, 160))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0).astype(float)
    img = (img - 127.5) / 128.0
    return img

def read_pb_graph(path):
  with gfile.FastGFile(path,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

def get_embedding(image_path, graph, input, phase_train_placeholder, embedding, sess):
    img = preprocess_image(image_path)
    emb = sess.run(embedding, feed_dict={input: img, phase_train_placeholder: False})
    return emb.squeeze()


TENSOR_FROZEN_MODEL_PATH = './models/frozen_model.pb'
TENSORRT_MODEL_PATH = './models/tensorrt2.pb'


    # print(result)

def get_time(path):
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.80))) as sess:
        trt_graph = read_pb_graph(path)

        tf.import_graph_def(trt_graph, name='')

        input = sess.graph.get_tensor_by_name('model/input:0')
        output = sess.graph.get_tensor_by_name('model/embeddings:0')
        phase_train_placeholder = sess.graph.get_tensor_by_name('model/phase_train:0')

        result = get_embedding("./test/test_img.png", trt_graph, input, phase_train_placeholder, output, sess)
        start = time.time()
        for i in range(500):
            result = get_embedding("./test/test_img.png", trt_graph, input, phase_train_placeholder, output, sess)
        return  time.time() - start

time1 = get_time(TENSOR_FROZEN_MODEL_PATH)

time2 = get_time(TENSORRT_MODEL_PATH)

print(time1 ,time2)

print(time1 - time2)