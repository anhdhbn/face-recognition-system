import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # must import this although we will not use it explicitly
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import time
from keras.preprocessing.image import load_img, save_img, img_to_array
# from matplotlib import pyplot as plt

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

img = preprocess_image("./test/test_img.png")

start = time.time()
for i in range (500):
    img = preprocess_image("./test/test_img.png")
print(time.time() - start)

def read_pb_graph(model):
  with gfile.FastGFile(model,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

TENSORRT_MODEL_PATH = './models/tensorrt.pb'



graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.80))) as sess:
        # read TensorRT model
        trt_graph = read_pb_graph(TENSORRT_MODEL_PATH)

        # obtain the corresponding input-output tensor
        tf.import_graph_def(trt_graph, name='')
        input = sess.graph.get_tensor_by_name('model/input:0')
        output = sess.graph.get_tensor_by_name('model/embeddings:0')
        phase_train_placeholder = graph.get_tensor_by_name('model/phase_train:0')

        # in this case, it demonstrates to perform inference for 50 times
        total_time = 0; n_time_inference = 500
        out_pred = sess.run(output, feed_dict={input: img, phase_train_placeholder: False})
        for i in range(n_time_inference):
            t1 = time.time()
            out_pred = sess.run(output, feed_dict={input: img, phase_train_placeholder: False})
            t2 = time.time()
            delta_time = t2 - t1
            total_time += delta_time
            print("needed time in inference-" + str(i) + ": ", delta_time)
        avg_time_tensorRT = total_time / n_time_inference
        print("average inference time: ", avg_time_tensorRT)
        print("total inference time: ", total_time)

FROZEN_MODEL_PATH = './models/frozen_model.pb'

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        # read TensorRT model
        frozen_graph = read_pb_graph(FROZEN_MODEL_PATH)

        # obtain the corresponding input-output tensor
        tf.import_graph_def(frozen_graph, name='')
        input = sess.graph.get_tensor_by_name('model/input:0')
        output = sess.graph.get_tensor_by_name('model/embeddings:0')
        phase_train_placeholder = graph.get_tensor_by_name('model/phase_train:0')

        # in this case, it demonstrates to perform inference for 50 times
        total_time2 = 0; n_time_inference = 500
        out_pred = sess.run(output, feed_dict={input: img, phase_train_placeholder: False})
        for i in range(n_time_inference):
            t1 = time.time()
            out_pred = sess.run(output, feed_dict={input: img, phase_train_placeholder: False})
            t2 = time.time()
            delta_time = t2 - t1
            total_time2 += delta_time
            print("needed time in inference-" + str(i) + ": ", delta_time)
        avg_time_original_model = total_time2 / n_time_inference
        print("average inference time: ", avg_time_original_model)
        print("total inference time: ", total_time2)
        print("TensorRT improvement compared to the original model:", avg_time_original_model/avg_time_tensorRT)
        print("TensorRT improvement compared to the original model:", total_time2 - total_time )