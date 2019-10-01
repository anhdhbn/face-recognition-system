import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import time
from keras.preprocessing.image import load_img, save_img, img_to_array
import os
import argparse
from PIL import Image
import glob
from pathlib import Path
import pickle
import utilities
import argparse

from utilities import load_graph, save_obj, get_embedding

PATH = "images_data"

args = argparse.ArgumentParser()
DATA_NAME = os.getenv('DATA_NAME', "data.pkl")
args.DATA_NAME = DATA_NAME

TENSOR_FROZEN_MODEL_PATH = './models/frozen_model.pb'


def main(args):
    graph = load_graph(TENSOR_FROZEN_MODEL_PATH)
    sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.60)))

    output = sess.graph.get_tensor_by_name('model/embeddings:0')
    input = sess.graph.get_tensor_by_name('model/input:0')
    phase_train_placeholder = sess.graph.get_tensor_by_name('model/phase_train:0')

    all_person = [path.split("/")[-1] for path in glob.glob(f"./{PATH}/*")]
    data = {}

    for person in all_person:
        all_files = glob.glob(f"./{PATH}/{person}/*.jpg")
        if(len(all_files) != 0):
            all_embeddings = np.asarray([get_embedding(file, input, phase_train_placeholder, output, sess) for file in all_files])
            avg_embedding = np.sum(all_embeddings, axis = 0)/len(all_files)
            data[person] = avg_embedding
        
    save_obj(data, args.DATA_NAME)

if __name__ == "__main__":
    if args.DATA_NAME is None:
        print("DATA_NAME is not None")
    else:
        main(args)




