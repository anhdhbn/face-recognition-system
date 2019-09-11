from keras.preprocessing.image import load_img, save_img, img_to_array
import tensorflow as tf
import pickle
import numpy as np

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
    
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
  with open( name + '.pkl', 'rb') as f:
      return pickle.load(f)

def find_distance(v1, v2):
  cos_sin = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
  return np.arccos(cos_sin)

def find_min(vector, db):
  dt = [(person_id, find_distance(vector, v2)) for person_id, v2 in db.items()]
  person_id, _ = min(dt, key=lambda x: x[1]) 
  return person_id
