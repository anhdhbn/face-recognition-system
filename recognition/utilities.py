from keras.preprocessing.image import load_img, save_img, img_to_array
import tensorflow as tf
import pickle
import numpy as np
import requests
import base64
import datetime
import os
import cv2
from PIL import Image
import io
from pytz import timezone

DATA_PATH = "./data"

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
    
def save_obj(obj, path ):
  try:
    os.makedirs(DATA_PATH)
  except:
    pass
  with open(os.path.join(DATA_PATH, path) , 'wb') as f:
      pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path ):
  print(DATA_PATH, path)
  with open( os.path.join(DATA_PATH, path), 'rb') as f:
    return pickle.load(f)

def find_distance(v1, v2):
  cos_sin = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
  return cos_sin

def find_min(vector, db):
  dt = [(person_id, find_distance(vector, v2)) for person_id, v2 in db.items()]
  person_id, _ = max(dt, key=lambda x: x[1])
  return person_id


def post_to_main_server(id_persion, parent_path, pos):
  API_SAVE_TAGGINGFACE = "http://api.recofat.vn/api/TaggingFaces?token=recofat@2019"
  item = {}
  x, y, w, h = pos
  # pos la x y w h, x y la toa do bat dau w h la chieu dai va chieu cao
  # dua vao do ma cat
  # print(image_to_base64(parent_path))
  img = cv2.imread(parent_path)
  cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(img, f'Person Id: {id_persion}', (x,y), font, 2, (0, 255, 0), 2, cv2.LINE_AA)

  item["TaggingFaceID"]=0  #truyền vào 0, ngầm hiểu làm thêm mới
  item["CameraID"]= 1 #Camera nhan dang, có thể thử với số 1, 2 gì đó
  item["PersonID"]= id_persion #Người đc nhận dạng, lấy theo PersonID đã lấy về hôm nọ nhé
  item["LocationID"]= 0 #Cái này thừa, đc lấy theo camera ở trên
  item["Time"]= datetime.datetime.now().astimezone(timezone('Asia/Ho_Chi_Minh'))
  item["Image"]= matrix_to_base64(img[:, :, ::-1])

  response = requests.post(API_SAVE_TAGGINGFACE, data=item)
  if response.status_code != 200:
    print("Can not save...")
  else:
    print(response.text)
    print("Save data successfully...")

def image_to_base64(path_image):
  with open(path_image, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    return encoded_string

def numpy_to_base64(img):
  """Convert a Numpy array to JSON string"""
  imdata = pickle.dumps(img)
  return base64.b64encode(imdata).decode('ascii')

def matrix_to_base64(img):
  img = Image.fromarray(img.astype("uint8"))
  rawBytes = io.BytesIO()
  img.save(rawBytes, "PNG")
  rawBytes.seek(0)
  return base64.b64encode(rawBytes.read()).decode("utf-8")

if __name__ == "__main__":
  post_to_main_server(3, "/home/ailab/projects/face-recognition-system/images/1685922a-3a8e-43f9-b08e-3dac09ace198.jpg", (361, 228, 362, 362))
