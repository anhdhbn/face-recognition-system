import cv2
import numpy as np
import requests
import json
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.preprocessing.image import load_img, save_img, img_to_array
import time


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

if __name__ == '__main__':
    # Read image file
    image_path = './test/test_img.png'

    norm_image = preprocess_image(image_path)

    headers = {'content-type': 'application/json'}

    data = json.dumps({
        'signature_name':'serving_default',
        'inputs': {
            "image": norm_image.tolist(),
            "train": False
        }
    })

    start = time.time()
    for _ in range(500):
        json_res = requests.post('http://localhost:8501/v1/models/facenet/versions/2:predict', data=data, headers=headers)
    print(time.time() - start)
    outputs = json.loads(json_res.text)['outputs']
    print(np.asarray(outputs)[0].shape)