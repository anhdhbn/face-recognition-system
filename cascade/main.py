import cv2
import uuid
import os
from pathlib import Path
import argparse
import asyncio
import requests
import sys
import aiohttp
from PIL import Image

PATH_IMAGES = os.getenv('PATH_IMAGES', None)
URL_STREAMING = os.getenv('URL_STREAMING', 0)
CASCADE = os.getenv('CASCADE', "haarcascade_frontalface_default.xml")
SCALE_FACTOR = os.getenv('SCALE_FACTOR', 1.3)
NEIGHBOURS = os.getenv('NEIGHBOURS', 5)
URL_ENDPOINT = os.getenv('URL_ENDPOINT', None)

args = argparse.ArgumentParser()

def main(args):
    if args.PATH_IMAGES is None:
        print("PATH_IMAGES is not None")
        return
    if args.URL_ENDPOINT is None:
        print("URL_ENDPOINT is not None")
        return
    cap = cv2.VideoCapture(args.URL_STREAMING)
    if cap is None or not cap.isOpened():
        print("unable to open video source")
        print(args.URL_STREAMING)
        return

    if not os.path.isfile(args.CASCADE):
        print("CASCADE is not exists")
        return
    
    face_cascade = cv2.CascadeClassifier(args.CASCADE)

    try:
        os.makedirs(PATH_IMAGES)
        args.PATH_IMAGES = PATH(args.PATH_IMAGES)
    except:
        pass

    try:
        os.makedirs(os.path.join(PATH_IMAGES, "cropped"))
        args.PATH_IMAGES = PATH(args.PATH_IMAGES)
    except:
        pass

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if ret is True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, float(args.SCALE_FACTOR), int(args.NEIGHBOURS))
                for position in faces:
                    post_processing(position, frame)
            else:
                break
        except:
            pass
        

def post_processing(position, frame):
    x, y, w, h = position
    id = uuid.uuid4()
    path = os.path.join(args.PATH_IMAGES, f"{id}.{x}.{y}.{w}.{h}.jpg")
    path_cropped = os.path.join(args.PATH_IMAGES, "cropped", f"{id}.jpg")
    # print(path)
    cv2.imwrite(path,frame)

    img = cv2.imread(path)
    crop_img = img[y:y+h, x:x+w]
    cv2.imwrite(path_cropped, crop_img)
    data = {
        'file_path': f"{id}.jpg",
        'x': int(x),
        'y': int(y),
        'w': int(w),
        'h': int(h)
    }
    # await requests.post(args.URL_ENDPOINT, data)
    # requests.post(args.URL_ENDPOINT, data)
    try:      
        if sys.version_info >= (3, 7):
            asyncio.run(request(data))
        else:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(request(data))
    except:
        pass

async def request(data):
    async with aiohttp.ClientSession() as session:
        async with session.post(args.URL_ENDPOINT, json=data) as resp:
            # print(resp.status)
            # print(await resp.text())
            pass

if __name__ == "__main__":
    
    args.PATH_IMAGES = PATH_IMAGES
    args.URL_STREAMING = URL_STREAMING
    args.CASCADE = CASCADE
    args.SCALE_FACTOR = SCALE_FACTOR
    args.NEIGHBOURS = NEIGHBOURS
    args.URL_ENDPOINT = URL_ENDPOINT
    main(args)
