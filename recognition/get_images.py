import requests
import os
import shutil

PATH = "images_data"

HOST = "http://recofat.vn"
API = "http://api.recofat.vn/api/PersonImages/ListAllPersonImage?token=recofat@2019"

response = requests.get(API)

try:
    shutil.rmtree(PATH)
except:
    pass

if response.status_code == 200: 
    infos = response.json()
    for info in infos:
        imagesPath = info['Images']
        try:
            person_path = os.path.join(PATH, str(info["PersonID"]))
            os.makedirs(person_path)
        except:
            pass
        for path in imagesPath:
            image = requests.get(HOST + path) 
            path_save = os.path.join(PATH, str(info["PersonID"]), path.split("/")[-1])
            print(path_save)
            try:
                with open(path_save, 'wb') as f:
                    f.write(image.content)
            except Exception as e:
                print(e)
else:
    print("Can not fetch data...")
