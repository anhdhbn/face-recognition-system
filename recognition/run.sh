sudo docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=/home/anhdh/Projects/Lab/face-recognition/recognition/models/export,target=/models/facenet -e MODEL_NAME=facenet -t tensorflow/serving