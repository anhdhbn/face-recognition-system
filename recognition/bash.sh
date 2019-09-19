sudo docker run -it --rm    \
-e PATH_IMAGES="images" \
-e DATA_NAME=data.pkl \
-e TF_CPP_MIN_LOG_LEVEL=3 \
-v /home/ailab/projects/face-recognition-system/images:/app/images \
-v /home/ailab/projects/face-recognition-system/data:/app/data \
face_recognition /bin/bash