sudo docker run -it --rm  \
-e PATH_IMAGES="images" \
-v /home/ailab/projects/face-recognition-system/cascade/images:/app/images \
face_recognition /bin/bash