sudo docker run -it --rm \
-e PATH_IMAGES="images" \
-e URL_STREAMING="http://10.0.30.110:8080/video" \
-e SCALE_FACTOR=1.3 \
-e NEIGHBOURS=5 \
-e URL_ENDPOINT="http://localhost:5000/" \
-v /home/ailab/projects/face-recognition-system/cascade/images:/app/images \
face_detection /bin/bash
