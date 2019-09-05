sudo docker run -dit --rm \
-e PATH_IMAGES="images" \
-e URL_STREAMING="http://10.0.30.219/video" \
-e SCALE_FACTOR=1.3 \
-e NEIGHBOURS=5 \
-v /home/Projects/Lab/face-recognition/cascade/images:/app/images \
face_detection