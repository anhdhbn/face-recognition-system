FROM nvcr.io/nvidia/tensorflow:19.08-py3

LABEL Author=AnhDH

WORKDIR /app
ADD . /app

RUN apt-get update && \
    apt-get install -y libsm6 libxext6 \
    libsm6 libxrender1 libfontconfig1

RUN python3 -m pip install -r requirements.txt

# RUN sh -c ./download_models.sh
RUN ["python3", "get_images.py"]

# CMD ["python3", "get_embeddings.py"]
CMD ["sh", "-c", "./download_models.sh && python3 get_embeddings.py"]