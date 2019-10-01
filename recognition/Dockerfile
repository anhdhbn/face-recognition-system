FROM nvcr.io/nvidia/tensorflow:19.08-py3

# If you prefer miniconda:
#FROM continuumio/miniconda3

LABEL Name=face-recognition Version=0.0.1
LABEL Author=AnhDH


WORKDIR /app
ADD . /app


RUN apt-get update && \
    apt-get install -y libsm6 libxext6 \
     libsm6 libxrender1 libfontconfig1

# Using pip:
RUN python3 -m pip install -r requirements.txt

CMD ["sh", "-c", "./download_models.sh &&", "sh", "-c", "./download_data.sh" , "sh", "-c", "./wait-for-it.sh", "&& python3", "server.py"]

# Using pipenv:
#RUN python3 -m pip install pipenv
#RUN pipenv install --ignore-pipfile
#CMD ["pipenv", "run", "python3", "-m", "face-recognition"]

# Using miniconda (make sure to replace 'myenv' w/ your environment name):
#RUN conda env create -f environment.yml
#CMD /bin/bash -c "source activate myenv && python3 -m face-recognition"
