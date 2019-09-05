# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM anhdhbn/docker-python-opencv-ffmpeg

# If you prefer miniconda:
#FROM continuumio/miniconda3

LABEL Name=face-detection Version=1.0
LABEL Author=AnhDH

# EXPOSE 3000

WORKDIR /app
ADD . /app


RUN \
apt-get update && \
# apt-get -y upgrade && \
# apt-get install software-properties-common -y && \
# add-apt-repository ppa:fkrull/deadsnakes && \
# apt-get update && \
# apt-get install -y nano vim curl
apt-get install -y vim curl

RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]

# Using pipenv:
#RUN python3 -m pip install pipenv
#RUN pipenv install --ignore-pipfile
#CMD ["pipenv", "run", "python3", "-m", "face-recognition"]

# Using miniconda (make sure to replace 'myenv' w/ your environment name):
#RUN conda env create -f environment.yml
#CMD /bin/bash -c "source activate myenv && python3 -m face-recognition"
