FROM python:3.8-bullseye

RUN apt-get update && apt-get install -y \
	git \
	openjdk-17-jdk \
	openjdk-17-jre 
	
# WORKDIR .
ADD ./requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip
RUN pip install -U pip setuptools wheel
RUN pip install -r /workspace/requirements.txt

