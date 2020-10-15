FROM pytorch/pytorch
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 
COPY . /workspace
WORKDIR /workspace
RUN pip install -r requirements.txt
RUN mkdir /workspace
RUN cd /workspace
ENTRYPOINT ["python", "cpuInfer.py"]