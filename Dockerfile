FROM nvidia/cuda:10.2-runtime-ubuntu18.04
WORKDIR /app

COPY . .

RUN apt-get update && apt-get -y install python3.7
RUN apt-get -y install python3-pip

RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y python3-opencv
#RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install cuda-10.2
#RUN apt-get -y install nvidia-cuda-toolkit
RUN apt-get -y install wget
RUN apt-get -y install unzip
RUN /usr/bin/python3.7 -m pip install virtualenv

ENV VIRTUAL_ENV=/app/face_videoanalytics_venv
RUN virtualenv $VIRTUAL_ENV --python=python3.7
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip
RUN python -m pip install virtualenv
RUN python -m pip install python-dotenv
RUN python -m pip install mxnet-cu102==1.7.0
RUN python -m pip install cython
RUN python -m pip install opencv-contrib-python
RUN python -m pip install onnxruntime
RUN python -m pip install insightface==0.1.5
RUN python -m pip install imutils
RUN python -m pip install uuid
RUN python -m pip install av
RUN python -m pip install aiortc
RUN python -m pip install aiohttp
RUN python -m pip install asyncio
RUN python -m pip install pyfiglet
RUN python -m pip install colorful
RUN python -m pip install iridi
RUN python -m pip install gdown

RUN mkdir ~/.insightface/ && mkdir ~/.insightface/models/ && mkdir ~/.insightface/models/retinaface_r50_v1/
RUN python models_downloader.py
RUN sleep 10

RUN unzip model-r100-ii.zip && rm model-r100-ii.zip
RUN mv model-r100-ii/ arcface_r100_v1/
RUN mv arcface_r100_v1/ ~/.insightface/models/

RUN unzip retinaface-R50.zip && rm retinaface-R50.zip
RUN mv R50-0000.params R50-symbol.json ~/.insightface/models/retinaface_r50_v1/

EXPOSE 6000

CMD [ "python" , "app.py" ]