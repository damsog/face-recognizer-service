# Face Recognizer Service

Face Detection and Recognition Module.

This service can be used to run a Face Detection Algorithm, or a Face Recognition Algorithm. Both can run on a single image or on a live video.

This module can be deployed as a stand alone app or as an API service.
This service can be used as a standalone app, but it was made to be deployed together with the main videoanaltytics server <br>
[Video-Analytics Server](https://github.com/damsog/video-analytics-server)


## Requirements
The python dependencies can be installed using the install script or with docker. just make sure to set up th gpu with cuda and cudnn.

##### Python Dependencies
- [Python 3.7](https://www.python.org/downloads/release/python-370/). Python should be >=3.7. this is due to flask async support.
- [Insightface 0.1.5](https://github.com/deepinsight/insightface) (Python Lib for Face Det and Face Rek)
- [mxnet-cu102](https://mxnet.apache.org/versions/1.7.0/get_started?platform=linux&language=python&processor=gpu&environ=pip&) could only make it work with this version of mxnet.
- [Virtualenv](https://pypi.org/project/virtualenv/)
##### Other Dependencies
- Nvidia GPU with [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive) and [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive) for CUDA 10.2 (It's the version I could made it work with. had to balance Insightface, mxnet and cuda versions and compatibility)
- [Docker](https://docs.docker.com/engine/install/ubuntu/) (Opcional)


## Set Up (Linux)

### *Set Up Nvidia GPU*

Before installing the application be it using docker or on the host machine directly you need to set up gpu support.
install [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive). I could only made it work with this version of CUDA due to balance CUDA, Insightface and mxnet versions compatibilities. <br>
install [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive) for CUDA 10.2. <br>

Now, you can either install the application locally, or skip the next section and go directly to build the docker container for an easier install.

### *Install the application*

#### *Local*
You can use the Makefile script which installs the python version required, virtualenv, creates a virtual environment and there installs all the
python packages necessary.

Install the pre-requisites (This is a one time run for system dependencies):
```sh
make setup-env
```

Now, to create a python venv and the python dependencies just run:
```sh
make
```

to remove the python env
```sh
make clean
```

the models for this version should be downloaded manually from :<br>
[this folder](https://onedrive.live.com/?authkey=%21ALRWS0dNaAhDRuc&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215836&parId=4A83B6B633B029CC%215834&action=locate) <br>
and then place them manually on ```~/.insightface/models```

Alternatively you can just build the image or pull it directly from docker hub
#### *Docker (Build the image Container)*

To build the image just open the terminal on the project directory and run the following command
```sh
docker build -t <img-name>:<tag> .
```

#### *Docker (Use the built container)*
pull it
```sh
docker pull laguzs/gnosis_recognizer_service .
```
## Deploy Service 

Before deploying the service you have to set up the environment variables. create a file called .env and copy the content from .base.env
```sh
cp .base.env .env
```

And now edit the .env file
```
SERVER_IP=<serveri>
SERVER_PORT=<port>
DETECTOR_LOAD_DEVICE=<-1/0/1/..>
RECOGNIZER_LOAD_DEVICE=<-1/0/1/..>
SERVER_SSL_CONTEXT=<context>
SERVER_SSL_KEYFILE=<keyfile>
LOGGER_LEVEL=<INFO/DEBUG>
ENV=<development/production>
```

Some of the variables are self explanatory, server ip and port, logger level for info or debug, env for dev or production. keep the following in mind:
Device ID for detector and recognizer. -1 for no gpu, 0-N for gpu id (if you have only 1 gpu use 0)
```
DETECTOR_LOAD_DEVICE=<-1/0/1/..>
RECOGNIZER_LOAD_DEVICE=<-1/0/1/..>
```
SSL context for https. leave empty for http
```
SERVER_SSL_CONTEXT=<context>
SERVER_SSL_KEYFILE=<keyfile>
```

### *Deploy from terminal*

To deploy the service first activate the virtualenv
```sh
source /path/to/venv/bin/activate
```
Then, navigate to the application folder and run
```sh
python run.py
```
or just:
```sh
make run
```
### *Deploy as Container*

Run Docker container. use --gpus all for gpu support, or remove it to run on without gpu

```sh
docker run --env-file .env --gpus all -p <host-port>:5000 laguzs/gnosis_recognizer_service:1.1
```
or in detatched mode
```sh
docker run --env-file .env --gpus all -p <host-port>:5000 laguzs/gnosis_recognizer_service:1.1
```
Or for ease, deploy using the docker compose file
edit the img name for your specific case. if you want to run on cpu remove or comment the lines 8-12 (deploy) which specifies gpu.
To run using docker compose
```sh
docker compose up
```
or in detatched mode
```sh
docker compose up -d
```
to stop 
```sh
docker compose down
```

Check outputs 
```sh
docker logs --tail N <container_id>
```
You can check out the endpoints on ```http://localhost:<host-port>/docs```

![image](https://user-images.githubusercontent.com/46113808/220405197-3dc861ca-c1df-440e-852f-49e62b3547da.png)

Encoder: takes one or multiple images with faces and generates an array that represents each face uniquely.
codes for faces of a person are very simmilar  between them (through vectorial distance) and very different from codes from other people.

Detector: takes one or multiple images and finds faces on it

Recognizer: takes one or multiple images and a dataset in json format (containing a set of labeled codes for multiple faces of people)
