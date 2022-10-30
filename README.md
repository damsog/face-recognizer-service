# Face Recognizer Service

This service can be used to run a Face Detection Algorithm, or a Face Recognition Algorithm. Both can run on a single image or on a live video.

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

You can use the install.sh script which installs the python version required, virtualenv, creates a virtual environment and there installs all the
python packages necessary. run the script with the following 
```sh
/bin/bash install.sh
```

the models for this version should be downloaded manually from :<br>
https://onedrive.live.com/?authkey=%21ALRWS0dNaAhDRuc&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215836&parId=4A83B6B633B029CC%215834&action=locate <br>
and then place them manually on ```~/.insightface/models```

This creates a python virtual environment in the current directory and installs all dependencies there.

```sh
```

### *With Docker (Build the image Container)*

To build the image just open the terminal on the project directory and run the following command
```sh
docker build -t laguzs/face_analytics:1.0 .
```

## Deploy Service 

### *Deploy from terminal*

To deploy the service first activate the virtualenv
```sh
source /path/to/venv/bin/activate
```
Then, navigate to the application folder and run
```sh
python app.py
```

### *Deploy as Container*

Run Docker container
```sh
docker run --gpus all -p <host-port>:6000 -d laguzs/face_analytics:1.0
```
Check outputs 
```sh
docker logs --tail N <container_id>
```
