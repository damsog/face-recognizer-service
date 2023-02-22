# Face Recognizer Service

Face Detection and Recognition Module. can be used to detect faces and apply recognition on images and videos. <br>
it can be used from CLI, imported as a module or run as an API for another application. <br>
It's basically an interface using [Insightface 0.1.5](https://github.com/deepinsight/insightface) backend models for detection and recognition.

This module was made to be deployed together with the main videoanaltytics server [Video-Analytics Server](https://github.com/damsog/gnosis-main-service)
However, Can also be used as a standalone app, continue reading to learn how.


## :clipboard: Requirements
The python dependencies can be installed using the install script or with docker. just make sure to set up th gpu with cuda and cudnn.

##### :snake: Python Dependencies
- [Python 3.7](https://www.python.org/downloads/release/python-370/). Python should be >=3.7. this is due to flask async support.
- [Insightface 0.1.5](https://github.com/deepinsight/insightface) (Python Lib for Face Det and Face Rek)
- [mxnet-cu102](https://mxnet.apache.org/versions/1.7.0/get_started?platform=linux&language=python&processor=gpu&environ=pip&) could only make it work with this version of mxnet.
- [Virtualenv](https://pypi.org/project/virtualenv/)
- [FastAPI](https://fastapi.tiangolo.com/)
##### :penguin: Other Dependencies
- Nvidia GPU with [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive) and [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive) for CUDA 10.2 (It's the version I could made it work with. had to balance Insightface, mxnet and cuda versions and compatibility)
- [Docker](https://docs.docker.com/engine/install/ubuntu/) (Opcional)


## :wrench: Set Up (Linux)

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
#### :whale: *Docker (Build the image Container)*

To build the image just open the terminal on the project directory and run the following command
```sh
docker build -t <img-name>:<tag> .
```

#### *Docker (Use the built container)*
pull it
```sh
docker pull laguzs/gnosis_recognizer_service
```

## :computer: Run on cli

You can just use the system on your terminal with the following commands (be sure to have activated the venv:  ```source /path/to/venv/bin/activate```)

#### :suspect: *Detection*
To find faces on an image you can run:
```sh
python videoAnalytics/processor.py detection -i <img> -s
```
the flag ```-s``` displays the image using opencv. you can add the flags ```-dd [0-N]``` to specify the device to run on (by default -1 for cpu, 0 to N for GPU)
```sh
python videoAnalytics/processor.py detection -i <img> -s -dd 0
```
the flag ```-p``` prints the faces bounding box on the terminal if you want more information.
```sh
python videoAnalytics/processor.py detection -i <img> -s -dd 0 -p
```
To test on a live video use the flag ```-t```. NOTE: don't use the ```-s``` together with ```-t```
```sh
python videoAnalytics/processor.py detection -t -dd 0
```
#### :memo: *Encoder*
Before using the recognizer we have to generate a database where to keep the codes of faces and people that the system will use as refference to then apply on images or videos and label them correctly. <br>
To extract the code for a set of faces we can use the encoder module. <br>
You can tell the encoder the input images using the flag ```-i``` and specifying the label of the person and the path to the images in a json format:
```sh
python videoAnalytics/encoder.py -i '{"person1":["path1.jpg","path2.jpg",...], "person2":[...], "person3":[...], ... }'
```
An example with 2 people and 2 images each:
```sh
python videoAnalytics/encoder.py -i '{"person1":["person1_1.jpg","person1_2.jpg"], "person2":["person2_1.jpg","person2_2.jpg"]}'
```
to save the output use ```-o <output>```.
```sh
python videoAnalytics/encoder.py -i <input> -o <output>
```
You can also add the flags ```-dd [0-N]``` to specify the device to run detection on and ```-rd [0-N]``` for recognition (by default -1 for cpu, 0 to N for GPU)
```sh
python videoAnalytics/encoder.py -i <input> -dd 0 -rd 0 -o <output>
```
Alternatively, you can use the flag ```-p``` to specify a location containing the images. they should be organized in a general folder which contains a folder for each person.
each person folder should have the label of the person (it's name) and inside there should be the face images for that person:
```
data/
    |_person1
            |_person1_1.jpg
            |_person1_2.jpg
            ...
    |_person2
            |_person2_1.jpg
            ...
    |_person3
            ...
    ...
```
```sh
python videoAnalytics/encoder.py -p <path-data> -o <output>
```
Either way, the output dataset should have the following structure:

*dataset.json*
```json
[
    [
        "person1",
        "0.2681454,0.9309953,-0.98818445,1.1641849...."
    ],
    [
        "person1",
        "2.291303,0.9101235,-0.6601216,0.14975247....."
    ],
    [
        "person2",
        "-1.9095982,-0.6114771,0.49495777,-1.1133002..."
    ],
    [
        "..."
    ]
    ...
]
```

#### :hurtrealbad: *Recognition*
For recognition you need to have a dataset with the coders of faces grouped and labeled by person in json format. use the encoder to generate the dataset as explained in the previous section.

Once we have our dataset containing the codes for each face image for each person we can use recognition.
```sh
python videoAnalytics/processor.py recognition -i <img> -d <dataset> -s
```
the flag ```-s``` displays the image using opencv. you can add the flags ```-dd [0-N]``` to specify the device to run detection on and ```-rd [0-N]``` for recognition (by default -1 for cpu, 0 to N for GPU)
```sh
python videoAnalytics/processor.py recognition -i <img> -d <dataset> -s -dd 0 -rd 0
```
the flag ```-p``` prints the faces bounding box on the terminal if you want more information.
```sh
python videoAnalytics/processor.py recognition -i <img> -d <dataset> -s -dd 0 -rd 0 -p
```
To test on a live video use the flag ```-t```. NOTE: don't use the ```-s``` together with ```-t```
```sh
python videoAnalytics/processor.py recognition -t -d <dataset> -dd 0 -rd 0
```

#### *Recognition*

## :white_check_mark: Run API Service 

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

### :penguin: *Run from terminal*

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
### :whale2: *Run as Container*

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
