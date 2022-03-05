sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7
#python3 -m pip install virtualenv
virtualenv face_videoanalytics_venv --python=python3.7
source face_videoanalytics_venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m pip install python-dotenv
python3 -m pip install mxnet-cu102==1.7.0
python3 -m pip install cython
python3 -m pip install opencv-contrib-python
python3 -m pip install onnxruntime
python3 -m pip install insightface==0.1.5
python3 -m pip install imutils
python3 -m pip install uuid
python3 -m pip install av
python3 -m pip install aiortc
python3 -m pip install aiohttp
python3 -m pip install asyncio
python3 -m pip install pyfiglet
python3 -m pip install colorful
python3 -m pip install iridi
