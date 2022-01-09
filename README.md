# Face_Recognizer_service

### install virtual env first.
### Python should be >=3.7. this is due to flask async support.
### python3 -m pip install virtualenv
### create a virtualenv like
### virtualenv /path/to/venv
### then activate with 
### source /path/to/venv/bin/activate
### 
### to install packaged use python3.x -m pip install package
### 
### for ease of use just create it inside this directory with an appropriate name,
### it creates its own .gitignore so don't worry about that.
### 
### Install Cuda, cuda drivers, and cudnn. and pay attention to which version, because cuda 
### su** a** and with every new release they make every dependency that uses cuda breaks.
### I had to reinstall cuda like a million times trying to juggle which version
### fits for my packages.
### 
### Anyway. match your cuda version to mxnet. for example for cuda 10.2 (which is the one I made work)
### use mxnet-cu102==1.7.0
### 
### Before installing insightface make sure that you have installed the appropriate python3.x-dev version 
### corresponding to your python
### 
### then install insightface. the current version of insightface wont work with this code. so install
### insightface==0.1.5
### 
### the models for this version should be downloaded manually from :
### https://onedrive.live.com/?authkey=%21ALRWS0dNaAhDRuc&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215836&parId=4A83B6B633B029CC%215834&action=locate
### and then placed on ~/.insightface/models