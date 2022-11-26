sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7
python3.7 -m pip install virtualenv
python3.7 -m virtualenv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
