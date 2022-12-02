PYTHON:=python3.7
VENV:=venv

all: venv

setup-env: 
	sudo add-apt-repository ppa:deadsnakes/ppa
	sudo apt install $(PYTHON)
	$(PYTHON) -m pip install virtualenv

./$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m virtualenv $(VENV)
	./$(VENV)/bin/pip install --upgrade pip
	./$(VENV)/bin/pip install -r requirements.txt

venv: ./$(VENV)/bin/activate

run:
	./$(VENV)/bin/$(PYTHON) run.py

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete

.PHONY: setup-env clean venv run