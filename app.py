import encoder_extractor
import json, os, time
from flask import Flask, Request, Response
from uuid import uuid4
from flask.globals import request
from dotenv import load_dotenv, find_dotenv

#initializations
load_dotenv(find_dotenv())

# Some definitions
NET = None
METADATA = None
MAX_SIMULTANEOUS_PROCESSES = 1
NUM_SESSIONS = 0
HOST = os.environ.get("SERVER_IP")
PORT = os.environ.get("SERVER_PORT")


# Creating our server
app = Flask(__name__)

#======================================================Requests============================================================

# Creating our face detection and recognition sistem.
encoder = encoder_extractor.encoderExtractor(None)

@app.route('/load_models', methods=['GET'])
def load_models():
    result = "0"
    print("load_models")
    return result

@app.route('/unload_models', methods=['GET'])
def unload_models():
    result = "0"
    print("unload_models")
    return result

@app.route('/encode_images', methods=['POST'])
def encode_images():
    result = "0"
    print("encode_images")
    encoder.set_input_data( str(request.get_json("imgs")).replace("'",'"') )
    result = encoder.process_data()
    
    return str(result).replace("'",'"')

@app.route('/compare_to_dataset', methods=['POST'])
def compare_to_dataset():
    result = "0"
    print("compare_to_dataset")
    return result

@app.route('/start_live_analytics', methods=['GET'])
def start_live_analytics():
    result = "0"
    print("start_live_analytics")
    return result

@app.route('/stop_live_analytics', methods=['GET'])
def stop_live_analytics():
    result = 0
    print("stop_live_analytics")
    return result

if __name__=="__main__":
    print(MAX_SIMULTANEOUS_PROCESSES)

    app.run(host=HOST, port=PORT)