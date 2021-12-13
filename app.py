import json
import os
import time
import insightface
from flask import Flask, Request, Response
from uuid import uuid4
from flask.globals import request
from dotenv import load_dotenv, find_dotenv
from videoAnalytics.processor import processor


def main():
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

    mProcessor = processor()

    #======================================================Requests============================================================
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
        result = mProcessor.encode_images( str(request.get_json("imgs")).replace("'",'"') )
        return str(result).replace("'",'"')

    @app.route('/analyze_image', methods=['POST'])
    def analyze_image():
        result = "0"
        print("analyze_image")
        #mProcessor.analyze_image(request.get_json("imgs"))
        print(request.get_json())
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

    #======================================================Start the Server====================================================
    app.run(host=HOST, port=PORT)

if __name__=="__main__":
    main()