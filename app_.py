import json
import os
import time
import insightface
from flask import Flask, Request, Response
from uuid import uuid4
from flask.globals import request
from dotenv import load_dotenv, find_dotenv
from videoAnalytics.processor import processor

from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
import uuid
import logging

class VideoProcessorTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, process, processor):
        super().__init__()  # don't forget this!
        self.track = track
        self.process = process
        self.processor = processor

    async def recv(self):
        frame = await self.track.recv()

        if self.process == "detect":
            img = frame.to_ndarray(format="bgr24")
            #new_frame = self.processor.detect_image(img, return_img=True)
            
            return img
        elif self.process == "analyze":
            #TODO: handle analyzer
            img = frame.to_ndarray(format="bgr24")
            return img
        else:
            return frame

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

    ROOT = os.path.dirname(__file__)

    # Creating our server
    app = Flask(__name__)
    mProcessor = processor()

    logger = logging.getLogger("pc")
    logging.basicConfig(level=logging.INFO)

    pcs = set()
    relay = MediaRelay()

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
        datajson = request.get_json()
        return_img_json = True if datajson["return_img"]==1 else False
        dataset_path = datajson["dataset_path"]
        img_b64 =  datajson["img"]

        result = mProcessor.analyze_image(dataset_path,mProcessor.b642cv2( img_b64 ),return_img_json=return_img_json)
        
        return str(result)

    @app.route('/facedet_stream', methods=['POST'])
    async def facedet_stream():
        result = "0"
        params = request.get_json()
        #print(params)
        
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        pcs.add(pc)

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        #log_info("Created for %s", params.remote)
        
        # prepare local media
        player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
        record = False
        if record:
            recorder = MediaRecorder("args.record_to")
        else:
            recorder = MediaBlackhole()

        @pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            def on_message(message):
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send("pong" + message[4:])

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)
        
        @pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)

            if track.kind == "audio":
                pc.addTrack(player.audio)
                recorder.addTrack(track)
            elif track.kind == "video":
                pc.addTrack(
                    VideoProcessorTrack(
                        relay.subscribe(track), process="detect", processor=mProcessor
                    )
                )
                if record:
                    recorder.addTrack(relay.subscribe(track))

            @track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)
                await recorder.stop()

        # handle offer
        await pc.setRemoteDescription(offer)
        await recorder.start()

            # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        print(json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ))
        
        return Response(
            mimetype="application/json",
            response=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )

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