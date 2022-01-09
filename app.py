import json
import os
from uuid import uuid4
from flask.globals import request
from dotenv import load_dotenv, find_dotenv
from videoAnalytics.processor import processor

from aiohttp import web
import asyncio
from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
import ssl
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
            _,img = self.processor.detect_image(img, return_img=True)
            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.process == "analyze":
            #TODO: Dont load the dataset for each image. its making it laggy.
            img = frame.to_ndarray(format="bgr24")
            _,img = self.processor.analyze_image('/mnt/72086E48086E0C03/Projects/VideoAnalytics_Server/resources/user_data/1/g1/g1embeddings.json',img, return_img=True)
            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
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
    SSL_CONTEXT = os.environ.get("SERVER_SSL_CONTEXT")
    SSL_KEYFILE = os.environ.get("SERVER_SSL_KEYFILE")
    ROOT = os.path.dirname(__file__)

    mProcessor = processor()

    logger = logging.getLogger("pc")

    logging.basicConfig(level=logging.INFO)

    pcs = set()
    relay = MediaRelay()

    #======================================================Requests============================================================
    async def load_models():
        result = "0"
        print("load_models")
        return result

    async def unload_models():
        result = "0"
        print("unload_models")
        return result

    async def encode_images():
        result = "0"
        print("encode_images")
        result = mProcessor.encode_images( str(request.get_json("imgs")).replace("'",'"') )
        return str(result).replace("'",'"')

    async def analyze_image(request):
        result = "0"
        print("analyze_image")
        #mProcessor.analyze_image(request.get_json("imgs"))
        datajson = await request.json()
        return_img_json = True if datajson["return_img"]==1 else False
        dataset_path = datajson["dataset_path"]
        img_b64 =  datajson["img"]

        result = mProcessor.analyze_image(dataset_path,mProcessor.b642cv2( img_b64 ),return_img_json=return_img_json)
        
        return str(result)

    async def facedet_stream(request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        pcs.add(pc)

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        log_info("Created for %s", request.remote)
        
        # prepare local media
        record = False
        if record:
            recorder = MediaRecorder("args.record_to")
        else:
            recorder = MediaBlackhole()

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)
        
        @pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)

            if track.kind == "video":
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
        
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )

    async def start_live_analytics():
        result = "0"
        print("start_live_analytics")
        return result

    async def stop_live_analytics():
        result = 0
        print("stop_live_analytics")
        return result
    
    async def on_shutdown(app):
        # close peer connections
        coros = [pc.close() for pc in pcs]
        await asyncio.gather(*coros)
        pcs.clear()

    #======================================================Start the Server====================================================

    if SSL_CONTEXT:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(SSL_CONTEXT, SSL_KEYFILE)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/load_models', load_models)
    app.router.add_get('/start_live_analytics', start_live_analytics)
    app.router.add_post('/facedet_stream', facedet_stream)
    app.router.add_get('/stop_live_analytics', stop_live_analytics)
    app.router.add_post('/analyze_image', analyze_image)
    app.router.add_get('/unload_models', unload_models)
    app.router.add_post('/encode_images', encode_images)
    web.run_app(
        app, access_log=None, host=HOST, port=PORT, ssl_context=ssl_context
    )

if __name__=="__main__":
    main()