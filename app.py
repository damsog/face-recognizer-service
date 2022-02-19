# 
# @author Felipe Serna
# @email damsog38@gmail.com
# @create date 2021-14-06 20:01:10
# @modify date 2022-02-13 14:19:39
# @desc API that serves Analytics Processes. Face Detection and Recognition for both
# single images as well as live processes. This app imports and calls a process object
# which loads the detection and recognition models on gpu memory and then can be used to initiate.
# 

#*************************************************************************************************
#                                              Dependencies
#*************************************************************************************************

# Importing configurations & Global Variables
from dotenv import load_dotenv, find_dotenv

# API Server dependencies
from aiohttp import web
import asyncio
from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
import ssl

# Other useful dependencies
import json
import os
import uuid

# Analytics personal Library
from videoAnalytics.processor import processor

# Logging, Terminal Styles & Colors Dependencies
import logging
import iridi
import pyfiglet
import colorful

#*************************************************************************************************
#                                              Definitions
#*************************************************************************************************
class VideoProcessorTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, process, processor, dataset_path=None):
        super().__init__()  # don't forget this!
        self.track = track
        self.process = process
        self.processor = processor
        if dataset_path:
            self.processor.set_analyzer_dataset(dataset_path)

    async def recv(self):
        frame = await self.track.recv()

        if self.process == "detection":
            img = frame.to_ndarray(format="bgr24")
            _,img = self.processor.detect_image(img, return_img=True)
            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.process == "recognition":
            #TODO: Dont load the dataset for each image. its making it laggy.
            img = frame.to_ndarray(format="bgr24")
            _,img = self.processor.analyze_image_2(img, return_img=True)
            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            return frame

#*************************************************************************************************
#                                              Main Function
#*************************************************************************************************
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
    LOGGER_LEVEL = os.environ.get("LOGGER_LEVEL")

    # Server Api title definition
    serverTitle = pyfiglet.figlet_format("Gnosis", font="isometric2", width=200)
    serverSubtitle = pyfiglet.figlet_format("Face Analytics Server", font="alligator2", width=300)

    logger = logging.getLogger(__name__)
    logger_format = '%(asctime)s : : %(levelname)s : : %(name)s : : %(message)s'
    logger_date_format = '[%Y:%m:%d %H:%M:%S %Z]'

    if LOGGER_LEVEL == "DEBUG":
        logging.basicConfig(level=logging.DEBUG, format=logger_format, datefmt=logger_date_format)
    else:
        logging.basicConfig(level=logging.INFO,  format=logger_format, datefmt=logger_date_format)

    iridi.print(serverTitle, ["#8A2387", "#E94057", "#F27121"], bold=True)
    iridi.print(serverSubtitle, ["#8A2387", "#E94057", "#F27121"], bold=True)

    mProcessor = processor()
    pcs = set()
    relay = MediaRelay()
    #======================================================Requests============================================================

    async def encode_images(request):
        logger.info("Encode images Requested")
        datajson = await request.json()

        logger.debug(f'Data received: {datajson}')

        logger.debug(f'Processing images: {datajson["imgs"]}')
        result = mProcessor.encode_images( str(datajson).replace("'",'"') )

        return web.Response(
            content_type="application/json",
            text=str(result).replace("'",'"')
        )

    async def analyze_image(request):
        logger.info("Analyze image requested")

        datajson = await request.json()
        logger.debug(f'Information: {datajson}')

        return_img_json = True if datajson["return_img"]==1 else False
        logger.debug(f'Return img as b64: {return_img_json}')

        dataset_path = datajson["dataset_path"]
        logger.debug(f'Dataset : {dataset_path}')

        img_b64 =  datajson["img"]

        logger.debug('Processing Image')
        result = mProcessor.analyze_image(dataset_path,mProcessor.b642cv2( img_b64 ),return_img_json=return_img_json)
        
        return web.Response( 
            content_type = "application/json",
            text = str(result)
        )
    
    async def detect_image(request):
        logger.info('Detect image requested')

        datajson = await request.json()
        logger.debug(f'Information: {datajson}')

        return_img_json = True if datajson["return_img"] == 1 else False
        logger.debug(f'Return img as b64: {return_img_json}')

        img_b64 = datajson["img"]

        logger.debug('Detecting Image')
        result = mProcessor.detect_image(mProcessor.b642cv2( img_b64 ), return_img_json=return_img_json )

        return web.Response(
            content_type = "application/json",
            text = str(result)
        )

    async def facedet_stream(request):
        params = await request.json()
        logger.info("Face detection stream requested")
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        pcs.add(pc)

        def log_info(msg, *args):
            logger.info( colorful.cyan( pc_id + " " + msg ), *args )

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
                        relay.subscribe(track), process="detection", processor=mProcessor
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
        
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )

    async def facerek_stream(request):
        params = await request.json()
        dataset_path = params["dataset_path"]
        logger.info(f'Face recognition stream requested. using group: {dataset_path}')
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
                        relay.subscribe(track), process="recognition", processor=mProcessor, dataset_path=dataset_path
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
        
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )
    
    async def on_shutdown(app):
        # close peer connections
        coros = [pc.close() for pc in pcs]
        await asyncio.gather(*coros)
        pcs.clear()

    if SSL_CONTEXT:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(SSL_CONTEXT, SSL_KEYFILE)
    else:
        ssl_context = None

    # Setting API Enpoints
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post('/facedet_stream', facedet_stream)
    app.router.add_post('/facerek_stream', facerek_stream)
    app.router.add_post('/analyze_image', analyze_image)
    app.router.add_post('/detect_image', detect_image)
    app.router.add_post('/encode_images', encode_images)

    # Running the server
    web.run_app(
        app, access_log=None, host=HOST, port=PORT, ssl_context=ssl_context
    )

#*************************************************************************************************
#                                              Running
#*************************************************************************************************
if __name__=="__main__":
    main()