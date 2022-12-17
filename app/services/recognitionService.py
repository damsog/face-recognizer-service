from typing import List
import uuid

from numpy import ndarray
from app.libs.videoProcessorTrack import VideoProcessorTrack
from app.models.sdpDTO import PeerConnectionDTO
from videoAnalytics.logger import Logger
from videoAnalytics.processor import processor
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRecorder, MediaRelay

class RecognitionService:
    @staticmethod
    async def recognize_image(g_processor: processor, img: ndarray, dataset: List[List[str]], return_img_b64: bool, logger: Logger = None) -> List[float]:
        def log(message: str):
            if logger is not None: logger.info(message)
        
        log("Processing image. Applying Face detection")
        result = g_processor.analyze_image(dataset, img, return_img_json=return_img_b64)
        return result
    
    @staticmethod
    async def recognize_stream(g_processor: processor, 
                               g_pcs: List[PeerConnectionDTO],
                               g_relay: MediaRelay, 
                               sdp: str,   
                               type: str,
                               dataset: List[List[str]],
                               record: bool= False,
                               logger: Logger = None
                               ) -> List[float]:
        def log(message: str):
            if logger is not None: logger.info(message)

        offer = RTCSessionDescription(sdp=sdp, type=type)

        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        g_pcs.add(pc)

        if record:
            recorder = MediaRecorder("args.record_to")
        else:
            recorder = MediaBlackhole()
        
        # Settting up events

        # When peerconnection changes
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log(f"{pc_id} connection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                g_pcs.discard(pc)
        
        # When a track is received
        @pc.on("track")
        def on_track(track):
            log(f"{pc_id} received {track.kind}")

            if track.kind == "video":
                pc.addTrack(
                    VideoProcessorTrack(
                        g_relay.subscribe(track),
                        process="recognition",
                        processor=g_processor,
                        dataset_path=dataset
                    )
                )
                if record:
                    recorder.addTrack(g_relay.subscribe(track))
        
            @track.on("ended")
            async def on_ended():
                log(f"{pc_id} track {track.kind} ended")
                await recorder.stop()

        # Handle offer
        await pc.setRemoteDescription(offer)
        await recorder.start()

        # Sending Answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return { "sdp": pc.localDescription.sdp , "type": pc.localDescription.type }