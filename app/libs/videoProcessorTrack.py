from av import VideoFrame
from aiortc import MediaStreamTrack

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