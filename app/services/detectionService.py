

from typing import List

from numpy import ndarray
from videoAnalytics.processor import processor


class DetectionService:
    @staticmethod
    async def detect_image(processor: processor, img: ndarray, return_img_b64: bool) -> List[float]:
        result = processor.detect_image(img, return_img_json=return_img_b64)
        return result
    