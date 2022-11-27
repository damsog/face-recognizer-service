

from typing import List

from numpy import ndarray
from videoAnalytics.processor import processor


class DetectionService:
    @staticmethod
    async def detect_image(processor: processor, img: ndarray) -> List[float]:
        result = processor.detect_image(img, return_img_json=True)
        return result
    