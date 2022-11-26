from typing import List
import numpy as np

from videoAnalytics.encoder import OutputData
from videoAnalytics.processor import processor

class EncoderService:
    @staticmethod
    async def encode_image(processor: processor, img: np.ndarray) -> List[float]:
        result = processor.encode_narray_image(img)
        return result

    @staticmethod
    async def encode_images(processor: processor, imgs: List[np.ndarray], key: str) -> OutputData:
        result = processor.encode_narray_images(imgs, key)
        return result