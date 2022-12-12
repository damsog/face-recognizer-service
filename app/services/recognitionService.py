

from typing import List
from numpy import ndarray
from videoAnalytics.logger import Logger

from videoAnalytics.processor import processor

class RecognitionService:
    @staticmethod
    async def recognize_image(g_processor: processor, img: ndarray, dataset: List[List[str]], return_img_b64: bool, logger: Logger = None) -> List[float]:
        def log(message: str):
            if logger is not None: logger.info(message)
        
        log("Processing image. Applying Face detection")
        result = g_processor.analyze_image(dataset, img, return_img_json=return_img_b64)
        return result