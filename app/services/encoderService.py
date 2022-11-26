from typing import List
import numpy as np

class EncoderService:
    @staticmethod
    async def encode(processor, img: np.ndarray) -> List[float]:
        result = processor.encode_narray_image(img)
        return result