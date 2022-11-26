from typing import List
import cv2
import numpy as np


class ImageHelper:  
    @staticmethod
    def multipart_to_image(content: bytes) -> np.ndarray:
        nparr = np.fromstring(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    
    @staticmethod
    def multipart_to_image_list(content_list: List[bytes]) -> List[np.ndarray]:
        imgs = []
        for content in content_list:
            imgs.append(ImageHelper.multipart_to_image(content))
        return imgs