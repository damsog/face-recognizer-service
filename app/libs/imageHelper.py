import cv2
import numpy as np


class ImageHelper:  
    @staticmethod
    def multipart_to_image(multipart):
        nparr = np.fromstring(multipart, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img