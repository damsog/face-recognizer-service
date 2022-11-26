from typing import List
import cv2
from fastapi import APIRouter, Form, Request, Response, status, File, UploadFile

from app.libs.imageHelper import ImageHelper
from app.services.encoderService import EncoderService
import numpy as np

from videoAnalytics.encoder import OutputData

router = APIRouter(prefix="/encoder", tags=["Encoder"])

@router.post("/image")
async def encode(request: Request, response: Response, file: UploadFile = File(...)) -> List[float]:
    try:
        # Reading the image
        file_content = await file.read()

        img = ImageHelper.multipart_to_image(file_content)

        result = await EncoderService.encode_image(request.app.state.processor, img)
        return result
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}

@router.post("/images")
async def encode(request: Request, response: Response, key: str = Form(...), files: List[UploadFile] = File(...)) -> OutputData:
    try:
        # Reading the images
        files_content = []
        for file in files:
            file_content = await file.read()
            files_content.append(file_content)

        imgs = ImageHelper.multipart_to_image_list(files_content)

        result = await EncoderService.encode_images(request.app.state.processor, imgs, key)
        return result
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}