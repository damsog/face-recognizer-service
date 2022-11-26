from typing import List
import cv2
from fastapi import APIRouter, Form, Request, Response, status, File, UploadFile

from ..libs.imageHelper import ImageHelper
from ..services.encoderService import EncoderService
import numpy as np

router = APIRouter(prefix="/encoder", tags=["Encoder"])

@router.post("/image")
async def encode(request: Request, response: Response, file: UploadFile = File(...)):
    try:
        # Reading the image
        contents = await file.read()

        img = ImageHelper.multipart_to_image(contents)

        result = await EncoderService.encode(request.app.state.processor, img)
        return result
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}

@router.post("/images")
async def encode(request: Request, response: Response, key: str = Form(...), file: List[UploadFile] = File(...)):
    try:
        # Reading the image
        contents = await file.read()

        img = ImageHelper.multipart_to_image(contents)

        result = await EncoderService.encode(request.app.state.processor, img)
        return result
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}