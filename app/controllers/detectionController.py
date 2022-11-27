from fastapi import APIRouter, Form, Request, Response, status, File, UploadFile

from app.libs.imageHelper import ImageHelper
from app.services.detectionService import DetectionService

router = APIRouter(prefix="/detector", tags=["Detector"])

@router.post("/image")
async def detect(request: Request, response: Response, file: UploadFile = File(...)):
    try:
        # Reading the image
        file_content = await file.read()

        img = ImageHelper.multipart_to_image(file_content)

        result = await DetectionService.detect_image(request.app.state.processor, img)
        return result
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}
