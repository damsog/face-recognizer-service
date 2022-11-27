from fastapi import APIRouter, Form, Request, Response, status, File, UploadFile

from app.libs.imageHelper import ImageHelper
from app.models.sdpDTO import PeerConnectionDTO
from app.services.detectionService import DetectionService

router = APIRouter(prefix="/detector", tags=["Detector"])

@router.post("/image")
async def detect(request: Request, response: Response, file: UploadFile = File(...), return_img_b64: bool = False):
    try:
        # Reading the image
        file_content = await file.read()

        img = ImageHelper.multipart_to_image(file_content)

        result = await DetectionService.detect_image(request.app.state.processor, img, return_img_b64)
        return result
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}

@router.post("/stream")
async def detect_stream(request: Request, response: Response, pc: PeerConnectionDTO):
    try:
        result = await DetectionService.detect_stream(request.app.state.processor, 
                                                    request.app.state.pcs, 
                                                    request.app.state.relay, 
                                                    pc.sdp, 
                                                    pc.sdp_type)
        return result
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}