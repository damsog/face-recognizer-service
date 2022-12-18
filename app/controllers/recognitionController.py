import json
from fastapi import APIRouter, Form, Request, Response, status, File, UploadFile

from app.libs.imageHelper import ImageHelper
from app.models.sdpDTO import PeerConnectionDTO
from app.services.recognitionService import RecognitionService

router = APIRouter(prefix="/recognizer", tags=["Recognizer"])

@router.post("/image")
async def detect(request: Request, response: Response, file: UploadFile = File(...), dataset: UploadFile = File(...) , return_img_b64: bool = False):
    try:
        # Reading the image
        file_content = await file.read()
        dataset_content = await dataset.read()
        dataset_json = json.loads(dataset_content.decode('utf-8'))

        img = ImageHelper.multipart_to_image(file_content)

        result = await RecognitionService.recognize_image(request.app.state.processor, img, dataset_json, return_img_b64)
        return result
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}

@router.post("/stream")
async def recognize_stream(request: Request, response: Response, sdp: str = Form(...), sdp_type: str = Form(...), dataset: UploadFile = File(...) ):
    try:
        dataset_content = await dataset.read()
        dataset_json = json.loads(dataset_content.decode('utf-8'))
        
        result = await RecognitionService.recognize_stream(request.app.state.processor, 
                                                    request.app.state.pcs, 
                                                    request.app.state.relay, 
                                                    sdp, 
                                                    sdp_type,
                                                    dataset_json)
        return result
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}