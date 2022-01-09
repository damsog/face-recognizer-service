import sys
import cv2
import base64
import argparse
import insightface
import numpy as np
import logging
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from videoAnalytics.encoder import encoderExtractor
from videoAnalytics.analyzer import faceAnalyzer

class processor:
    def __init__(self, face_detection_model='retinaface_r50_v1', face_recognition_model='arcface_r100_v1') -> None:
        #loading the face detection model. 0 means to work with GPU. -1 is for CPU.
        detector = insightface.model_zoo.get_model(face_detection_model)
        detector.prepare(ctx_id = 0, nms=0.4)

        #loading the face recognition model. 0 means to work with GPU. -1 is for CPU.
        recognizer = insightface.model_zoo.get_model(face_recognition_model)
        recognizer.prepare(ctx_id = 0)

        self.encoder = encoderExtractor(None, detector, recognizer)
        self.analyzer = faceAnalyzer(detector, recognizer, None)

    def b642cv2(self, in_img_b64):
        try:
            img_b64 = in_img_b64.split(',')[1]
        except:
            img_b64 = in_img_b64
        nparr = np.fromstring(base64.b64decode(img_b64), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def encode_images(self, input_data):
        self.encoder.set_input_data( input_data )
        result = self.encoder.process_data()
        return result
    
    def set_analyzer_dataset(self, dataset_path):
        self.analyzer.set_dataset(dataset_path)

    def analyze_image(self, dataset_path, image, return_img=False, return_landmarks=False, return_img_json=False):
        self.set_analyzer_dataset(dataset_path)
        if return_img:
            result,img = self.analyzer.analyze_img(image, return_img, return_landmarks, return_img_json)
            return result, img
        else:
            result = self.analyzer.analyze_img(image, return_img, return_landmarks, return_img_json)
            return result
    
    def detect_image(self, image, return_img=False, return_landmarks=False, return_img_json=False):
        if return_img:
            result,img = self.analyzer.detect_img(image, return_img, return_landmarks, return_img_json)
            return result, img
        else:
            result = self.analyzer.detect_img(image, return_img, return_landmarks, return_img_json)
            return result

    # TODO: Work on video feed to frontend
    def analyze_video(self, dataset_path, video_source, return_img=False, return_landmarks=False, return_img_json=False, show_video=False):
        self.set_analyzer_dataset(dataset_path)
        cap = cv2.VideoCapture(video_source)

        while cap.isOpened():
            ret, img_read = cap.read()
            result,img = self.analyzer.analyze_img(img_read, True, return_landmarks, return_img_json)

            if show_video:
                cv2.namedWindow('Frame')
                cv2.imshow('Frame', img)

                fin = cv2.waitKey(1) & 0xFF
                if(fin == ord('q')):
                    break
    
    # TODO: Work on video feed to frontend
    def detect_video(self, video_source, return_img=False, return_landmarks=False, return_img_json=False, show_video=False):
        cap = cv2.VideoCapture(video_source)

        while cap.isOpened():
            ret, img_read = cap.read()
            result,img = self.analyzer.detect_img(img_read, True, return_landmarks, return_img_json)

            if show_video:
                cv2.namedWindow('Frame')
                cv2.imshow('Frame', img)

                fin = cv2.waitKey(1) & 0xFF
                if(fin == ord('q')):
                    break

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("type", choices=["detection", "recognition"], help="Select detection for Plain FaceDet and Recognition for FaceDetection and Recognition.")
    ap.add_argument("-i", "--image", required=False, help="Reads an image from path. If not given, opens camera")
    ap.add_argument("-d", "--dataset", required=False, help="path to the dataset json file containing refference info")
    ap.add_argument("-t", "--video_test", action="store_true", help="If selected, opens camera to live test")
    ap.add_argument("-v", "--verbose", action="store_true", help="Debug level for logger output")
    ap.add_argument("-s", "--show_img", action="store_true", help="Shows image output")
    args = vars(ap.parse_args())

    # Logger Configuration
    logger = logging.getLogger(__name__)
    logger_format = '%(asctime)s:%(name)s:%(levelname)s:%(message)s'
    logger_date_format = '[%Y/%m/%d %H:%M:%S]'

    if args["verbose"]:
        logging.basicConfig(level=logging.DEBUG, format=logger_format, datefmt=logger_date_format)
    else:
        logging.basicConfig(level=logging.INFO,  format=logger_format, datefmt=logger_date_format)

    if not args["dataset"]:
       if args["type"] == "recognition":
           logger.info("For recognition you must provide a dataset. Use the flag: -d /path/to/dataset")
           sys.exit()

    logger.info("Preparing the models")
    face_detection_model = 'retinaface_r50_v1'
    face_recognition_model = 'arcface_r100_v1'
    logger.debug(f'Face detection model: {face_detection_model}')
    logger.debug(f'Face recognition model: {face_recognition_model}')
    mProcessor = processor(face_detection_model,face_recognition_model)

    # Getting input image. -i to get it from path. else get it from camera
    if args["image"]:
        logger.info("Reading image from path")
        input_img = cv2.imread(args['image'])
    else:
        logger.info("Image not provided. reading camera instead")
        cap = cv2.VideoCapture(0)
        ret, input_img = cap.read()
        cap.release()
        pass
    
    logger.info(f'Reading dataset from path {args["dataset"]}')
    dataset_path = args["dataset"]

    #input_img = cv2.imread('/mnt/72086E48086E0C03/Projects/Face_Recognizer_Service/imgs/00000002.jpg')
    #-d '/mnt/72086E48086E0C03/Projects/VideoAnalytics_Server/resources/user_data/1/g1/g1embeddings.json' -t

    if args["type"] == "recognition":
        logger.debug("Performing face recognition on image")
        result,img = mProcessor.analyze_image(dataset_path, input_img, return_img=True)
    else:
        logger.debug("Performing face detection on image")
        result,img = mProcessor.detect_image(input_img, return_img=True)

    if args["show_img"]:
        logger.info("Image processed will be shown")
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Video Testing if option selected. stop it pressing q
    if args["video_test"]:
        logger.info("Starting video test")
        if args["type"] == "recognition":
            mProcessor.analyze_video(dataset_path, 0, show_video=True)
        else:
            mProcessor.detect_video(0, show_video=True )

if __name__=="__main__":
    main()