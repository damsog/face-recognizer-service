import sys
from typing import List
import cv2
import base64
import argparse
import insightface
import numpy as np
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from videoAnalytics.encoder import OutputData, encoderExtractor
from videoAnalytics.analyzer import faceAnalyzer
from videoAnalytics.logger import Logger

class processor:
    def __init__(self, detector_load_device=0, recognizer_load_device=0,face_detection_model='retinaface_r50_v1', face_recognition_model='arcface_r100_v1') -> None:
        #loading the face detection model. detector_load_device 0 means to work with GPU. -1 is for CPU.
        detector = insightface.model_zoo.get_model(face_detection_model)
        detector.prepare(ctx_id = detector_load_device, nms=0.4)

        #loading the face recognition model. recognizer_load_device 0 means to work with GPU. -1 is for CPU.
        recognizer = insightface.model_zoo.get_model(face_recognition_model)
        recognizer.prepare(ctx_id = recognizer_load_device)

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

    def encode_narray_image(self, img: np.ndarray) -> List[float]:
        result = self.encoder.process_image( img )
        return result
    
    def encode_narray_images(self, imgs: List[np.ndarray], key: str) -> OutputData:
        result = self.encoder.process_images( imgs, key )
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
    
    def analyze_image_2(self, image, return_img=False, return_landmarks=False, return_img_json=False):
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
    def analyze_video(self, dataset_path, video_source, return_img=False, return_landmarks=False, return_img_json=False, show_video=False, print_output=False, logger=None):
        self.set_analyzer_dataset(dataset_path)
        cap = cv2.VideoCapture(video_source)

        img_id = 0
        while cap.isOpened():
            ret, img_read = cap.read()
            result,img = self.analyzer.analyze_img(img_read, True, return_landmarks, return_img_json)
            if print_output and logger:
                logger.info(f'frame_{img_id}: {result}')
            img_id += 1

            if show_video:
                cv2.namedWindow('Frame')
                cv2.imshow('Frame', img)

                fin = cv2.waitKey(1) & 0xFF
                if(fin == ord('q')):
                    break
    
    # TODO: Work on video feed to frontend
    def detect_video(self, video_source, return_img=False, return_landmarks=False, return_img_json=False, show_video=False, print_output=False, logger=None):
        cap = cv2.VideoCapture(video_source)

        img_id = 0
        while cap.isOpened():
            ret, img_read = cap.read()
            result,img = self.analyzer.detect_img(img_read, True, return_landmarks, return_img_json)
            if print_output and logger:
                logger.info(f'frame_{img_id}: {result}')
            img_id += 1

            if show_video:
                cv2.namedWindow('Frame')
                cv2.imshow('Frame', img)

                fin = cv2.waitKey(1) & 0xFF
                if(fin == ord('q')):
                    break

def main() -> None:
    # Importing these two libs only if this modules is called as a standalone script
    import pyfiglet
    import iridi

    ap = argparse.ArgumentParser()
    ap.add_argument("type", choices=["detection", "recognition"], help="Select detection for Plain FaceDet and Recognition for FaceDetection and Recognition.")
    ap.add_argument("-i", "--image", required=False, help="Reads an image from path. If not given, opens camera")
    ap.add_argument("-d", "--dataset", required=False, help="path to the dataset json file containing refference info")
    ap.add_argument("-t", "--video_test", action="store_true", help="If selected, opens camera to live test")
    ap.add_argument("-v", "--verbose", action="store_true", help="Debug level for logger output")
    ap.add_argument("-s", "--show_img", action="store_true", help="Shows image output")
    ap.add_argument("-p", "--print_output", action="store_true", help="Prints output to console")
    ap.add_argument("-dd", "--det-dev", required=False, help="Detection device to use. Default is CPU. -1 for CPU, 0-N for GPU id")
    ap.add_argument("-rd", "--rek-dev", required=False, help="Recognition device to use. Default is CPU. -1 for CPU, 0-N for GPU id")
    args = vars(ap.parse_args())

    # Cool title
    moduleTitle = pyfiglet.figlet_format("Face Processor", font="isometric2", width=200)
    iridi.print(moduleTitle, ["#8A2387", "#E94057", "#F27121"], bold=True)
    iridi.print("Loading face processor module as a standalone program", ["#8A2387", "#E94057", "#F27121"], bold=True)
    iridi.print("Prepare your mind to be blown!!!!", ["#8A2387", "#E94057", "#F27121"], bold=True)

    # Logger Configuration
    MODULE_NAME = "FACE ANALITICS PROCESSOR"
    if args["verbose"]:
        logger = Logger("DEBUG", COLORED=True, TAG_MODULE= MODULE_NAME)
    else:
        logger = Logger("INFO", COLORED=True, TAG_MODULE= MODULE_NAME)
    
    def print_output(label: str, result: any):
        if args['image']: logger.info(f'{args["image"]}: {result}')

    # Check if the user provided a valid device id
    try:
        detection_using_cpu = int(args["det_dev"]) if args["det_dev"] else -1
        rekognition_using_cpu = int(args["rek_dev"]) if args["rek_dev"] else -1
    except ValueError:
        logger.error("Invalid device id. Must be an integer")
        sys.exit()

    logger.info("Using CPU for detection") if detection_using_cpu == -1 else logger.info(f"Using GPU {detection_using_cpu} for detection")
    logger.info("Using CPU for recognition") if rekognition_using_cpu == -1 else logger.info(f"Using GPU {rekognition_using_cpu} for recognition")

    if not args["dataset"]:
       if args["type"] == "recognition":
           logger.error("For recognition you must provide a dataset. Use the flag: -d /path/to/dataset")
           sys.exit()

    logger.info("Preparing the models")
    face_detection_model = 'retinaface_r50_v1'
    face_recognition_model = 'arcface_r100_v1'
    logger.debug(f'Face detection model: {face_detection_model}')
    logger.debug(f'Face recognition model: {face_recognition_model}')
    mProcessor = processor(
        detection_using_cpu,
        rekognition_using_cpu,
        face_detection_model,
        face_recognition_model
    )

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

    # Video Recognition Example
    # python videoAnalytics/processor.py recognition -d 'embeddings.json' -t
    # Image Recognition Example
    # python videoAnalytics/processor.py recognition -i 'img.jpg' -d 'embeddings.json' -s

    if args["type"] == "recognition":
        logger.debug("Performing face recognition on image")
        result,img = mProcessor.analyze_image(dataset_path, input_img, return_img=True)
    else:
        logger.debug("Performing face detection on image")
        result,img = mProcessor.detect_image(input_img, return_img=True)

    if args["print_output"]:
        logger.info("Printing output to console")
        if args['image']: print_output(args['image'], result)

    if args["show_img"]:
        logger.info("Image processed will be shown")
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Video Testing if option selected. stop it pressing q
    if args["video_test"]:
        logger.info("Starting video test")
        print_video_output = True if args["print_output"] else False
        if args["type"] == "recognition":
            mProcessor.analyze_video(dataset_path, 0, show_video=True, print_output=print_video_output, logger=logger)
        else:
            mProcessor.detect_video(0, show_video=True, print_output=print_video_output, logger=logger)

if __name__=="__main__":
    main()