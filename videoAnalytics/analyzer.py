import argparse
import base64
import json
import imutils
import insightface
import cv2
import sys
import logging
import numpy as np
from mxnet.base import _NullType
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from videoAnalytics.utils import scaller_conc, true_match
from videoAnalytics.logger import Logger


# This class is used to Receive an image and a dataset, detect all faces on said image, then 
# get a code for each image, then compare it to the dataset to determine if matches a known 
# face or not.
# the dataset is a json file which containes, the label of the person and the embedding of the
# face.
class faceAnalyzer:
    def __init__(self, detector, recognizer, dataset_path, input_img = None) -> None:
        self.img = input_img
        self.dataset_path = dataset_path
        self.detector = detector
        self.recognizer = recognizer
        self.json_output = {}

        if dataset_path:
            # Reading Dataset embeddings
            self.dataset_embeddings, self.dataset_names = self.parse_dataset_json(self.dataset_path)

            # getting labels
            _,idx = np.unique(np.asarray(self.dataset_names), return_index=True)
            self.labels = np.asarray(self.dataset_names)[np.sort(idx)]
        else:
            self.dataset_embeddings, self.dataset_names, self.labels, self.dataset_unames = None, None, None, None
    

    # this method reads the json which containes the codes for a dataset and
    # puts information on a list to be used by the analyzer
    def parse_dataset_json(self, dataset_path):
        with open(dataset_path) as f:
            dataset = json.load(f)

        embeddings_array = np.zeros( (1,512) )
        names_list = []

        # Each row is a face code. just transform the code as tring back into float and
        # concatenate it with its face label (person id, or label)
        for data in dataset:
            embedding = np.array( [float(value) for value in data[1].split(',')] )
            embeddings_array = np.row_stack(( embeddings_array, embedding ))
            names_list.append(str(data[0])) 
        
        embeddings_array = np.delete(embeddings_array , 0, 0)

        # getting uniques
        _,idx = np.unique(np.asarray(names_list), return_index=True)
        self.dataset_unames = np.asarray(names_list)[np.sort(idx)]

        names_list = ['Uknown'] + names_list
        return embeddings_array, names_list

    def set_input_img(self, input_img):
        self.img = input_img
    
    def get_input_img(self):
        return self.img

    def set_dataset(self, dataset_path):
        self.dataset_path = dataset_path

        # Reading Dataset embeddings
        self.dataset_embeddings, self.dataset_names = self.parse_dataset_json(self.dataset_path)

        # getting labels
        _,idx = np.unique(np.asarray(self.dataset_names), return_index=True)
        self.labels = np.asarray(self.dataset_names)[np.sort(idx)]
    
    def get_dataset_path(self):
        return self.dataset_path
    
    def get_dataset(self):
        return self.dataset_embeddings, self.dataset_names, self.labels

    def get_output_data(self):
        return self.json_output
    
    def img2b64(self, img):
        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)
        return jpg_as_text
        
    def analyze_img(self, img, return_img=False, return_landmarks=False, return_img_json=False):

        # Initializations
        self.img = img
        self.json_output = {}
        output_array = []
        faces = []
        embeddings = []
        WIDTHDIVIDER = 1

        img = imutils.resize(img, width=int(img.shape[1]/WIDTHDIVIDER))
        bboxs, landmarks = self.detector.detect(img, threshold=0.5, scale=1.0)

        if bboxs is None: return None

        # Cleaning some data. some faces that are kind of outside the area
        # TODO: extract this as a function
        todel = []
        for i in range(bboxs.shape[0]):
            if(any(x<0 for x in bboxs[i])):
                todel.append(i)
        for i in todel:
            bboxs = np.delete(bboxs, i, 0)

        # Processing the faces detected. Drawing the bboxes, landmarks and cutting the faces
        # to pass to the recognizer
        # TODO: Extract this as a function
        for bbox, landmark in zip(bboxs, landmarks):
            # Cutting each Face
            face = scaller_conc( img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]),:] )
            # Drawing the bboxes
            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0, 255, 0), 1)
            # Drawing landmarks
            for cord in landmark:
                cv2.circle(img, (int(cord[0]),int(cord[1])), 3, (0, 0, 255), -1)
            faces.append( face )

        embeddings = np.zeros( (1,512) )

        # Processing faces for recognition                        
        if faces:
            # Gets embedding for each face
            for face in faces:
                if face is None: continue
                embeddings = np.row_stack(( embeddings,self.recognizer.get_embedding(face)  ))
            embeddings = np.delete(embeddings, 0 , 0 )

            
            # Now process the embeddings for each face to find matches
            if(embeddings is not None):
                # TODO: Check the true match function which is working kind of weird for profiles with few images
                matches = true_match(embeddings,self.dataset_embeddings, self.dataset_names, self.dataset_unames, 0.3)  #0.5
                
                # Generating final output. iterating through the faces and getting their info
                for indx, (bbox, landmark) in enumerate(zip(bboxs, landmarks)):
                    face_json = { "label" : self.labels[matches[indx]], "bbox" : bbox }
                    if return_landmarks:
                        face_json["landmarks"] = landmark
                    cv2.putText(img, self.labels[matches[indx]], (int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    output_array.append(face_json)
        
        # Output Json
        self.json_output = { "faces" : output_array }
        if return_img_json:
            self.json_output["img_b64"] = self.img2b64(img)

        if return_img:
            return self.json_output, img
        else:
            return self.json_output

    def detect_img(self, img, return_img=False, return_landmarks=False, return_img_json=False):

        # Initializations
        self.img = img
        self.json_output = {}
        output_array = []
        WIDTHDIVIDER = 1

        img = imutils.resize(img, width=int(img.shape[1]/WIDTHDIVIDER))
        bboxs, landmarks = self.detector.detect(img, threshold=0.5, scale=1.0)
        if( bboxs is not None):

            # Processing the faces detected. Drawing the bboxes, landmarks and cutting the faces
            # TODO: Extract this as a function
            for idx,(bbox, landmark) in enumerate(zip(bboxs, landmarks)):
                # Drawing the bboxes
                cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0, 255, 0), 1)
                # Drawing landmarks
                for cord in landmark:
                    cv2.circle(img, (int(cord[0]),int(cord[1])), 3, (0, 0, 255), -1)
                
                face_json = { "label" : idx, "bbox" : bbox.tolist() }
                if return_landmarks:
                    face_json["landmarks"] = landmark
                cv2.putText(img, str(idx), (int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                output_array.append(face_json)

        # Output Json
        self.json_output = { "faces" : output_array }
        if return_img_json:
            self.json_output["img_b64"] = self.img2b64(img)
            
        if return_img:
            return self.json_output, img
        else:
            return self.json_output

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
    args = vars(ap.parse_args())

    # Cool title
    moduleTitle = pyfiglet.figlet_format("Face Analyzer", font="isometric2", width=200)
    iridi.print(moduleTitle, ["#8A2387", "#E94057", "#F27121"], bold=True)
    iridi.print("Loading face Analyzer module as a standalone program", ["#8A2387", "#E94057", "#F27121"], bold=True)
    iridi.print("Prepare your mind to be blown!!!!", ["#8A2387", "#E94057", "#F27121"], bold=True)

    # Logger Configuration
    MODULE_NAME = "FACE ANALYZER"
    if args["verbose"]:
        logger = Logger("DEBUG", COLORED=True, TAG_MODULE= MODULE_NAME)
    else:
        logger = Logger("INFO", COLORED=True, TAG_MODULE= MODULE_NAME)

    if not args["dataset"]:
       if args["type"] == "recognition":
           logger.error("For recognition you must provide a dataset. Use the flag: -d /path/to/dataset")
           sys.exit()

    # Getting input image. -i to get it from path. else get it from camera
    if args["image"]:
        input_img = cv2.imread(args['image'])
    else:
        cap = cv2.VideoCapture(0)
        ret, input_img = cap.read()
        pass
    
    dataset_path = args["dataset"]

    # input_img = cv2.imread('/media/felipe/Otros/Projects/Face_Recognizer_Service/imgs/00000002.jpg')

    # To Run Face recognition Test example
    # python videoAnalytics/analyzer.py recognition -d '/mnt/72086E48086E0C03/Projects/VideoAnalytics_Server/resources/user_data/1/g1/g1embeddings.json' -t -v
    # To Run Face detection Test example
    # python videoAnalytics/analyzer.py detection -t -v

    #loading the face detection model. 0 means to work with GPU. -1 is for CPU.
    face_detection_model = 'retinaface_r50_v1'
    face_recognition_model = 'arcface_r100_v1'

    logger.info(f'Loading face detection model {face_detection_model}')
    detector = insightface.model_zoo.get_model('retinaface_r50_v1')
    detector.prepare(ctx_id = 0, nms=0.4)

    if args["type"]== "recognition":
    #loading the face recognition model. 0 means to work with GPU. -1 is for CPU.
        logger.info('Face recognition Selected')
        logger.info(f'Loading face detection model {face_recognition_model}')
        recognizer = insightface.model_zoo.get_model('arcface_r100_v1')
        recognizer.prepare(ctx_id = 0)
    else:
        recognizer =None
        logger.info('Plain Face Detection selected')

    analyzer = faceAnalyzer(detector, recognizer, dataset_path)
    if args["type"] == "recognition":
        result,img = analyzer.analyze_img(input_img, return_img=True)
        if args["show_img"]:
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        result,img = analyzer.detect_img(input_img, return_img=True)
        if args["show_img"]:
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Video Testing if option selected. stop it pressing q
    if args["video_test"]:
        logger.info("Starting video test")
        while cap.isOpened():
            ret, img_read = cap.read()
            
            if args["type"]=="recognition":
                result,img = analyzer.analyze_img(img_read, return_img=True)
            else:
                result,img = analyzer.detect_img(img_read, return_img=True)
            
            logger.debug(result)

            cv2.namedWindow('Frame')
            cv2.imshow('Frame', img)

            fin = cv2.waitKey(1) & 0xFF
            if(fin == ord('q')):
                logger.info("Terminate key pressed. Closing program")
                break
        
if __name__=="__main__":
    main()