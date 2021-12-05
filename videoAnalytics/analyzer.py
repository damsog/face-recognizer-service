import argparse
import base64
import json
import imutils
import insightface
import cv2
import sys
import numpy as np
from mxnet.base import _NullType
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from videoAnalytics.utils import scaller_conc


# This class is used to Receive an image and a dataset, detect all faces on said image, then 
# get a code for each image, then compare it to the dataset to determine if matches a known 
# face or not.
# the dataset is a json file which containes, the label of the person and the embedding of the
# face.
class faceAnalyzer:
    def __init__(self, input_img, detector, recognizer, dataset_path) -> None:
        self.img = input_img
        self.dataset_path = dataset_path
        self.detector = detector
        self.recognizer = recognizer
        self.json_output = {}

        # Reading Dataset embeddings
        self.dataset = self.parse_dataset_json(self.dataset_path)
    

    # this method reads the json which containes the codes for a dataset and
    # puts information on a list to be used by the analyzer
    def parse_dataset_json(self, dataset_path):
        with open(dataset_path) as f:
            dataset = json.load(f)
        data_array = []

        # Each row is a face code. just transform the code as tring back into float and
        # concatenate it with its face label (person id, or label)
        for data in dataset:
            data_array.append([data[0], [float(value) for value in data[1].split(',')] ])
        
        return data_array

    def set_input_img(self, input_img):
        self.img = input_img
    
    def get_input_img(self):
        return self.img

    def set_dataset(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = self.parse_dataset_json(self.dataset_path)
    
    def get_dataset_path(self):
        return self.dataset_path
    
    def get_dataset(self):
        return self.dataset

    def get_output_data(self):
        return self.json_output
        
    def process_data(self):

        if not self.img:
            print("Input img empty. please set an image")
            return "{}"

        self.json_output = {}
        self.json_output['name'] = self.json_data['name']
        self.json_output['embeddings'] = []

        embeddings = []
        WIDTHDIVIDER = 4

        for img_name in self.json_data['imgs']:
            img = self.read_img( img_name, self.json_data["img_format"] )
            img = imutils.resize(img, width=int(img.shape[1]/WIDTHDIVIDER))

            bboxs, _ = self.model.detect(img, threshold=0.5, scale=1.0)

            if( bboxs is not None):
                todel = []
                for i in range(bboxs.shape[0]):
                    if(any(x<0 for x in bboxs[i])):
                        todel.append(i)
                for i in todel:
                    bboxs = np.delete(bboxs, i, 0)

                m_area = 0
                id_max = 0
                
                for (i, bbox ) in enumerate(bboxs):
                    #print(bbox)
                    area = (int(bbox[3]) - int(bbox[1]))*(int(bbox[2]) -int(bbox[0]))

                    if(area > m_area):
                        id_max = i
                        m_area = area

            
                face = scaller_conc( img[int(bboxs[id_max][1]):int(bboxs[id_max][3]), int(bboxs[id_max][0]):int(bboxs[id_max][2]), :] )
                if face is not None:
                    embedding = self.recognizer.get_embedding(face)
                    #ENCODDING NEED TO BE CONVERTED INTO SOMETHING THAT A DB CAN STORE EASILY
                    #np.set_printoptions(suppress=True)
                    #embedding_string = np.array2string( embedding[0] )
                    #print(embedding[0])
                    #Creating a list may round the data
                    embeddings.append( {"img":img_name , "embedding": [ num for num in embedding[0] ] } )



                print('File Coded: ', img_name)

        #print(self.json_data)
        self.json_output['embeddings'] = embeddings
        
        return self.json_output

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="Reads an image from path. If not given, opens camera")
    ap.add_argument("-p", "--print", required=False, help="Prints output on console", default=True)
    args = vars(ap.parse_args())

    if args["image"]:
        input_img = cv2.imread(args['image'])
    else:
        cap = cv2.VideoCapture(0)
        ret, input_img = cap.read()
            
    cv2.imshow('image', input_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #loading the face detection model. 0 means to work with GPU. -1 is for CPU.
    detector = insightface.model_zoo.get_model('retinaface_r50_v1')
    detector.prepare(ctx_id = 0, nms=0.4)

    #loading the face recognition model. 0 means to work with GPU. -1 is for CPU.
    recognizer = insightface.model_zoo.get_model('arcface_r100_v1')
    recognizer.prepare(ctx_id = 0)

    analyzer = faceAnalyzer(None, detector, recognizer, dataset_path='/media/felipe/Otros/Projects/VideoAnalytics_Server/resources/user_data/1/g1/g1embeddings.json')
    
    pass
        
if __name__=="__main__":
    main()