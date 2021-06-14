import argparse
import json
import imutils
import insightface
import cv2
import numpy as np
from utils import scaller_conc


#This script  gets the embedding of a set of images.
#Receives a json containing the path to siad images as input.
#{"name":"what","imgs":["imgs/felipe1.jpg","imgs/felipe7.jpg"]}
#Returns a json containing the embedding of each image to be stored or processed.
#This script can be called from terminal. 
#: Create a another python script to test this script independenly

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_data", required=True,
	help="json containing input data")
args = vars(ap.parse_args())

input_data = args['input_data']
#input_data = '{"name":"what","imgs":["imgs/felipe1.jpg","imgs/felipe7.jpg"]}'

def main():
    json_data = json.loads(input_data)
    json_output = {}
    json_output['name'] = json_data['name']
    json_output['embeddings'] = []
    
    
    #loading the face detection model. 0 means to work with GPU. -1 is for CPU.
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id = 0, nms=0.4)

    #loading the face recognition model. 0 means to work with GPU. -1 is for CPU.
    recognizer = insightface.model_zoo.get_model('arcface_r100_v1')
    recognizer.prepare(ctx_id = 0)

    embeddings = []
    WIDTHDIVIDER = 4

    for img_name in json_data['imgs']:
        img = cv2.imread(img_name)
        img = imutils.resize(img, width=int(img.shape[1]/WIDTHDIVIDER))

        bboxs, _ = model.detect(img, threshold=0.5, scale=1.0)

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
                embedding = recognizer.get_embedding(face)
                #ENCODDING NEED TO BE CONVERTED INTO SOMETHING THAT A DB CAN STORE EASILY
                #np.set_printoptions(suppress=True)
                #embedding_string = np.array2string( embedding[0] )
                #print(embedding[0])
                #Creating a list may round the data
                embeddings.append( {"img":img_name , "embedding": [ num for num in embedding[0] ] } )



            print('File Coded: ', img_name)

    print(json_data)
    json_output['embeddings'] = embeddings
    print("[OUTPUT:BEGIN]")
    print(json_output)
    print("[OUTPUT:END]")

if __name__ == "__main__":
    main()