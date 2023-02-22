import os
import sys
import cv2
import argparse
import numpy as np
import uuid

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def capture_images(person, dataset_path):
    video = cv2.VideoCapture(0)

    apply_gamma = False
    while video.isOpened():
        ret, img = video.read()
        if(ret != True):
            break
        if(apply_gamma):
            img = adjust_gamma(img, gamma = 1.5)

        cv2.namedWindow('Frame')
        cv2.imshow('Frame', img)

        fin = cv2.waitKey(1) & 0xFF

        # Press q to exit the program
        if(fin == ord('q')):
            video.release()
            cv2.destroyAllWindows()
            sys.exit()

        # Press s to save the image
        if(fin == ord('s')):
            file_name = f'{dataset_path}/{person}/{person}{uuid.uuid4()}.jpg'
            cv2.imwrite( file_name, img )
            print('IMAGE SAVED', file_name)
            continue

        # Press g to toggle gamma correction
        if(fin == ord('g')):
            apply_gamma = not apply_gamma
            print(apply_gamma)
            continue

        # Press n to continue to the next person
        if(fin == ord('n')):
            print(f'Finished capturing images for {person}')
            break

    video.release()
    cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="dataset path")
    ap.add_argument("-n", "--name", required=False, help="name to register")
    args = vars(ap.parse_args())

    dataset_path = args["path"]
    person = args["name"]

    # Create the dataset folder if it doesn't exist
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    while True:
        if not person:
            person = input('Enter the name of the person to register: ')

        # Create the person folder if it doesn't exist
        if not os.path.exists(f'{dataset_path}/{person}'):
            os.mkdir(f'{dataset_path}/{person}')

        capture_images(person, dataset_path)
        person = None

if __name__ == "__main__":
    main()
