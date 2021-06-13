#import utils
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_data", required=True,
	help="json containing input data")
args = vars(ap.parse_args())

json_data = args['input_data']

def main():
    print("not much for now")

if __name__ == "__main__":
    main()