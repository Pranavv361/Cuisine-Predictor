#Libraries Used
import json
import sys
import argparse
import numpy as np
import os
import pandas as pd
from modules import PreprocessAndModelling
import traceback


# Load Data from the file
def load_data(data_file_name):
    # get the directory
    dir_path = os.path.dirname(os.path.abspath(__file__))

    # load the data file using a relative path
    data_file_path = os.path.join(dir_path, data_file_name)
    data = pd.read_json(data_file_path)
    
    return data


#Main function
def main(args):
    # Load the data from .json file and call the preprocess method and modelling function
    try:
        data = load_data('yummly.json')

        cuisine, score, closest = PreprocessAndModelling.modeltrain(data, args.ingredient, args.N)

        #Generating the Json output file
        json_raw_output = {}
        json_raw_output["cuisine"] = cuisine[0]
        json_raw_output["score"] = score[0]
        json_raw_output["closest"] = [{"id": str(c['id']), "score": c['score']}
                                    for c in closest]
        #Writing the output file
        with open('output.json', 'w') as f:
            json.dump(json_raw_output, f, indent=2)
        
        #Opening the output file and printing it on command terminal
        with open('output.json','r') as f:
            json_data = json.load(f)
        
        print(json.dumps(json_data, indent=2))
        
    except Exception as e:
        # handle any exceptions that occur and print them
        traceback.print_exc()



# Function to get input from input terminal
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the type of cuisine based on a set of ingredients.')
    parser.add_argument('--N', required = True, type=int, help='number of similar foods to return', default=5)
    parser.add_argument('--ingredient',required = True,  type=str, help='an ingredient of the food', action='append')

    args = parser.parse_args()

    #To handle errors 
    try:
        sys.stderr.write("\n")
        main(args)

    except Exception as e:
        #handle any exceptions that occur and print them
        sys.stderr.write(str(e) + "\n")
