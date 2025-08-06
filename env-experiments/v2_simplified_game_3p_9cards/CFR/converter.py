import json
import pickle
import datetime
import numpy as np
from CFRalgorithm import TempleCFR, NodeInfoSet, NodeState
import sys
import os

# This is to handle the GameModel import in CFRalgorithm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def convert_to_json():
    # Get the latest checkpoint file path
    #try:
    checkpoint_file = sys.argv[1]
    #except:
    #    with open('latest_checkpoint.txt', 'r') as f:
    #         checkpoint_file = f.read().strip()

    # Load the checkpoint file
    with open(checkpoint_file, 'rb') as f:
        k = pickle.load(f)

    # Prepare the data for JSON
    dump = {node: tuple(k.nodes[node].get_average_strategy()) for node in k.nodes}

    # Create a filename with date and hour
    now = datetime.datetime.now()
    checkpoint_number = checkpoint_file.split('/')[-1].split('_')[-1].split('.')[0]
    filename = f"models/modelv2-defenders-could-lie/cfr_strategies_{now.strftime('%Y-%m-%d_%H-%M-%S')}-{checkpoint_number}.json"

    # Write the JSON file
    with open(filename, 'w') as f:
        json.dump(dump, f, indent=4)

    print(f"Successfully converted {checkpoint_file} to {filename}")

if __name__ == '__main__':
    convert_to_json()
