# In case this module is invoked from other modules, e.g., preprocessing
from pathlib import Path
import sys
sys.path.append(str(Path.cwd() / "annotation"))

import json
import os
import argparse
from sklearn.model_selection import train_test_split
from datatypes import VideoAnnotation, Metadata
from annotate import dump_json
from utils import get_metadata, filter_video
from typing import List



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog = 'train_test.py',
        description='Annotate video dataset with JSON format'
    )
    parser.add_argument(
        '--folders',
        type = str,
        nargs = '+',
        required = True,
        help = "List of folder paths to video data"
    )
    parser.add_argument(
        '--train_size', 
        type=float, 
        default=0.8, 
        help='Proportion of the dataset for training'
    )
    parser.add_argument(
        '--output_train_file', 
        type=str, 
        default='data/EnTube_train.json', 
        help='Output JSON file for training'
    )
    parser.add_argument(
        '--output_test_file',
        type=str,
        default='data/EnTube_test.json',
        help='Output JSON file for testing'
    )
    parser.add_argument(
        '--max_duration', 
        type=int, 
        help='Maximum duration of video in seconds'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for train-test split'
    )
    args = parser.parse_args()

    folder_paths: List[str] = args.folders
    metadata: Metadata = get_metadata(folder_paths)
    # split metadata into 3 submetadata corresponding to 3 labels
    metadata_label = {0: [], 1: [], 2: []}
    for video, label in metadata:
        metadata_label[int(label)].append((video, label))
    train = []
    test = []
    for label, videos in metadata_label.items():
        train_l, test_l = train_test_split(
            videos,
            train_size=args.train_size,
            random_state=args.random_state
        )
        print(f'Label {label}: {len(train_l)} training videos, {len(test_l)} testing videos')
        train.extend(train_l)
        test.extend(test_l)

    json_train: List[VideoAnnotation] = dump_json(train, filter_video, **vars(args))
    json_test: List[VideoAnnotation] = dump_json(test, filter_video, **vars(args))

    with open(args.output_train_file, 'w') as f:
        json.dump(json_train, f, indent=4)
    print(f"Training data saved to {args.output_train_file}")
    with open(args.output_test_file, 'w') as f:
        json.dump(json_test, f, indent=4)
    print(f"Testing data saved to {args.output_test_file}")