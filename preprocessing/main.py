import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from annotation.utils import get_optimal_workers
import torch.multiprocessing as mp

import os
import argparse
from typing import List, Dict
from mm_datautils import process_video_frames
from preprocessor import CambrianConfig, CambrianEncoders
import torch
from safetensors.torch import save_file
from collections import defaultdict
import logging
from multiprocessing import cpu_count
from entube_dataset import EnTubeDataset, collate_fn
from torch.utils.data import Dataset, DataLoader
from transformers import BaseImageProcessor
from constants import *


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_optimal_workers() -> int:
    """Determine the optimal number of workers based on available CPU cores."""
    try:
        return max(1, cpu_count() - 1)  # Leave one core free
    except (NotImplementedError, ValueError):
        return 1  # Fallback to a single worker in case of an error

def extract_features(processor: CambrianEncoders, file_path: str, file_name: str) -> Dict[str, torch.Tensor]:
    try:
        video, image_sizes = process_video_frames(file_path)
        image_aux_features_list = processor.prepare_mm_features(images=video, image_sizes=image_sizes)
        return {
            file_name + '-siglip': image_aux_features_list[0],
            file_name + '-dino': image_aux_features_list[1]
        }
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folders',
        type=str,
        nargs='+',
        required=True,
        help="List of folder paths to video data"
    )
    parser.add_argument(
        '--output_file',
        type = str,
        default = 'entube_tensors.safetensors',
        help = 'Safetensor file to store embeddings of EnTube dataset by vision encoders'
    )
    parser.add_argument(
        '--config_file',
        type = str,
        default = 'config.json',
        help = 'Path to configuration file of encoders parameters'
    )
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(SAFETENSORS_PATH, exist_ok=True)
    # mp.set_start_method('spawn')

    cambrianConfig = CambrianConfig.from_json_file(args.config_file)
    processor = CambrianEncoders(cambrianConfig)
    image_processors = []
    # if not processor.vision_tower_aux_list[0].is_loaded:
    #     processor.vision_tower_aux_list[0].load_model()
    # image_processors.append(processor.vision_tower_aux_list[0].image_processor)
    for vision_tower_aux in processor.vision_tower_aux_list:
        if not vision_tower_aux.is_loaded:
            vision_tower_aux.load_model()
        vision_tower_aux.to(device)
        image_processors.append(vision_tower_aux.image_processor)

    folder_paths: List[str] = args.folders
    entube_dataset = EnTubeDataset(folder_paths, image_processors)
    dataloader = DataLoader(
        entube_dataset, 
        batch_size=1, 
        collate_fn=collate_fn,
    )

    for batch_idx, (videos, image_sizes, file_names) in enumerate(dataloader):
        print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
        assert isinstance(videos, list), "List of videos features for each processor (vision encoder)"
        assert isinstance(videos[0], list) or isinstance(videos[0], torch.Tensor), "List of videos in the batch"
        # tensor(num_reduced_frames, len=576, hidden_dim=1152/1536) image_aux_features_list[num_processors]
        image_aux_features_list = processor.prepare_mm_features(videos, image_sizes)
        tensor_siglip = image_aux_features_list[0].to('cpu')
        tensor_dino = image_aux_features_list[1].to('cpu')
        # file_name = file_names[0] # the batch has only one file
        for file_name in file_names:
            print(f'file_name={file_name}')
            save_tensor = {
                file_name + '-siglip': tensor_siglip,
                file_name + '-dino': tensor_dino
            }
            save_file(save_tensor, os.path.join(SAFETENSORS_PATH, file_name + '.safetensors'))
        
        

    # save_file(dict(data_tensor), args.output_file)
