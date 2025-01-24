import torch
import os
import argparse
from typing import List, Dict
from mm_datautils import process_video_frames
from preprocessor import CambrianConfig, CambrianEncoders
from safetensors.torch import save_file
from collections import defaultdict
import logging
from multiprocessing import cpu_count
from entube_dataset import EnTubeDataset, collate_fn
from torch.utils.data import Dataset, DataLoader
from transformers import BaseImageProcessor
from constants import *

# import annotation.utils (which imports decord) after torch to avoid bug
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from annotation.utils import get_optimal_workers
import torch.multiprocessing as mp


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_fileid(file_path: str) -> str:
    return file_path.split('.')[0]

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

    start_allocated = torch.cuda.memory_allocated(device)
    start_reserved = torch.cuda.memory_reserved(device)

    cambrianConfig = CambrianConfig.from_json_file(args.config_file)
    processor = CambrianEncoders(cambrianConfig)
    image_processors = []
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

    end_allocated = torch.cuda.memory_allocated(device)
    end_reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    print(f"Allocated before init: {start_allocated/1e6:.2f} MB")
    print(f"Allocated after init:  {end_allocated/1e6:.2f} MB")
    print(f"Net allocated change:   {(end_allocated - start_allocated)/1e6:.2f} MB")
    print(f"Reserved before init:  {start_reserved/1e6:.2f} MB")
    print(f"Reserved after init:   {end_reserved/1e6:.2f} MB")
    print(f"Peak allocated:         {peak_allocated/1e6:.2f} MB")

    for batch_idx, (videos, image_sizes, file_names) in enumerate(dataloader):
        print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
        assert isinstance(videos, list), "List of videos features for each processor (vision encoder)"
        assert isinstance(videos[0], list) or isinstance(videos[0], torch.Tensor), "List of videos in the batch"
        # tensor(num_reduced_frames, len=576, hidden_dim=1152/1536) image_aux_features_list[num_processors]

        start_allocated = torch.cuda.memory_allocated(device)
        start_reserved = torch.cuda.memory_reserved(device)

        image_aux_features_list = processor.prepare_mm_features(videos, image_sizes)

        end_allocated = torch.cuda.memory_allocated(device)
        end_reserved = torch.cuda.memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        print(f"Allocated before prepare_mm_features: {start_allocated/1e6:.2f} MB")
        print(f"Allocated after prepare_mm_features:  {end_allocated/1e6:.2f} MB")
        print(f"Net allocated change:   {(end_allocated - start_allocated)/1e6:.2f} MB")
        print(f"Reserved before prepare_mm_features:  {start_reserved/1e6:.2f} MB")
        print(f"Reserved after prepare_mm_features:   {end_reserved/1e6:.2f} MB")
        print(f"Peak allocated:         {peak_allocated/1e6:.2f} MB")
        
        tensor_siglip = image_aux_features_list[0].to('cpu')
        tensor_dino = image_aux_features_list[1].to('cpu')
        # file_name = file_names[0] # the batch has only one file
        for file_name in file_names:
            print(f'file_name={file_name}')
            file_id = extract_fileid(file_name)
            save_tensor = {
                file_id + '-siglip': tensor_siglip,
                file_id + '-dino': tensor_dino
            }
            save_file(save_tensor, os.path.join(SAFETENSORS_PATH, file_id + '.safetensors'))
