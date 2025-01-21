# In case this module is invoked from other modules, e.g., preprocessing
from pathlib import Path
import sys
sys.path.append(str(Path.cwd() / "annotation"))

import decord as de
from datatypes import Metadata
from typing import List
import os
from multiprocessing import cpu_count
import traceback
from pathlib import Path


def convert_to_linux_path(path: str) -> str:
    return Path(path).as_posix()

def extract_label(path: str) -> str:
    idx = len(path) - 1
    while idx >= 0:
        if path[idx].isnumeric():
            return path[idx]
        idx -= 1
    return '-1'

def get_duration(path: str) -> int:
    try:
        vr = de.VideoReader(path, ctx=de.cpu(0), num_threads=1)
        return int(len(vr) / vr.get_avg_fps())
    except Exception as e:
        print(f"Error reading video {path}: {e}")
        print(traceback.format_exc())  # Include the full traceback for debugging
        return -1  # Use -1 to indicate an invalid duration

def filter_video(path: str, **kwargs) -> bool:
    try:
        max_duration = kwargs.get('max_duration', None)
        if max_duration is not None:
            duration = get_duration(path)
            if duration == -1:  # Handle invalid duration
                print(f"Skipping invalid video: {path}")
                return False
            return duration <= max_duration
        return True
    except Exception as e:
        print(f"Error in filter_video for {path}: {e}")
        return False

def get_optimal_workers() -> int:
    """Determine the optimal number of workers based on available CPU cores."""
    try:
        return max(1, cpu_count() - 1)  # Leave one core free
    except (NotImplementedError, ValueError):
        return 1  # Fallback to a single worker in case of an error

def get_metadata(
    folder_paths: List[str]
) -> Metadata:
    metadata: Metadata = []
    for folder_path in folder_paths:
        label: str = extract_label(folder_path)
        assert label != '-1', f"Invalid folder path: {folder_path}"
        for file_name in os.listdir(folder_path):
            file_path: str = os.path.join(folder_path.rstrip('/'), file_name)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                metadata.append((file_path, label))
    print(f'Found {len(metadata)} videos')
    return metadata