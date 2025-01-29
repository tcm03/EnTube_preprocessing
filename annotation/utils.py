# In case this module is invoked from other modules, e.g., preprocessing
from pathlib import Path
import sys
sys.path.append(str(Path.cwd() / "annotation"))

import decord
from datatypes import Metadata
from typing import List
import os
from multiprocessing import cpu_count
import traceback
from typing import Union


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
        vr = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
        return int(len(vr) / vr.get_avg_fps())
    except Exception as e:
        print(f"Error reading video {path}: {e}")
        print(traceback.format_exc())  # Include the full traceback for debugging
        return -1  # Use -1 to indicate an invalid duration

def filter_video(path: str, **kwargs) -> bool:
    try:
        max_duration = kwargs.get('max_duration', None)
        min_duration = kwargs.get('min_duration', None)
        satisfy_max_duration = False
        if max_duration is not None:
            duration = get_duration(path)
            if duration == -1:  # Handle invalid duration
                print(f"Skipping invalid video: {path}")
                return False
            satisfy_max_duration = duration <= max_duration
        else:
            satisfy_max_duration = True
        if min_duration is not None:
            duration = get_duration(path)
            satisfy_min_duration = duration >= min_duration
        else:
            satisfy_min_duration = True
        return satisfy_max_duration and satisfy_min_duration
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
    data_path: Union[str, Path]
) -> Metadata:
    metadata: Metadata = []
    folder_paths = os.listdir(data_path)
    for folder_path in folder_paths:
        folder_path = os.path.join(data_path, folder_path)
        label: str = extract_label(folder_path)
        assert label != '-1', f"Invalid folder path: {folder_path}"
        for file_name in os.listdir(folder_path):
            file_path: str = os.path.join(folder_path.rstrip('/'), file_name)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                metadata.append((file_path, label))
    print(f'Found {len(metadata)} videos')
    return metadata

def shorten_file_path(path: str) -> str:
    dirs = path.split('/')
    # loop dirs in reverse
    for i in range(len(dirs) - 1, -1, -1):
        if dirs[i].isnumeric():
            return '/'.join(dirs[i:])
    return path
