# In case this module is invoked from other modules, e.g., preprocessing
from pathlib import Path
import sys
sys.path.append(str(Path.cwd() / "annotation"))

import json
import os
from typing import List, Union, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datatypes import VideoAnnotation, Metadata
from utils import *


def annotate_video(
    file_path: str,
    label: str,
    video_filter: Callable[[str, Any], bool] = lambda path: True,
    **kwargs
) -> VideoAnnotation:
    if not video_filter(file_path, **kwargs):
        return None
    # print(f'Begin annotating {file_path}...')
    json_content: VideoAnnotation = {
        'video': convert_to_linux_path(shorten_file_path(file_path)),
        'label': label,
        'conversations': [
            {
                'from': 'human',    
                'value': '<image>\n' + get_input_prompt()
            },
            {
                'from': 'gpt',
                'value': get_output_prompt(label)
            }
        ]
    }
    return json_content



def dump_json(
    metadata: Metadata,
    video_filter: Callable[[str, Any], bool] = lambda path: True,
    **kwargs
) -> List[VideoAnnotation]:
    print(f'Annotating {len(metadata)} videos...')
    json_contents: List[VideoAnnotation] = []

    with ThreadPoolExecutor(max_workers=get_optimal_workers()) as executor:
        futures = []
        for (file_path, label) in metadata:
            futures.append(executor.submit(annotate_video, file_path, label, video_filter=video_filter, **kwargs))
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                json_contents.append(result)
    
    return json_contents