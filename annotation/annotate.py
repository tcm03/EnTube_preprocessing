# In case this module is invoked from other modules, e.g., preprocessing
from pathlib import Path
import sys
sys.path.append(str(Path.cwd() / "annotation"))

import json
import os
from typing import List, Union, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datatypes import VideoAnnotation, Metadata
from utils import get_optimal_workers, extract_label, convert_to_linux_path


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
        'video': convert_to_linux_path(file_path),
        'label': label,
        'conversations': [
            {
                'from': 'human',    
                'value': '<image>\nThis video is a Youtube video on one of many categories such as Education, Film & Animation, Comedy, Entertainment, Music, Howto & Style, and People & Blogs, etc. The engagement rate defined for each such video is based on the number of potential likes and dislikes only when published on Youtube. The higher number of likes and lower number of dislikes, the more engaged the video is. The final prediction label is either 0 (not engaged), 1 (neutral), or 2 (engaged). Please predict one of the three labels for this video, based on its contents only.'
            },
            {
                'from': 'gpt',
                'value': f'The engagement label of the video is {label}.'
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