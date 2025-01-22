import torch
from torch.utils.data import Dataset
from typing import List
import os
from mm_datautils import process_video_frames
from transformers import BaseImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed

class EnTubeDataset(Dataset):
    
    def __init__(
        self,   
        folder_paths: List[str],
        image_processors: List[BaseImageProcessor],
    ) -> None:
        self.file_paths = []
        self.image_processors = image_processors

        for folder_path in folder_paths:
            file_names = os.listdir(folder_path)
            for file_name in file_names:
                file_path = os.path.join(folder_path, file_name)
                self.file_paths.append(file_path)

        # with ThreadPoolExecutor(max_workers=get_optimal_workers()) as executor:
        #     futures = []
        #     for folder_path in folder_paths:
        #         print(f'@tcm: In EnTubeDataset.__init__(): folder_path={folder_path}')
        #         file_names = os.listdir(folder_path)
        #         for file_name in file_names:
        #             file_path = os.path.join(folder_path, file_name)
        #             print(f'@tcm: In EnTubeDataset.__init__(): file_path={file_path}')
        #             future = executor.submit(process_video_frames, file_path, image_processor)
        #             futures.append(future)

        #     for future in as_completed(futures):
        #         result = future.result()
        #         if result is not None:
        #             video, image_size = result
        #             self.videos.append(video)
        #             self.image_sizes.append(image_size)

        

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        print(f'@tcm: In EnTubeDataset.__getitem__(): idx={idx}')
        video, image_size = process_video_frames(self.file_paths[idx], self.image_processors)
        return video, image_size

def collate_fn(batch):
    """
    batch: list of samples from EnTubeDataset.__getitem__()
    """
    assert isinstance(batch, list)
    assert isinstance(batch[0], tuple)
    print(f'@tcm: collate_fn')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_sizes = batch[0][1]
    batch_videos = [video for video, _ in batch]
    # batch_videos = [[video.to(device) for video in videos] for videos in zip(*batch_videos)]
    tmp_batch_videos = []
    for i, videos in enumerate(zip(*batch_videos)):
        print(f'@tcm: processor {i}')
        tmp = []
        for j, video in enumerate(videos):
            print(f'@tcm: video {j} shape: {video.shape}')
            video = video.to(device)
            tmp.append(video)
        tmp_batch_videos.append(tmp)
    batch_videos = tmp_batch_videos
    return batch_videos, image_sizes
