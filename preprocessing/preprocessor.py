import torch
from vision_encoders.builder import build_vision_tower_aux_list
from transformers import Qwen2Config
from typing import Optional, List, Tuple
import json
from transformers import BaseImageProcessor
from resource_logging import measure_resource_usage, MeasureResourceUsage

class CambrianConfig(Qwen2Config):
    model_type = "cambrian_qwen"
    debug = "debug"

    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_json_file(cls, json_file_path):
        """Load a config from a json file."""
        with open(json_file_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class CambrianEncoders:

    def __init__(
        self, 
        config: CambrianConfig
    ) -> None:
        self.config: CambrianConfig = config
        self.vision_tower_aux_list = build_vision_tower_aux_list(config, delay_load=True)

    @measure_resource_usage()
    def encode_images(self, image_aux_list, encode_type=None):
        vision_tower_aux_list = self.vision_tower_aux_list
        image_aux_features_list = []
        chunk_size = 64
        if encode_type == "dino":
            # print(f'@tcm: In CambrianEncoders.encode_images(): dinov2')
            image_aux = image_aux_list[-1]
            vision_tower_aux = vision_tower_aux_list[-1]
            if image_aux.shape[0] > chunk_size:
                image_aux_features_chunks = []
                for start_idx in range(0, image_aux.shape[0], chunk_size):
                    # print(f'@tcm: In CambrianEncoders.encode_images(): dinov2 chunk start_idx={start_idx}')
                    end_idx = min(start_idx + chunk_size, image_aux.shape[0])
                    chunk = image_aux[start_idx:end_idx]
                    image_aux_features_chunk = vision_tower_aux(chunk)
                    image_aux_features_chunks.append(image_aux_features_chunk)
                image_aux_features = torch.cat(image_aux_features_chunks, dim=0)
            else:
                # print(f'@tcm: In CambrianEncoders.encode_images(): image_aux shape: {image_aux.shape}')
                image_aux_features = vision_tower_aux(image_aux)
            return image_aux_features
        elif encode_type == "siglip":
            # print(f'@tcm: In CambrianEncoders.encode_images(): siglip')
            image_aux = image_aux_list[0]
            vision_tower_aux = vision_tower_aux_list[0]
            if image_aux.shape[0] > chunk_size:
                image_aux_features_chunks = []
                for start_idx in range(0, image_aux.shape[0], chunk_size):
                    # print(f'@tcm: In CambrianEncoders.encode_images(): siglip chunk start_idx={start_idx}')
                    end_idx = min(start_idx + chunk_size, image_aux.shape[0])
                    chunk = image_aux[start_idx:end_idx]
                    image_aux_features_chunk = vision_tower_aux(chunk)
                    image_aux_features_chunks.append(image_aux_features_chunk)
                image_aux_features = torch.cat(image_aux_features_chunks, dim=0)
            else:
                image_aux_features = vision_tower_aux(image_aux)
            return image_aux_features
        else:
            # print(f'@tcm: In CambrianEncoders.encode_images(): both encode_type')
            for image_aux, vision_tower_aux in zip(
                image_aux_list, vision_tower_aux_list
            ):
                if image_aux.shape[0] > chunk_size:
                    image_aux_features_chunks = []
                    for start_idx in range(0, image_aux.shape[0], chunk_size):
                        # print(f'@tcm: In CambrianEncoders.encode_images(): both encode_type chunk start_idx={start_idx}')
                        end_idx = min(start_idx + chunk_size, image_aux.shape[0])
                        chunk = image_aux[start_idx:end_idx]
                        image_aux_features_chunk = vision_tower_aux(chunk)
                        image_aux_features_chunks.append(image_aux_features_chunk)
                    image_aux_features = torch.cat(image_aux_features_chunks, dim=0)
                else:
                    image_aux_features = vision_tower_aux(image_aux)
                image_aux_features_list.append(image_aux_features)
            return image_aux_features_list

    @measure_resource_usage()
    def select_frame(
            self,
            feature_list,
            split_sizes,
            new_image_aux_list,
            image_sizes,
            window_size=16,
            threshold=0.83,
        ):
        dino_features_batch = torch.split(feature_list, split_sizes, dim=0)
        new_image_aux_batch_0 = torch.split(new_image_aux_list[0], split_sizes, dim=0)
        new_image_aux_batch_1 = torch.split(new_image_aux_list[1], split_sizes, dim=0)
        new_split_sizes = []
        selected_frames_all_0 = []
        selected_frames_all_1 = []
        selected_frames_feature_all = []
        selected_frame_indices_all = []
        for i_batch, frame_features in enumerate(dino_features_batch):
            # print(f'@tcm: In CambrianEncoders.select_frame(): dino features batch {i_batch}')
            if isinstance(frame_features, torch.Tensor):
                # print(f'@tcm: In CambrianEncoders.select_frame(): frame_features shape: {frame_features.shape}')
                pass
            original_width, original_height = image_sizes[i_batch]
            if getattr(self.config, "highres", False):
                token_per_frame = self.config.lowres_token ** 2
            else:
                token_per_frame = self.config.image_token_len

            max_num_frames = max(
                1,
                (
                    self.config.tokenizer_model_max_length
                    - getattr(self.config, "inference_max_length", 16)
                )
                // token_per_frame,
            )
            if len(frame_features) < max_num_frames:
                selected_frames_all_0.append(new_image_aux_batch_0[i_batch])
                selected_frames_all_1.append(new_image_aux_batch_1[i_batch])
                selected_frames_feature_all.append(frame_features)
                new_split_sizes.append(len(frame_features))
                selected_frame_indices_all.append(torch.arange(len(frame_features)))
                continue

            num_segments = len(frame_features) // window_size
            if num_segments == 0:
                query_feature = frame_features.flatten(1, 2)
                query_feature = query_feature / torch.norm(
                    (query_feature), dim=1, keepdim=True
                )
                similarities = torch.mean(query_feature @ query_feature.T, dim=1)
                similarities[len(frame_features) // 2] = 0
                indices = torch.where(similarities < threshold)[0]
                selected_frame_indices_all.append(indices)
                selected_frames_all_0.append(new_image_aux_batch_0[i_batch][indices])
                selected_frames_all_1.append(new_image_aux_batch_1[i_batch][indices])
                selected_frames_feature_all.append(frame_features[indices])
                new_split_sizes.append(len(indices))
                continue
            segments_frames_0 = []
            segments_frames_1 = []
            segments_features = []
            for start_idx in range(0, len(frame_features), window_size):
                end_idx = min(start_idx + window_size, len(frame_features))
                segments_frames_0.append(
                    new_image_aux_batch_0[i_batch][start_idx:end_idx]
                )
                segments_frames_1.append(
                    new_image_aux_batch_1[i_batch][start_idx:end_idx]
                )
                segments_features.append(frame_features[start_idx:end_idx])
            selected_frames_0 = []
            selected_frames_1 = []
            selected_features = []
            selected_frame_indices = []
            for i, segment in enumerate(segments_features):
                query_feature = segment.flatten(1, 2)
                query_feature = query_feature / torch.norm(
                    (query_feature), dim=1, keepdim=True
                )
                similarities = torch.mean(query_feature @ query_feature.T, dim=1)
                similarities[len(segment) // 2] = 0
                indices = torch.where(similarities < threshold)[0]
                selected_frames_0.append(segments_frames_0[i][indices])
                selected_frames_1.append(segments_frames_1[i][indices])
                selected_features.append(segment[indices])
                selected_frame_indices.extend(indices + i * window_size)
            selected_frames_0 = torch.cat(selected_frames_0, dim=0)
            selected_frames_1 = torch.cat(selected_frames_1, dim=0)
            selected_features = torch.cat(selected_features, dim=0)
            selected_frame_indices = torch.tensor(selected_frame_indices)
            # ablation
            max_num_frames = 400  # in case of OOM
            if len(selected_frames_0) > max_num_frames:
                interval = len(selected_frames_0) / float(max_num_frames)
                indices = [int(interval * i) for i in range(max_num_frames)]
                new_split_sizes.append(len(indices))
                selected_frames_all_0.append(selected_frames_0[indices])
                selected_frames_all_1.append(selected_frames_1[indices])
                selected_frames_feature_all.append(selected_features[indices])
                selected_frame_indices = selected_frame_indices[indices]
            else:
                new_split_sizes.append(len(selected_frames_0))
                selected_frames_all_0.append(selected_frames_0)
                selected_frames_all_1.append(selected_frames_1)
                selected_frames_feature_all.append(selected_features)
            selected_frame_indices_all.append(selected_frame_indices)
        selected_frames_all_0 = torch.cat(selected_frames_all_0, dim=0)
        selected_frames_all_1 = torch.cat(selected_frames_all_1, dim=0)
        selected_frames_feature_all = torch.cat(selected_frames_feature_all, dim=0)
        return (
            selected_frames_feature_all,
            new_split_sizes,
            [selected_frames_all_0, selected_frames_all_1],
            selected_frame_indices_all,
        )

    @measure_resource_usage()
    def prepare_mm_features(
        self,
        images: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        with MeasureResourceUsage():
            image_aux_list = images
            split_sizes_ori = [
                1 if image.ndim == 3 else image.shape[0] for image in image_aux_list[0]
            ]
            new_image_aux_list = []
            for image_aux in image_aux_list:
                if type(image_aux) is list:
                    # image_aux = [
                    #     x.unsqueeze(0) if x.ndim == 3 else x for x in image_aux
                    # ]
                    tmp_image_aux = []
                    for x in image_aux:
                        assert x.ndim == 3 or x.ndim == 4, 'Only allow video tensor to have 3 or 4 dimensions'
                        # print(f'@tcm: In CambrianEncoders.prepare_mm_features(): x.shape={x.shape}')
                        if x.ndim == 3:
                            # add num_frames dimension
                            x = x.unsqueeze(0)
                        # if x.shape[0] == 1:
                        #     x = x.squeeze(0)
                        tmp_image_aux.append(x)
                    image_aux = tmp_image_aux
                    
                concat_image_aux = torch.cat([image for image in image_aux], dim=0)
                new_image_aux_list.append(concat_image_aux)
        # print(f'@tcm: In CambrianEncoders.prepare_mm_features(): extracting DINOv2 features...')
        # dinov2_start_time = time.time()
        image_aux_features_dino = self.encode_images(
            new_image_aux_list, encode_type="dino"
        )
        # print(f'@tcm: In CambrianEncoders.prepare_mm_features(): DINOv2 time: {time.time() - dinov2_start_time:4f}')
        # print(f'@tcm: In CambrianEncoders.prepare_mm_features(): selecting frames...')
        # select_frame_start_time = time.time()
        (
            image_aux_features_dino,
            split_sizes,
            new_image_aux_list,
            selected_frame_indices_all,
        ) = self.select_frame(
            image_aux_features_dino,
            split_sizes_ori,
            new_image_aux_list,
            image_sizes,
            threshold=getattr(self.config, "dino_threshold", 0.83),
        )
        # print(f'@tcm: In CambrianEncoders.prepare_mm_features(): select frame time: {time.time() - select_frame_start_time:4f}')
        # print(f'@tcm: In CambrianEncoders.prepare_mm_features(): extracting SIGLIP features...')
        # siglip_start_time = time.time()
        image_aux_features_siglip = self.encode_images(
            new_image_aux_list, encode_type="siglip"
        )
        # print(f'@tcm: In CambrianEncoders.prepare_mm_features(): select frame time: {time.time() - siglip_start_time:4f}')
        image_aux_features_list = [
            image_aux_features_siglip,
            image_aux_features_dino,
        ]
        return image_aux_features_list