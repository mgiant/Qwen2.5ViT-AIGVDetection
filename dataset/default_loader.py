from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import os
import random
import numpy as np
import logging
from PIL import Image

class VideoFrameDataset(Dataset):
    """Unified dataset for video/image frame loading across train, val, and test phases.

    Phase behavior:
        'train' - Random augmentations, random frame sampling, mp4 video support.
        'val'   - Deterministic transforms, center frame sampling.
        'test'  - Deterministic transforms, center frame sampling.
    """
    VALID_PHASES = ("train", "val", "test")
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    def __init__(self, cfg, df, phase='train'):
        if phase not in self.VALID_PHASES:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'train', 'val', or 'test'.")

        self.df = df
        self.phase = phase
        self.h = cfg.get('height', 224)
        self.w = cfg.get('width', 224)
        self.select_frame_nums = cfg.get('select_frame_nums', 8)
        self.num_class = cfg.get('num_class', 2)
        self.temporal_pad = cfg.get('temporal_pad', True)

        self.frame_strategy = cfg.get('frame_strategy', 'fps')
        self.fps = cfg.get('fps', 2)

        self.trans = self._build_transforms(cfg)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        content_path = row['content_path']
        label = int(row['label'])

        frames = self._load_and_transform(row, content_path)

        label_onehot = torch.nn.functional.one_hot(
            torch.tensor(label), num_classes=self.num_class
        ).float()
        binary_label = torch.tensor(label, dtype=torch.float32)

        return frames, label_onehot, binary_label, content_path

    def _load_and_transform(self, row, content_path):
        """Load frames from video file or image directory and apply transforms."""
        # loading directly from mp4/avi files
        if os.path.isfile(content_path) and content_path.lower().endswith(('.mp4', '.avi', '.mov')):
            raw_frames = self._load_frames_mp4(content_path)
            return torch.stack(self.trans(raw_frames), dim=0)

        if os.path.isdir(content_path):
            frame_list = self._parse_frame_seq(row.get('frame_seq'))
            image_ext = os.path.splitext(row.get('image_path', '.jpg'))[-1] or '.jpg'
            raw_frames = self._load_frames_dir(content_path, frame_list, image_ext)
            return torch.stack(self.trans(raw_frames), dim=0)

        if os.path.isfile(content_path) and content_path.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
            image_path = content_path
            raw_frame = self._load_image_rgb(image_path)
            frame_tensor = self.trans(raw_frame)
            return frame_tensor.unsqueeze(0).repeat(self.select_frame_nums, 1, 1, 1)

        else:
            # Empty frame list fallback
            logging.warning(f"Empty frame list for '{content_path}', returning black frames.")
            black = Image.new('RGB', (self.w, self.h), (0, 0, 0))
            frame_tensor = self.trans(black)
            return frame_tensor.unsqueeze(0).repeat(self.select_frame_nums, 1, 1, 1)

    def _parse_frame_seq(self, frame_seq_str):
        """Parse frame sequence string like '[1,2,3]' into a list of integers."""
        return [int(i) for i in frame_seq_str[1:-1].split(',') if i.strip()]

    def _load_frames_dir(self, video_dir, frame_list, image_ext='.jpg'):
        """Load frames from a directory of image files.

        For training, a random contiguous window is selected.
        For val/test, the center window is selected.
        If fewer frames are available than needed, the last frame is repeated for padding.
        """
        num_available = len(frame_list)
        select_num = self.select_frame_nums

        if num_available == 0:
            logging.warning(f"Empty frame list for '{video_dir}', returning black frames.")
            return [Image.new('RGB', (self.w, self.h), (0, 0, 0))] * select_num

        if num_available >= select_num:
            if self.phase == 'train':
                start = random.randint(0, num_available - select_num)
            else:
                start = (num_available - select_num) // 2
            indices = frame_list[start:start + select_num]
        else:
            # Copy to avoid mutating the original list
            indices = list(frame_list)
            if self.temporal_pad:
                indices.extend([frame_list[-1]] * (select_num - num_available))

        return [
            self._load_image_rgb(os.path.join(video_dir, f"{fn}{image_ext}"))
            for fn in indices
        ]

    def _load_frames_mp4(self, video_path):
        """Load frames from an mp4/avi video file using decord.

        Supports two sampling strategies:
            'uniform' - Uniformly sampled contiguous clip.
            'fps'     - FPS-based contiguous clip; random start in train phase.
        """
        import decord
        from decord import VideoReader, cpu

        decord.bridge.set_bridge('native')
        select_num = self.select_frame_nums

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
        except Exception as e:
            logging.error(f"Failed to load video '{video_path}': {e}. Returning black frames.")
            return [Image.new('RGB', (self.w, self.h), (0, 0, 0))] * select_num

        if total_frames == 0:
            logging.warning(f"Video '{video_path}' has 0 frames. Returning black frames.")
            return [Image.new('RGB', (self.w, self.h), (0, 0, 0))] * select_num

        indices = self._compute_video_sample_indices(total_frames, video_fps)

        try:
            frames_array = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
        except Exception as e:
            logging.error(f"Failed to decode frames from '{video_path}': {e}. Returning black frames.")
            return [Image.new('RGB', (self.w, self.h), (0, 0, 0))] * select_num

        return [Image.fromarray(frames_array[i]) for i in range(len(frames_array))]

    def _compute_video_sample_indices(self, total_frames, video_fps):
        select_num = self.select_frame_nums
        strategy = self.frame_strategy

        if strategy == 'fps':
            interval = max(1, int(video_fps / self.fps))
        elif strategy == 'uniform':
            if select_num <= 1 or total_frames <= 1:
                interval = 1
            else:
                interval = max(1, (total_frames - 1) // (select_num - 1))
        else:
            raise ValueError(f"Unknown frame strategy '{strategy}'. Use 'uniform' or 'fps'.")

        max_samples = (total_frames - 1) // interval + 1

        if max_samples >= select_num:
            span_needed = interval * (select_num - 1) + 1
            max_start = total_frames - span_needed
            if self.phase == 'train':
                start = random.randint(0, max(0, max_start))
            else:
                start = max(0, max_start) // 2
            indices = [start + i * interval for i in range(select_num)]
        else:
            indices = [i * interval for i in range(max_samples)]
            if self.temporal_pad:
                indices.extend([indices[-1]] * (select_num - max_samples))

        return np.array(indices, dtype=int)

    def _load_image_rgb(self, image_path):
        """Load a single image as RGB PIL Image, with black image fallback on error."""
        try:
            with Image.open(image_path) as img:
                return img.convert('RGB')
        except Exception as e:
            logging.error(f"Failed to load image '{image_path}': {e}. Returning black image.")
            return Image.new('RGB', (self.w, self.h), (0, 0, 0))

    # -------------------------------------------------------------------------
    # Transform construction
    # -------------------------------------------------------------------------

    def _build_transforms(self, cfg):
        """Build the image transform pipeline based on phase and processing config."""
        processing = cfg.get('processing', 'default')
        interpolation = cfg.get('interpolation', 3)  # bicubic
        h, w = self.h, self.w
        min_dim = min(h, w)

        if self.phase == 'train':
            return self._build_train_transforms(processing, interpolation, cfg, h, w, min_dim)
        else:
            return self._build_eval_transforms(processing, interpolation, cfg, h, w, min_dim)

    def _build_train_transforms(self, processing, interpolation, cfg, h, w, min_dim):
        """Build transforms with random augmentations for training."""

        if processing == 'default':
            return v2.Compose([
                v2.Resize(min_dim, interpolation=interpolation),
                v2.RandomCrop(min_dim),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ])
        elif processing == 'resize':
            return v2.Compose([
                v2.Resize((h, w), interpolation=interpolation),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
        elif processing == 'original':
            return v2.Compose([v2.ToImage()])
        else:
            raise ValueError(f"Unsupported training processing: '{processing}'")

    def _build_eval_transforms(self, processing, interpolation, cfg, h, w, min_dim):
        """Build deterministic transforms for validation and testing."""

        if processing == 'default':
            return v2.Compose([
                v2.Resize(min_dim, interpolation=interpolation),
                v2.CenterCrop(min_dim),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
        elif processing == 'resize':
            return v2.Compose([
                v2.Resize((h, w), interpolation=interpolation),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
        elif processing == 'original':
            return v2.Compose([v2.ToImage()])
        else:
            # For train-only processings (random_jpeg, etc.) used in eval, fall back to default
            logging.warning(f"Unsupported eval processing '{processing}', falling back to default.")
            return v2.Compose([
                v2.Resize(min_dim, interpolation=interpolation),
                v2.CenterCrop(min_dim),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])

    # -------------------------------------------------------------------------
    # Collate
    # -------------------------------------------------------------------------

    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        frames = torch.stack([item[0] for item in batch], dim=0)
        labels_onehot = torch.stack([item[1] for item in batch], dim=0)
        binary_labels = torch.stack([item[2] for item in batch], dim=0)
        content_paths = [item[3] for item in batch]

        return {
            'pixel_values': frames,
            'label_onehot': labels_onehot,
            'binary_label': binary_labels,
            'id': content_paths,
        }
