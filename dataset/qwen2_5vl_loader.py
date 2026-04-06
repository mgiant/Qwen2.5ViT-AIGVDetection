import torch.utils
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.io import decode_jpeg, encode_jpeg
import torch
import os
from PIL import Image
import random
from transformers import AutoVideoProcessor
import numpy as np

class _RandomJPEGCompression:
    def __init__(self, p, quality):
        self.p = p
        self.min_quality, self.max_quality = quality

    def __call__(self, frames):
        if random.random() >= self.p:
            return frames

        quality = random.randint(self.min_quality, self.max_quality)
        compressed = []
        for frame in frames:
            encoded = encode_jpeg(F.pil_to_tensor(frame), quality=quality)
            decoded = decode_jpeg(encoded)
            compressed.append(F.to_pil_image(decoded))
        return compressed


class _RandomResize:
    def __init__(self, min_hw, scale=(0.5, 1.0)):
        self.min_hw = int(min_hw)
        self.scale = scale

    def __call__(self, frames):
        if not frames:
            return frames

        width, height = frames[0].size
        short_side = min(width, height)
        min_scale = self.min_hw / short_side if short_side > 0 else 1.0
        effective_min_scale = max(self.scale[0], min_scale)
        effective_max_scale = self.scale[1]
        scale_factor = (
            effective_min_scale
            if effective_min_scale > effective_max_scale
            else random.uniform(effective_min_scale, effective_max_scale)
        )

        new_height = max(1, int(height * scale_factor))
        new_width = max(1, int(width * scale_factor))
        return [F.resize(frame, size=[new_height, new_width]) for frame in frames]


class _RandomScaleCrop:
    def __init__(self, min_hw, scale=(0.5, 1.0)):
        self.min_hw = int(min_hw)
        self.scale = scale

    def __call__(self, frames):
        if not frames:
            return frames

        width, height = frames[0].size
        short_side = min(width, height)
        min_scale = self.min_hw / short_side if short_side > 0 else 1.0
        effective_min_scale = max(self.scale[0], min_scale)
        effective_max_scale = self.scale[1]

        if effective_min_scale > effective_max_scale:
            if effective_min_scale > 1.0:
                return frames
            scale_factor = effective_min_scale
        else:
            scale_factor = random.uniform(effective_min_scale, effective_max_scale)

        crop_width = min(max(1, int(width * scale_factor)), width)
        crop_height = min(max(1, int(height * scale_factor)), height)
        left = 0 if width == crop_width else random.randint(0, width - crop_width)
        top = 0 if height == crop_height else random.randint(0, height - crop_height)
        return [F.crop(frame, top, left, crop_height, crop_width) for frame in frames]

class Dataset_Qwen25ViT(Dataset):
    """
    Unified dataset for the Qwen2.5 ViT model.
    Supports train / val / test phases with phase-specific logic for
    frame selection, augmentation, and output format.
    Handles both video files (.mp4 / .avi) and image-frame folders.
    """

    VALID_PHASES = ("train", "val", "test")
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")

    def __init__(self, cfg, df=None, phase="train"):
        assert phase in self.VALID_PHASES, (
            f"Invalid phase '{phase}'. Must be one of {self.VALID_PHASES}."
        )

        self.df = df
        self.phase = phase
        self.h = cfg.get("h", 224)
        self.w = cfg.get("w", 224)
        self.select_frame_nums = cfg.get("select_frame_nums", 8)
        self.temporal_pad = cfg.get("temporal_pad", True)
        self.patch_size = int(cfg.get("patch_size", 14))
        self.num_classes = cfg.get("num_classes", 2)
        self.frame_strategy = cfg.get("frame_strategy", "fps")
        self.fps = cfg.get("fps", 2)
        self.default_image_ext = cfg.get("image_ext", ".jpg")

        self.processor = AutoVideoProcessor.from_pretrained(
            cfg.get('model_source'),
            size={
                "shortest_edge": int(cfg["min_pixels"]),
                "longest_edge": int(cfg["max_pixels"]),
            },
        )
        self.transform = self._build_augmentation(cfg) if phase == "train" else None

    # ------------------------------------------------------------------ #
    #  Public interface
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row["content_path"]
        label = int(row["label"])

        # Load frames from video file or image folder
        frames = self._load_frames(row)

        # Apply augmentation (train only), then common preprocessing
        if self.transform is not None:
            frames = self.transform(frames)

        binary_label = torch.tensor(label, dtype=torch.float32) 
        label_onehot = torch.nn.functional.one_hot(
            torch.tensor(label), num_classes=self.num_classes
        ).float()
        return frames, label_onehot, binary_label, video_id

    def collate_fn(self, batch):
        videos, label_onehot, binary_label, video_ids = zip(*batch)

        processed = self.processor(
            videos=list(videos), return_tensors="pt", patch_size=self.patch_size
        )

        result = {
            "pixel_values": processed["pixel_values_videos"],
            "image_grid_hws": processed["video_grid_thw"],
            "label_onehot": torch.stack(label_onehot),
            "binary_label": torch.stack(binary_label),
            "id": video_ids,
        }

        return result

    # ------------------------------------------------------------------ #
    #  Frame loading – top-level dispatcher
    # ------------------------------------------------------------------ #

    def _load_frames(self, row):
        """Dispatch to the video-file loader or the image-folder loader."""
        content_path = row["content_path"]

        if os.path.isfile(content_path) and content_path.lower().endswith(self.VIDEO_EXTENSIONS):
            return self._load_frames_from_video(content_path)

        return self._load_frames_from_folder(row)

    # ------------------------------------------------------------------ #
    #  Image-folder loader
    # ------------------------------------------------------------------ #

    def _load_frames_from_folder(self, row):
        """Load frames from a directory of numbered image files."""
        video_dir = row["content_path"]
        frame_list = self._parse_frame_seq(row["frame_seq"])

        if len(frame_list) == 1:
            img = self._load_image_rgb(row["image_path"])
            return [img]

        # Determine file extension from image_path if available
        image_ext = self.default_image_ext
        if "image_path" in row.index and row["image_path"]:
            ext = os.path.splitext(row["image_path"])[-1]
            if ext:
                image_ext = ext

        selected = self._select_frame_subset(frame_list)
        frames = [
            self._load_image_rgb(os.path.join(video_dir, f"{fn}{image_ext}"))
            for fn in selected
        ]

        return frames

    def _select_frame_subset(self, frame_list):
        """
        Pick `select_frame_nums` entries from *frame_list*.
        - train: random start position
        - val / test: center crop
        If there are fewer frames than needed, pad by repeating the last frame.
        """
        n = len(frame_list)
        t = self.select_frame_nums

        if n == 0:
            return []

        if n >= t:
            start = (
                random.randint(0, n - t) if self.phase == "train"
                else (n - t) // 2
            )
            return frame_list[start : start + t]

        selected = list(frame_list)
        if self.temporal_pad:
            selected += [frame_list[-1]] * (t - n)
        return selected

    # ------------------------------------------------------------------ #
    #  Video-file loader (decord)
    # ------------------------------------------------------------------ #

    def _load_frames_from_video(self, video_path):
        """
        Read frames from a video file using decord.

        Sampling strategies
        -------------------
        - 'uniform' : evenly spaced T frames across the whole video.
        - 'fps'     : sample at a fixed target FPS; train uses a random
                       start while val/test use the centre of the video.
        """
        import decord
        from decord import VideoReader, cpu

        decord.bridge.set_bridge("native")

        def _black_frames():
            return [
                Image.new("RGB", (self.w, self.h), (0, 0, 0))
                for _ in range(self.select_frame_nums)
            ]

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            video_fps = vr.get_avg_fps()
        except Exception as e:
            # print(f"Failed to load video {video_path}. Error: {e}. Returning black frames.")
            return _black_frames()

        if total == 0:
            # print(f"Video {video_path} has 0 frames. Returning black frames.")
            return _black_frames()

        indices = self._compute_video_sample_indices(total, video_fps)

        try:
            batch = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
        except Exception as e:
            print(f"Failed to decode frames from {video_path}. Error: {e}. Returning black frames.")
            return _black_frames()

        return [Image.fromarray(batch[i]) for i in range(len(batch))]

    def _compute_video_sample_indices(self, total, video_fps):
        """
        Return a numpy array of frame indices to read from the video,
        according to `self.frame_strategy` and `self.phase`.
        """
        t = self.select_frame_nums
        strategy = self.frame_strategy

        if strategy == "uniform":
            if t <= 1 or total <= 1:
                interval = 1
            else:
                interval = max(1, (total - 1) // (t - 1))
        elif strategy == "fps":
            interval = max(1, int(video_fps / self.fps))
        else:
            raise ValueError(
                f"Unknown frame strategy '{strategy}'. Expected 'uniform' or 'fps'."
            )

        max_samples = (total - 1) // interval + 1

        if max_samples >= t:
            span = interval * (t - 1) + 1
            max_start = total - span
            start = (
                random.randint(0, max_start) if self.phase == "train"
                else max_start // 2
            )
            indices = [start + i * interval for i in range(t)]
        else:
            indices = [i * interval for i in range(max_samples)]
            if self.temporal_pad:
                indices += [indices[-1]] * (t - max_samples)

        return np.array(indices, dtype=int)

    def _parse_frame_seq(self, frame_seq_str):
        return [int(i) for i in frame_seq_str[1:-1].split(",") if i.strip()]

    # ------------------------------------------------------------------ #
    #  Image I/O utility
    # ------------------------------------------------------------------ #

    def _load_image_rgb(self, image_path):
        """Load a single image as RGB. Returns a black placeholder on failure."""
        try:
            with Image.open(image_path) as img:
                if img.size[0] < 28 or img.size[1] < 28:
                    print(
                        f"Image {image_path} is too small {img.size}. "
                        "Returning black image."
                    )
                    return Image.new("RGB", (self.w, self.h), (0, 0, 0))
                return img.convert("RGB")
        except Exception as e:
            print(f"Failed to load image {image_path}. Error: {e}. Returning black image.")
            return Image.new("RGB", (self.w, self.h), (0, 0, 0))

    # ------------------------------------------------------------------ #
    #  Augmentation pipeline (train only)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_augmentation(cfg):
        """Build a composed augmentation pipeline from the config dict."""
        ops = []

        if cfg.get("aug_jpeg_compression"):  # [p, min_quality, max_quality]
            p = cfg["aug_jpeg_compression"]
            ops.append(_RandomJPEGCompression(p[0], quality=(int(p[1]), int(p[2]))))

        if cfg.get("aug_randomresizecrop"):  # [h, w, scale_min, scale_max]
            p = cfg["aug_randomresizecrop"]
            ops.append(
                v2.RandomResizedCrop(
                    (int(p[0]), int(p[1])), scale=(p[2], p[3]), ratio=(1.0, 1.0)
                )
            )

        if cfg.get("aug_randomresize"):  # [min_hw, scale_min, scale_max]
            p = cfg["aug_randomresize"]
            ops.append(_RandomResize(int(p[0]), scale=(p[1], p[2])))

        if cfg.get("aug_randomcrop"):  # [h, w]
            p = cfg["aug_randomcrop"]
            ops.append(v2.RandomCrop((int(p[0]), int(p[1]))))

        if cfg.get("aug_randomscalecrop"):  # [min_hw, scale_min, scale_max]
            p = cfg["aug_randomscalecrop"]
            ops.append(_RandomScaleCrop(int(p[0]), scale=(p[1], p[2])))

        if cfg.get("aug_randomhorizontalflip"):  # [p]
            p = cfg["aug_randomhorizontalflip"]
            ops.append(v2.RandomHorizontalFlip(p[0]))

        if cfg.get("aug_randomrotation"):  # [degrees]
            p = cfg["aug_randomrotation"]
            ops.append(v2.RandomRotation(p[0]))

        if cfg.get("aug_colorjitter"):  # [brightness, contrast, saturation]
            p = cfg["aug_colorjitter"]
            ops.append(
                v2.ColorJitter(brightness=p[0], contrast=p[1], saturation=p[2])
            )

        return v2.Compose(ops) if ops else v2.Identity()
