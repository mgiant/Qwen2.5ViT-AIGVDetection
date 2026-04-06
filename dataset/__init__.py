import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .default_loader import VideoFrameDataset
from .metrics import calc_metrics, generate_test_csv
from .qwen2_5vl_loader import Dataset_Qwen25ViT


LOADER_REGISTRY = {
    "train": {"default": VideoFrameDataset, "qwen": Dataset_Qwen25ViT},
    "val": {"default": VideoFrameDataset, "qwen": Dataset_Qwen25ViT},
    "test": {"default": VideoFrameDataset, "qwen": Dataset_Qwen25ViT},
}

PHASE_DEFAULTS = {
    "train": {"shuffle": True, "drop_last": True, "persistent_workers": True},
    "val": {"shuffle": False, "drop_last": False, "persistent_workers": True},
    "test": {"shuffle": False, "drop_last": False, "persistent_workers": False},
}


def _read_csv(csv_path):
    if isinstance(csv_path, pd.DataFrame):
        df = csv_path.copy()
    elif isinstance(csv_path, list):
        df = pd.concat([pd.read_csv(path) for path in csv_path], ignore_index=True)
    else:
        df = pd.read_csv(csv_path)
    df.reset_index(drop=True, inplace=True)
    return df


def _resolve_loader_name(cfg, phase):
    name = cfg.get(f"{phase}_loader")
    if name is None and phase == "test":
        name = cfg.get("val_loader")
    return name or "default"


def _build_dataset(phase, cfg):
    df = _read_csv(cfg[f"{phase}_csv"])
    loader_name = _resolve_loader_name(cfg, phase)
    dataset = LOADER_REGISTRY[phase][loader_name](cfg, df=df, phase=phase)
    return df, dataset


def get_dataloader_dp(phase, cfg):
    assert phase in ("train", "val", "test"), f"Unknown phase: {phase}"

    df, dataset = _build_dataset(phase, cfg)
    defaults = PHASE_DEFAULTS[phase]
    batch_size = int(cfg[f"{phase}_batch_size"])
    num_workers = int(cfg.get("num_workers", 4))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=defaults["shuffle"],
        num_workers=num_workers,
        drop_last=defaults["drop_last"],
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        persistent_workers=defaults["persistent_workers"] and num_workers > 0,
    )
    print(f"******* [{phase.upper()}] samples={len(df)}  batch_size={batch_size} *******")
    return dataloader


def get_dataloader_ddp(rank, world_size, phase, cfg):
    assert phase in ("train", "val", "test"), f"Unknown phase: {phase}"

    df, dataset = _build_dataset(phase, cfg)
    defaults = PHASE_DEFAULTS[phase]
    batch_size = int(cfg[f"{phase}_batch_size"])
    num_workers = int(cfg.get("num_workers", 4))

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=defaults["shuffle"],
        drop_last=defaults["drop_last"],
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=defaults["drop_last"],
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        persistent_workers=defaults["persistent_workers"] and num_workers > 0,
    )
    if rank == 0:
        print(f"******* [{phase.upper()}] samples={len(df)}  batch_size={batch_size} *******")
    return dataloader
