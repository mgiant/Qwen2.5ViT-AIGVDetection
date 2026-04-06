import os
import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel


DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}

CHECKPOINT = {
    "Qwen2.5-VL-3B-Instruct": "model-00001-of-00002.safetensors",
    "Qwen2.5-VL-7B-Instruct": "model-00001-of-00005.safetensors",
    "Qwen2.5-VL-32B-Instruct": "model-00001-of-00018.safetensors",
    "Qwen2.5-VL-72B-Instruct": "model-00001-of-00038.safetensors",
}


def _resolve_model_init_kwargs(kwargs):
    dtype = torch.bfloat16 if kwargs.get("use_bf16", False) else torch.float32
    head_dtype = DTYPE_MAP[kwargs.get("head_dtype", "bf16")]
    model_source = kwargs.get("model_source", "Qwen2.5-VL-3B-Instruct")
    attn_implementation = kwargs.get("attn_implementation", "flash_attention_2")
    return dtype, head_dtype, model_source, attn_implementation


def _load_visual_backbone(model_source, dtype, attn_implementation):
    checkpoint = os.path.join(model_source, CHECKPOINT[os.path.basename(model_source)])
    config = AutoConfig.from_pretrained(model_source)
    visual_model = Qwen2_5_VisionTransformerPretrainedModel._from_config(
        config.vision_config,
        dtype=dtype,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )

    visual_state_dict = {}
    replace_prefix = "visual."
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(replace_prefix):
                new_key = key[len(replace_prefix):]
                tensor = handle.get_tensor(key)
                visual_state_dict[new_key] = tensor.to(dtype)

    visual_model.load_state_dict(visual_state_dict, strict=True)
    return visual_model


class Qwen2_5_ViT(nn.Module):
    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        dtype, head_dtype, model_source, attn_implementation = _resolve_model_init_kwargs(kwargs)
        self.backbone = _load_visual_backbone(model_source, dtype, attn_implementation)
        self.head = nn.Linear(self.backbone.config.out_hidden_size, num_classes, dtype=head_dtype)

        tuning_mode = kwargs.get("tuning_mode", "default")
        if tuning_mode == "lp":
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, image_grid_hws, **kwargs):
        features = self.backbone(hidden_states=pixel_values, grid_thw=image_grid_hws).pooler_output
        patches_per_sample = (
            image_grid_hws[:, 0] * image_grid_hws[:, 1] * image_grid_hws[:, 2]
        ) // (self.backbone.config.spatial_merge_size ** 2)
        patches_per_sample = patches_per_sample.tolist()
        features_batch = torch.split(features, patches_per_sample, dim=0)
        averaged_features = torch.stack([torch.mean(feature, dim=0) for feature in features_batch])
        logits = self.head(averaged_features)

        pred_dict = {"cls": logits, "prob": logits[:, 0].sigmoid(), "feat": averaged_features}
        return pred_dict

    def load_checkpoint(self, state_dict):
        self.load_state_dict(state_dict, strict=True)


class Qwen2_5_ViT_Lora(nn.Module):
    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        dtype, head_dtype, model_source, attn_implementation = _resolve_model_init_kwargs(kwargs)
        self.backbone = _load_visual_backbone(model_source, dtype, attn_implementation)
        self.head = nn.Linear(self.backbone.config.out_hidden_size, num_classes, dtype=head_dtype)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

    def forward(self, pixel_values, image_grid_hws, **kwargs):
        features = self.backbone(hidden_states=pixel_values, grid_thw=image_grid_hws).pooler_output
        patches_per_sample = (
            image_grid_hws[:, 0] * image_grid_hws[:, 1] * image_grid_hws[:, 2]
        ) // (self.backbone.config.spatial_merge_size ** 2)
        patches_per_sample = patches_per_sample.tolist()
        features_batch = torch.split(features, patches_per_sample, dim=0)
        averaged_features = torch.stack([torch.mean(feature, dim=0) for feature in features_batch])
        logits = self.head(averaged_features)

        pred_dict = {"cls": logits, "prob": logits[:, 0].sigmoid(), "feat": averaged_features}
        return pred_dict

    def load_checkpoint(self, state_dict):
        incompatible_keys = self.load_state_dict(state_dict, strict=False)
        if incompatible_keys.unexpected_keys:
            print(f"Warning: Unexpected keys during loading: {incompatible_keys.unexpected_keys}")

    def get_checkpoint(self):
        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.cpu()
        return trainable_state_dict
