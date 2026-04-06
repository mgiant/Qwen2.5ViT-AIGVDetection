import importlib


MODEL_CONFIG = {
    "Qwen2_5-ViT": (".Qwen2_5VL_ViT", "Qwen2_5_ViT"),
    "Qwen2_5-ViT-Lora": (".Qwen2_5VL_ViT", "Qwen2_5_ViT_Lora"),
    "NPR": (".NPR", "resnet50_npr"),
}


def build_model(model_name, **model_kwargs):
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_CONFIG.keys())}")

    module_rel_path, class_name = MODEL_CONFIG[model_name]
    module = importlib.import_module(module_rel_path, package=__package__)
    model_class = getattr(module, class_name)
    return model_class(**model_kwargs)
