import torch
from torch.amp import autocast
import yaml
import argparse
import pandas as pd
from argparse import Namespace

import util
import models
from dataset import get_dataloader_dp


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', dest='checkpoint_path', help='path to model checkpoint', required=True)
    parser.add_argument('--input', dest='input_path', help='path to a single input video', required=True)
    parser.add_argument('--config', dest='config', default=None)

    args, unknown = parser.parse_known_args()
    if unknown:
        extra_args = {}
        for i in range(0, len(unknown), 2):
            key = unknown[i].lstrip('-')
            value = unknown[i + 1]
            extra_args[key] = value
        args = Namespace(**vars(args), **extra_args)
    return args


def build_inference_loader(cfg, input_path):
    sample_df = pd.DataFrame([
        {
            'content_path': input_path,
            'label': 0,
        }
    ])
    test_cfg = dict(cfg)
    test_cfg['test_batch_size'] = 1
    test_cfg['test_csv'] = sample_df
    return get_dataloader_dp(phase='test', cfg=test_cfg)


def infer_single_video(model, data_loader, cfg):
    use_bf16 = cfg.get('use_bf16', False)
    num_classes = cfg.get('num_classes', 1)

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            video_id = batch['id'][0]
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.cuda(non_blocking=True)

            with autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(**batch)

            if isinstance(outputs, dict):
                logits = outputs['cls']
            else:
                logits = outputs

            if num_classes == 1:
                fake_prob = torch.sigmoid(logits.squeeze(1))[0].item()
                pred_label = int(fake_prob >= 0.5)
                return {
                    'video': video_id,
                    'pred_label': pred_label,
                    'score': fake_prob,
                }

            probs = torch.softmax(logits, dim=1)[0].detach().float().cpu().tolist()
            pred_label = int(torch.argmax(logits, dim=1)[0].item())
            return {
                'video': video_id,
                'pred_label': pred_label,
                'score': probs[pred_label],
                'probs': probs,
            }

    raise RuntimeError("No samples were produced for inference.")


if __name__ == '__main__':
    args = get_arguments()
    cfg = {}
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    cfg = util.merge_configs(cfg, args)

    model_name = cfg.get('model')
    checkpoint_path = cfg.get('checkpoint_path')

    print(f"******* Loading model {model_name}. *******")
    if torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA devices available. This DP inference script requires at least one GPU.")
    torch.manual_seed(42)
    model = models.build_model(model_name, **cfg)
    util.load_checkpoint(model, checkpoint_path)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    test_loader = build_inference_loader(cfg, args.input_path)
    result = infer_single_video(model, test_loader, cfg)
    print(result)
