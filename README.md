# Preserving Forgery Artifacts: AI-Generated Video Detection at Native Scale
[[📑**Paper**](https://openreview.net/forum?id=XD43lfRCg6) | [📑**arXiv**](https://arxiv.org/abs/2604.04634) | [🔗**Dataset**](https://huggingface.co/datasets/mgiant/magic_videos) | [🚀**Checkpoint**](#model-checkpoints)]

# Data Preparation
See `scripts/video2frame_*.sh` for dataset-specific preparation instructions.

## Evaluation Set
Magic Videos (Ours): https://huggingface.co/datasets/mgiant/magic_videos  
GenVideo-Val: https://modelscope.cn/datasets/cccnju/Gen-Video/files  
DVF: https://github.com/SparkleXFantasy/MM-Det  
DeepTraceReward: https://huggingface.co/datasets/DeeptraceReward/RewardData

## Training Set
VBench sampled videos: https://github.com/Vchitect/VBench/tree/master/sampled_videos  
Kinetics: https://github.com/cvdfoundation/kinetics-dataset

# Model Checkpoints

[🚀Qwen2.5ViT-448p](https://drive.google.com/file/d/155ywOZpTu69tm6khb54gYNcOnHQlhyBi)  
[🚀Qwen2.5ViT-720p](https://drive.google.com/file/d/1XKOlVLgFK--dlTNC71O0pHcuhwmx0_Z6)

# Example Commands

```bash
python train.py --config configs/Qwen2.5-ViT/140k/448p.yaml
python eval.py --config configs/Qwen2.5-ViT/720p.yaml --checkpoint weights/Qwen2.5-ViT_140k_224p_720p.pth --val magic2 genvideo
python test.py --config <path-to-config> --checkpoint <path-to-weight> --input /path/to/video.mp4
```
# Citation
If you find our work useful, please cite our paper:
```
@inproceedings{
li2026preserving,
title={Preserving Forgery Artifacts: {AI}-Generated Video Detection at Native Scale},
author={Zhengcen Li and Chenyang Jiang and Hang Zhao and Shiyang Zhou and Yunyang Mo and Feng Gao and Fan Yang and Qiben Shan and Shaocong Wu and Jingyong Su},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=XD43lfRCg6}
}
```

# Acknowledgements
We thank [DeMamba](https://github.com/chenhaoxing/DeMamba) and [ShareGPT4Video](https://github.com/ShareGPT4Omni/ShareGPT4Video) for their excellent work.
