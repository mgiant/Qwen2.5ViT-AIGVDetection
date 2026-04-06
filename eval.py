import csv
import datetime
import logging
import sys
import torch
import torch.nn as nn
import torch.multiprocessing.spawn
import yaml

import util
import models
from dataset import get_dataloader_ddp, calc_metrics, generate_test_csv


def get_arguments():
    import argparse
    from argparse import Namespace

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', dest='checkpoint_path', help='path to model checkpoint', required=True)
    parser.add_argument('--val', nargs='+', dest='val_datasets', required=True)

    parser.add_argument('--config', dest='config', default=None)
    parser.add_argument('--analysis', dest='analysis')
    parser.add_argument('--save-feature', dest='save_feature', action='store_true', help='save features to a pickle file')

    # parser.add_argument('--model', dest='model', default=argparse.SUPPRESS, help='model name')
    # parser.add_argument('--use_bf16', dest='use_bf16', action='store_true', default=argparse.SUPPRESS)
    # parser.add_argument('--val-batch-size', dest='val_batch_size', type=int, default=argparse.SUPPRESS)
    # parser.add_argument('--val_loader', dest='val_loader', default=argparse.SUPPRESS, help='test loader type')
    # parser.add_argument('--height', dest='height', type=int, default=argparse.SUPPRESS, help='height')
    # parser.add_argument('--width', dest='width', type=int, default=argparse.SUPPRESS, help='width')
    # parser.add_argument('--frame', dest='select_frame_nums', type=int, default=argparse.SUPPRESS, help='time')
    # parser.add_argument('--temporal-pad', dest='temporal_pad', type=bool, default=argparse.SUPPRESS)

    args, unknown = parser.parse_known_args()

    if unknown:
        extra_args = {}
        for i in range(0, len(unknown), 2):
            key = unknown[i].lstrip('-')
            value = unknown[i + 1]
            extra_args[key] = value
        args = Namespace(**vars(args), **extra_args)
    return args


def _save_analysis_csv(dataset_name, rows):
    now = datetime.datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    with open(f'{dataset_name}-{formatted_time}.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def _save_features(model_name, video_list, true_labels, outpred, features):
    now = datetime.datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    with open(f'{model_name}-{formatted_time}.pkl', 'wb') as f:
        import pickle
        pickle.dump({
            'video_list': video_list,
            'gt_label_list': true_labels,
            'pred': outpred,
            'features': features,
        }, f)


def _evaluate_one_dataset(rank, world_size, cfg, model, loss_fn, dataset_name, args):
    test_batch_size = cfg.get('test_batch_size', 32)
    test_csv, subset_list = generate_test_csv(dataset_name)

    dataset_cfg = dict(cfg)
    dataset_cfg['test_csv'] = test_csv
    dataset_cfg['test_batch_size'] = test_batch_size
    dataset_cfg['num_workers'] = max(1, test_batch_size // 8)

    test_loader = get_dataloader_ddp(
        rank=rank,
        world_size=world_size,
        phase='test',
        cfg=dataset_cfg,
    )

    if rank == 0:
        logging.info(f"******* Test on {dataset_name}. *******")
        logging.info(f"******* Testing Video IDs {len(test_loader.dataset)}, Batch size {test_batch_size} *******")

    outputs = util.eval_model(
        rank=rank,
        world_size=world_size,
        cfg=dataset_cfg,
        model=model,
        val_loader=test_loader,
        loss_fn=loss_fn,
    )

    if rank != 0:
        return None

    true_labels = outputs['true_labels']
    pred_labels = outputs['pred_labels']
    outpred = outputs['outpred']
    video_list = outputs['video_list']
    features = outputs.get('features', None)
    num_classes = cfg.get('num_classes', 1)

    acc, ap = calc_metrics(
        true_labels,
        pred_labels,
        outpred,
        video_list,
        dataset_name,
        subset_list,
        num_classes=num_classes,
    )

    if args.analysis:
        rows = zip(video_list, true_labels, outpred)
        _save_analysis_csv(dataset_name, rows)

    if args.save_feature:
        _save_features(model_name=cfg.get('model'), video_list=video_list, true_labels=true_labels, outpred=outpred, features=features)

    return acc, ap


def eval_ddp(rank, world_size, port, args):
    logging.basicConfig(format='', stream=sys.stdout, level=logging.INFO, force=True)
    util.setup(rank, world_size, port)

    cfg = {}
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    cfg = util.merge_configs(cfg, args)

    model_name = cfg.get('model')
    checkpoint_path = cfg.get('checkpoint_path')
    val_datasets = cfg.get('val_datasets') or []
    num_classes = cfg.get('num_classes', 1)

    if rank == 0:
        logging.info(f"******* Loading model {model_name}. *******")

    torch.manual_seed(42)
    model = models.build_model(model_name, **cfg)
    util.load_checkpoint(model, checkpoint_path)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    model.eval()

    all_acc = []
    all_ap = []
    for dataset_name in val_datasets:
        result = _evaluate_one_dataset(rank, world_size, cfg, model, loss_fn, dataset_name, args)
        if rank == 0 and result is not None:
            acc, ap = result
            all_acc.append(acc)
            all_ap.append(ap)

    if rank == 0 and len(all_acc) > 1:
        logging.info("******* Overall *******")
        logging.info('ACC: ' + '\t'.join(f'{n * 100:.2f}' for n in all_acc))
        logging.info(' AP: ' + '\t'.join(f'{n * 100:.2f}' for n in all_ap))
        logging.info(f'Average ACC: {sum(all_acc) / len(all_acc):.2%}, AP: {sum(all_ap) / len(all_ap):.2%}')

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main():
    args = get_arguments()
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available. This DDP evaluation script requires at least one GPU.")
    port = util.find_free_port()
    torch.multiprocessing.spawn(eval_ddp, args=(world_size, port, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
