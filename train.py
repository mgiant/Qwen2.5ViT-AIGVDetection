import argparse
from argparse import Namespace
import torch.distributed
import yaml
import torch
import os
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime
import shutil

from util import train_one_epoch, EarlyStopper, eval_model, save_checkpoint, find_free_port, setup, load_checkpoint, merge_configs
from dataset import get_dataloader_ddp
import models
from dataset import calc_metrics, generate_test_csv


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of detector in yaml format')
    parser.add_argument('--debug', dest='debug', action='store_true')
    args, unknown = parser.parse_known_args()

    # Parse unknown arguments as key-value overrides.
    if unknown:
        # print(f"Unknown arguments: {known}")
        extra_args = {}
        for i in range(0, len(unknown), 2):
            key = unknown[i].lstrip('-')
            value = unknown[i+1]
            extra_args[key] = value
        args = Namespace(**vars(args), **extra_args)
    return args

def set_logging(save_path, debug=False):
    if debug:
        save_path = './results/debug'
    else:
        now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%d_%H:%M:%S')
        save_path = os.path.join(save_path, formatted_time)
    
    os.makedirs(save_path, exist_ok=True)

    log_format = '[ %(asctime)s ] %(message)s'
    date_format = '%Y/%m/%d %H:%M:%S'  # Custom timestamp format.
    logging.basicConfig(format=log_format, datefmt=date_format, level=logging.INFO)
    handler = logging.FileHandler(os.path.join(save_path, 'log.txt'), mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter(log_format, date_format))
    logging.getLogger().addHandler(handler)
    
    return save_path

def train(rank, world_size, port, cfg, args):
    setup(rank, world_size, port)
    tensorboard_writer = None
    if rank == 0:
        save_dir = set_logging(cfg['save_dir'], args.debug)
        shutil.copy(args.config, os.path.join(save_dir, 'config.yaml'))
        logging.info("******* Building models. *******")
        tensorboard_writer = SummaryWriter(save_dir)
    else:
        save_dir = ''
    sync_data = [save_dir]
    torch.distributed.broadcast_object_list(sync_data, src=0)
    save_dir = sync_data[0]
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    model = models.build_model(cfg['model'], **cfg)
    model = model.to(rank)

    # Resume training or initialize from a pretrained checkpoint.
    checkpoint_path = cfg.get('checkpoint', None)
    if checkpoint_path:
        if rank == 0:
            logging.info(f"******* Loading pretrained checkpoint from {checkpoint_path}. *******")
        load_checkpoint(model, checkpoint_path)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=cfg.get('find_unused_parameters', False),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 1e-8))
    scheduler_name = cfg.get('scheduler', 'multistep')
    if scheduler_name == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=[cfg['max_epoch']-5], gamma=0.1)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg['max_epoch'], eta_min=1e-7)
    else:
        raise ValueError
    num_classes = cfg.get('num_classes', 1)
    loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss() 
    
    if rank == 0:
        logging.info("******* Building datasets. *******")
    train_loader = get_dataloader_ddp(rank, world_size, phase='train', cfg=cfg)
    val_loader = get_dataloader_ddp(rank, world_size, phase='val', cfg=cfg)
    if rank == 0:
        logging.info(f"******* Training Video IDs {len(train_loader.dataset)} Training Batch size {cfg['train_batch_size']} *******")
        logging.info(f"******* Validation Video IDs {len(val_loader.dataset)} Validation Batch size {cfg['val_batch_size']} *******")
    
    # Build the early stopper on rank 0 only.
    early_stopper = None
    if rank == 0:
        early_stopper = EarlyStopper(patience=5)

    max_epoch, max_acc = 0, 0
    global_step = 0
    for epoch in range(0, cfg['max_epoch']):
        # Training
        if rank == 0:
            logging.info(f"******* Training epoch {epoch}, *******")
        avg_train_loss, global_step = train_one_epoch(
            rank, world_size, epoch, model, loss_fn, scheduler, optimizer, train_loader, 
            global_step, tensorboard_writer, cfg
        )
        if rank == 0:
            logging.info(f"Epoch {epoch} | Avg Train Loss: {avg_train_loss:.4f}")
        # Evaluation
        if (epoch + 1) % cfg.get('eval_interval', 1) == 0 or epoch == cfg['max_epoch'] - 1:
            eval_results = eval_model(
                rank, world_size, cfg, model, val_loader, loss_fn
                )
            
            stop_signal = torch.tensor(0.0, device=rank)
            if rank == 0:
                val_loss = eval_results['val_loss']
                cm = eval_results['confusion_matrix']
                acc = eval_results['pred_accuracy']
                logging.info(f"Epoch {epoch} | ACC: {acc:.2%} Validation Loss: {val_loss:.4f}, Confusion Matrix: \n{cm}")
                tensorboard_writer.add_scalar('Loss/val_epoch', val_loss, epoch)
                # Save checkpoints.
                if acc > max_acc:
                    max_epoch, max_acc = epoch, acc
                    save_checkpoint(model, os.path.join(save_dir, f"best_acc.pth"))
                save_checkpoint(model, os.path.join(save_dir, f"last.pth"))
                # Check early stopping.
                if early_stopper and early_stopper(val_loss):
                    logging.info(f"Early stopping at epoch {epoch}.")
                    stop_signal.fill_(1.0)
        
            # Broadcast the stop signal to all processes.
            torch.distributed.broadcast(stop_signal, src=0)
            if stop_signal.item() > 0:
                break
    if rank == 0:
        logging.info(f"Training finished. Best epoch: {max_epoch}, Best accuracy: {max_acc}")
    
    if 'test_csv' in cfg:
        test_csv_list = cfg['test_csv'] if isinstance(cfg['test_csv'], list) else [cfg['test_csv']]
        all_acc = []
        all_ap = []
        for test_name in test_csv_list:
            best_checkpoint_path = os.path.join(save_dir, f"best_acc.pth")
            load_checkpoint(model.module, best_checkpoint_path)
            test_csv, subset_list = generate_test_csv(test_name)
            cfg['val_csv'] = test_csv
            test_loader = get_dataloader_ddp(rank, world_size, phase='val', cfg=cfg)
            eval_results = eval_model(rank, world_size, cfg, model, test_loader, loss_fn)
            if rank == 0:
                logging.info(f"******* Evaluate on {test_name} *******")
                true_labels = eval_results['true_labels']
                pred_labels = eval_results['pred_labels']
                outpred = eval_results['outpred']
                video_list = eval_results['video_list']
        
                acc, ap = calc_metrics(true_labels, pred_labels, outpred, video_list, test_name, subset_list, num_classes=num_classes)
                all_acc.append(acc)
                all_ap.append(ap)
        if rank == 0 and len(all_acc) > 1:
            logging.info(f"******* Overall *******")
            logging.info(test_csv_list)
            logging.info('ACC: ' + '\t'.join(f'{n*100:.2f}' for n in all_acc))
            logging.info('AP: ' + '\t'.join(f'{n*100:.2f}' for n in all_ap))
            logging.info(f'Average ACC: {sum(all_acc)/len(all_acc):.2%}, AP: {sum(all_ap)/len(all_ap):.2%}')

    # Clean up distributed state.
    torch.distributed.destroy_process_group()

if __name__ == '__main__': 
    args = get_arguments()
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg = merge_configs(cfg, args)

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available. This training script requires at least one GPU.")
    port = find_free_port()
    torch.multiprocessing.spawn(train, args=(world_size, port, cfg, args), nprocs=world_size, join=True)
    
