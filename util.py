import torch
import torch.distributed
import models
import time
import math
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, average_precision_score, roc_auc_score, confusion_matrix
import logging
from torch.amp import autocast, GradScaler
import socket

class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-5):
        """
        Args:
            patience (int): stop after consecutive patience epochs
            min_delta (float): minimum change in the monitored quantity to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            # update min loss and counter
            self.min_validation_loss = validation_loss
            self.counter = 0
            return False # no stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True # stop
            else:
                return False # no stop

def eval_model(rank, world_size, cfg, model, val_loader, loss_fn):
    """
    Evaluate the model on the validation dataset in DDP
    """
    model.eval()

    local_val_loss = 0.0
    local_samples = 0
    local_video_ids = []
    local_true_labels = []
    local_outpred = []
    local_features = []

    pbar = tqdm(val_loader, desc="Evaluation", total=len(val_loader), ncols=75) if rank == 0 else val_loader

    with torch.no_grad():  # No need to track gradients during validation
        for batch in pbar:
            b = batch.get('image_grid_hws', batch['pixel_values']).size()[0]
            binary_label = batch['binary_label'].clone().detach()
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.cuda(non_blocking=True)
            with autocast('cuda', dtype=torch.bfloat16, enabled=cfg.get('use_bf16', False)):
                output = model(**batch)
                if type(output) is dict:
                    logits = output['cls']
                    features = output.get('feat', None)
                else:
                    logits = output
                    features = None

                if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                    loss = loss_fn(logits[:, 0], batch['binary_label'])
                elif isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    loss = loss_fn(logits, batch['label_onehot'])
                else:
                    raise ValueError
            # get prob
            num_classes = logits.shape[1]
            if num_classes == 1:
                prob_list = logits[:, 0].sigmoid().to(torch.float32).cpu().tolist()
            elif num_classes >= 2:
                prob_list = torch.softmax(logits, dim=1).to(torch.float32).cpu().tolist()

            local_val_loss += loss.item() * b
            local_samples += b

            local_video_ids.extend(batch['id'])
            local_true_labels.extend(binary_label.to(torch.int64).cpu().tolist())
            local_outpred.extend(prob_list)
            if features is not None:
                local_features.extend(features.to(torch.float32).cpu().tolist())
            else:
                local_features.extend([None] * b)
    
    # Aggregate results across all processes
    metrics_tensor = torch.tensor([local_val_loss, local_samples], dtype=torch.float64, device=rank)
    torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)

    local_results = {
        'ids': local_video_ids,
        'trues': local_true_labels,
        'probs': local_outpred,
        'featrues': local_features,  # Placeholder for features if needed
    }

    if rank > 0:
        torch.distributed.gather_object(
            local_results,
            None,
            dst=0,
        )
        return {}
    if rank == 0:
        gathered_lists = [None] * world_size
        torch.distributed.gather_object(local_results, gathered_lists, dst=0)
        global_val_loss = metrics_tensor[0].item()
        total_samples = metrics_tensor[1].item()
        avg_val_loss = global_val_loss / total_samples if total_samples > 0 else 0

        # Flatten the gathered lists
        all_video_ids = [item for sublist in [d['ids'] for d in gathered_lists] for item in sublist]
        all_true_labels = [item for sublist in [d['trues'] for d in gathered_lists] for item in sublist]
        all_features = [item for sublist in [d['featrues'] for d in gathered_lists] for item in sublist]

        all_outpred = [item for sublist in [d['probs'] for d in gathered_lists] for item in sublist]
        all_outpred_np = np.array(all_outpred)

        if all_outpred_np.ndim == 1:
            all_pred_labels = (all_outpred_np > 0.5).astype(int).tolist()
        else:
            all_pred_labels = np.argmax(all_outpred_np, axis=1).tolist()

        pred_accuracy = accuracy_score(all_true_labels, all_pred_labels)
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        return_dict = {
            'pred_accuracy': pred_accuracy,
            'video_list': all_video_ids,
            'pred_labels': all_pred_labels,
            'true_labels': all_true_labels,
            'outpred': all_outpred,
            'val_loss': avg_val_loss,
            'confusion_matrix': cm,
            'features': all_features,  # Placeholder for features if needed
        }
        return return_dict
    

def train_one_epoch(rank, world_size, epoch, model, loss_fn, scheduler, optimizer, train_loader, global_step, writer, cfg):
    use_bf16 = cfg.get('use_bf16', False)
    accumulation_steps = cfg.get('accumulation_steps', 1)
    stay_positive = cfg.get('stay_positive', False)

    model.train()
    train_loader.sampler.set_epoch(epoch)  # Set epoch for DistributedSampler

    local_train_loss = 0.0
    local_samples = 0

    # scaler = GradScaler(enabled=use_bf16)
    
    pbar = tqdm(train_loader, total=len(train_loader), ncols=75) if rank == 0 else train_loader

    for i, batch in enumerate(pbar):
        b = batch.get('image_grid_hws', batch['pixel_values']).size()[0]
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda(non_blocking=True)

        with autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):  # Automatic Mixed Precision
            output = model(**batch)
            logits = output['cls'] if type(output) is dict else output

            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                loss = loss_fn(logits[:, 0], batch['binary_label'])
            elif isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                loss = loss_fn(logits, batch['label_onehot'])
            else:
                raise ValueError

            if accumulation_steps > 1:
                loss = loss / accumulation_steps
        
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward()

        if (i+1) % accumulation_steps == 0 or (i+1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        step_loss = loss.detach().item() * accumulation_steps
        if rank == 0:
            # Log the loss of the current step to TensorBoard
            # This is the local loss on rank 0, which is a good estimate of the global loss
            writer.add_scalar('Loss/train_step', step_loss, global_step)
            
        global_step += 1

        local_train_loss += loss.detach().item() * b * accumulation_steps
        local_samples += b
        if rank == 0:
            pbar.set_postfix(loss=(local_train_loss / local_samples))

        if stay_positive:
            with torch.no_grad():
                model.module.head.weight.data.clamp_(min=0)
     
    # Average the loss across all processes
    metrics_tensor = torch.tensor([local_train_loss, local_samples], dtype=torch.float64, device=rank)
    torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)
    global_avg_train_loss = metrics_tensor[0].item() / metrics_tensor[1].item()

    if rank == 0:
        # Log the accurate, globally-averaged loss for the epoch
        writer.add_scalar('Loss/train_epoch', global_avg_train_loss, epoch)
        
        # Also log the learning rate at the end of the epoch
        # This is useful to see how the scheduler affects the LR epoch by epoch
        writer.add_scalar('Learning_Rate/epoch', scheduler.get_last_lr()[0], epoch)

    scheduler.step()
    torch.cuda.empty_cache()
    return global_avg_train_loss, global_step


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # 加载文件
    if 'model_state_dict' in checkpoint:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
    else:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_checkpoint(state_dict)


def save_checkpoint(model, checkpoint_path):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if hasattr(model, 'get_checkpoint'):
        state_dict = model.get_checkpoint()
    else:
        state_dict = model.state_dict()
    torch.save(
        {'model_state_dict': state_dict},
        checkpoint_path
    )


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  
        return s.getsockname()[1] 


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size, device_id=torch.device(rank))


def merge_configs(cfg, args):
    """
    Merge command line arguments into the configuration dictionary.
    """
    final_config = cfg.copy() if cfg else dict()
    final_config.update(vars(args))
    
    return final_config
