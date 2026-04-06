import warnings
from sklearn.metrics import f1_score, accuracy_score, recall_score, average_precision_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import logging
import os
import numpy as np
from collections import defaultdict

def _calc_metrics_genvideo(video_list, pred_labels, true_labels, outpred, subset_list):
    video_arr = np.array(video_list)
    pred_arr = np.array(pred_labels)
    true_arr = np.array(true_labels)
    out_arr = np.array(outpred)

    # extract idxs of real samples
    # real_mask = np.array(['real' in v for v in video_arr])
    real_mask = true_arr == 0
    real_pred = pred_arr[real_mask]
    real_true = true_arr[real_mask]
    real_out = out_arr[real_mask]

    subset_masks = {
        subset: np.array([subset in v for v in video_arr])
        for subset in subset_list
    }

    recalls, f1s, aps = [], [], []

    for subset in subset_list:
        fake_pred = pred_arr[subset_masks[subset]]
        fake_true = true_arr[subset_masks[subset]]
        fake_out = out_arr[subset_masks[subset]]

        combined_true = np.concatenate([real_true, fake_true])
        combined_pred = np.concatenate([real_pred, fake_pred])
        combined_out = np.concatenate([real_out, fake_out])

        recall = recall_score(combined_true, combined_pred)
        f1 = f1_score(combined_true, combined_pred)
        ap = average_precision_score(combined_true, combined_out)
        cm = confusion_matrix(combined_true, combined_pred)

        logging.info(f'--------------- {subset} ---------------')
        logging.info(f'Recall: {recall:.2%}, F1: {f1:.2%}, AP: {ap:.2%}')
        logging.info(cm)

        recalls.append(recall)
        f1s.append(f1)
        aps.append(ap)

    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    avg_ap = np.mean(aps)

    logging.info(f'--------------- Average ---------------')
    logging.info('\t'.join(f'{r * 100:.2f}' for r in recalls))
    logging.info('\t'.join(f'{f * 100:.2f}' for f in f1s))
    logging.info('\t'.join(f'{a * 100:.2f}' for a in aps))
    logging.info(f'Average Recall: {avg_recall:.2%}, Average F1: {avg_f1:.2%}, Average AP: {avg_ap:.2%}')

    overall_acc = accuracy_score(true_arr, pred_arr)
    overall_ap = average_precision_score(true_arr, out_arr)
    overall_recall = recall_score(true_arr, pred_arr)

    logging.info(f'--------------- Overall ---------------')
    logging.info(f'Overall ACC: {overall_acc:.2%}, AP: {overall_ap:.2%}, Recall: {overall_recall:.2%}')

    return overall_acc, overall_ap

def _calc_metrics_dvf(video_list, pred_labels, true_labels, outpred, subset_list):
    #  subset_list = ['zeroscope', 'opensora', 'videocrafter1', 'sora', 'pika', 'stablediffusion', 'stablevideo']
    
    # real_pred_accuracy = []
    real_pred_labels = []
    real_true_labels = []
    real_outpred = []
    for i in range(len(video_list)):
        if 'real' in video_list[i]:
            real_pred_labels.append(pred_labels[i])
            real_true_labels.append(true_labels[i])
            real_outpred.append(outpred[i])
    all_auc = []
    all_acc = []
    all_ap = []
    for subset in subset_list:
        fake_pred_labels = []
        fake_true_labels = []
        fake_outpred = []
        for i in range(len(video_list)):
            if subset in video_list[i]:
                fake_pred_labels.append(pred_labels[i])
                fake_true_labels.append(true_labels[i])
                fake_outpred.append(outpred[i])
        logging.info(f'--------------- {subset} ---------------')

        recall = recall_score(
            real_true_labels+fake_true_labels, real_pred_labels+fake_pred_labels)
        auc = roc_auc_score(
            real_true_labels+fake_true_labels, real_outpred+fake_outpred)
        cm = confusion_matrix(
            real_true_labels+fake_true_labels, real_pred_labels+fake_pred_labels)
        ap = average_precision_score(
            real_true_labels+fake_true_labels, real_outpred+fake_outpred)
        f1 = f1_score(
            real_true_labels+fake_true_labels, real_pred_labels+fake_pred_labels)
        acc = accuracy_score(real_true_labels+fake_true_labels, real_pred_labels+fake_pred_labels)
        logging.info(f'AUC:{auc:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}, AP: {ap:.2%}')
        logging.info(cm)

        all_auc.append(auc)
        all_acc.append(acc)
        all_ap.append(ap)
    logging.info(f'Mean AUC:{sum(all_auc)/len(all_auc):.2%}')

    return sum(all_acc) / len(all_acc), sum(all_ap) / len(all_ap)

def _calc_metrics_subset(image_list, pred_labels, true_labels, outpred, subset_list):
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    outpred = np.array(outpred)
    
    subset_to_indices = defaultdict(list)
    for idx, img_path in enumerate(image_list):
        for subset in subset_list:
            if subset in img_path:
                subset_to_indices[subset].append(idx)

    all_acc = []
    all_ap = []

    for subset in subset_list:
        indices = subset_to_indices.get(subset, [])
        
        if not indices:
            logging.warning(f'--------------- {subset}: No samples found! ---------------')
            continue

        # 使用 NumPy 高效提取子集
        s_true = true_labels[indices]
        s_pred = pred_labels[indices]
        s_out = outpred[indices]

        logging.info(f'--------------- {subset} ---------------')
        
        acc = accuracy_score(s_true, s_pred)
        ap = average_precision_score(s_true, s_out)
        # except ValueError:
        #     ap = 0.0
        #     logging.warning(f"Could not calculate AP for {subset} (possibly only one class present).")
            
        cm = confusion_matrix(s_true, s_pred)
        
        logging.info(f'ACC: {acc:.2%}, AP: {ap:.2%}')
        logging.info(f'{cm}')

        all_acc.append(acc)
        all_ap.append(ap)

    if not all_acc:
        return 0.0

    logging.info(f'--------------- Overall AP ---------------')
    logging.info('\t'.join(f'{n*100:.2f}' for n in all_ap))
    logging.info(f'--------------- Overall ACC ---------------')
    logging.info('\t'.join(f'{n*100:.2f}' for n in all_acc))
    
    average_acc = np.mean(all_acc)
    average_ap = np.mean(all_ap)
    
    logging.info(f'--------------- Average ---------------')
    logging.info(f'Average ACC: {average_acc:.2%}, Average AP: {average_ap:.2%}')

    return average_acc, average_ap

def _calc_metrics_magic(video_list, pred_labels, true_labels, outpred, subset_list):
    """Calculate per-subset and overall metrics for real/fake video classification."""
    all_acc = []
    all_ap = []

    # Single pass through video_list to separate real samples and
    # pre-build per-subset index lists, avoiding repeated full iterations.
    real_mixkit = {'pred': [], 'true': [], 'outpred': []}
    real_pexels = {'pred': [], 'true': [], 'outpred': []}
    subset_indices = {subset: [] for subset in subset_list}

    for i, (video, label) in enumerate(zip(video_list, true_labels)):
        # Only consider samples with label == 0 as real candidates
        if label == 0:
            if 'mixkit' in video or 'mixkit' in video:
                real_mixkit['pred'].append(pred_labels[i])
                real_mixkit['true'].append(true_labels[i])
                real_mixkit['outpred'].append(outpred[i])
                continue
            if 'pexels' in video:
                real_pexels['pred'].append(pred_labels[i])
                real_pexels['true'].append(true_labels[i])
                real_pexels['outpred'].append(outpred[i])
                continue

        # Match fake samples to their subset (each video belongs to at most one)
        for subset in subset_list:
            if subset in video:
                subset_indices[subset].append(i)
                break

    for subset in subset_list:
        # Select the matching real group based on subset source
        real = real_pexels if 'pexels' in subset else real_mixkit

        # Gather fake samples using pre-built indices
        indices = subset_indices[subset]
        fake_pred = [pred_labels[i] for i in indices]
        fake_true = [true_labels[i] for i in indices]
        fake_out  = [outpred[i]     for i in indices]

        # Merge real and fake samples
        merged_true = real['true']    + fake_true
        merged_pred = real['pred']    + fake_pred
        merged_out  = real['outpred'] + fake_out

        # Compute metrics
        acc    = accuracy_score(merged_true, merged_pred)
        recall = recall_score(merged_true, merged_pred, average='macro')
        cm     = confusion_matrix(merged_true, merged_pred)
        ap     = average_precision_score(merged_true, merged_out)

        logging.info(f'--------------- {subset} ---------------')
        logging.info(f'ACC: {acc:.2%}, AP: {ap:.2%}, Recall: {recall:.2%}')
        logging.info(cm)

        all_acc.append(acc)
        all_ap.append(ap)

    # Log overall results
    logging.info('--------------- Overall AP ---------------')
    logging.info('\t'.join(f'{n * 100:.2f}' for n in all_ap))
    logging.info('--------------- Overall ACC ---------------')
    logging.info('\t'.join(f'{n * 100:.2f}' for n in all_acc))

    average_acc = sum(all_acc) / len(all_acc)
    average_ap  = sum(all_ap)  / len(all_ap)
    logging.info('--------------- Average ---------------')
    logging.info(f'Average ACC: {average_acc:.2%}, Average AP: {average_ap:.2%}')

    return average_acc, average_ap


METRICS_MAP = {
    'magic': {
        'csv': 'data/magic/all.csv',
        'subset': ['wan2.1', 'wan1.3B_pexels', 'hailuo', 'jimeng2.0', 'jimeng3.0', 'stepvideo'],
        'func': _calc_metrics_magic,
    },
    'magic_mp4': {
        'csv': 'data/videos/magic/all.csv',
        'subset': ['wanx2.1', 'wan2.1-T2V-1.3B', 'hailuo', 'jimeng-S2.0', 'jimeng-S3.0', 'step-video'],
        'func': _calc_metrics_magic,
    },
    'genvideo': {
        'csv': ['data/genvideo_val/all.csv'],
        'subset': ['sora', 'morphstudio', 'gen2', 'hotshot', 'lavie', 'show1', 'moonvalley', 'crafter', 'modelscope', 'wildscrape'],
        'func': _calc_metrics_genvideo,
    },
    'genvideo_mp4': {
        'csv': ['data/videos/GenVideo-Val/all.csv'],
        'subset': ['HotShot', 'Lavie', 'Crafter', 'MoonValley', 'Gen2', 'Sora', 'WildScrape', 'ModelScope', 'Show_1', 'MorphStudio'],
        'func': _calc_metrics_genvideo,
    },
    'dvf': {
        'csv': 'data/dvf_test_images/all.csv',
        'subset': ['zeroscope', 'opensora', 'videocrafter1', 'sora', 'pika', 'stablediffusion', 'stablevideo'],
        'func': _calc_metrics_dvf,
    },
    'reward': {
        'csv': 'data/reward_data/all.csv',
        'subset': [],
    },
}

def generate_test_csv(val_csv):
    if val_csv in METRICS_MAP:
        return METRICS_MAP[val_csv]['csv'], METRICS_MAP[val_csv]['subset']
    else:
        return val_csv, []


def calc_metrics(true_labels, pred_labels, outpred, video_list, val_csv, subset_list, num_classes=1):

    if num_classes == 2:
        outpred = outpred[:, 1]
            
    if subset_list:
        func = METRICS_MAP[val_csv]['func']
        acc, ap = func(video_list, pred_labels, true_labels, outpred, subset_list)
    else:
        if num_classes <= 2:                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                acc = accuracy_score(true_labels, pred_labels)
                recall = recall_score(true_labels, pred_labels, average='macro')
                auc = roc_auc_score(true_labels, outpred)
                cm = confusion_matrix(true_labels, pred_labels)
                ap = average_precision_score(true_labels, outpred)
                f1 = f1_score(true_labels, pred_labels)
                tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
                fake_acc = tp / (tp + fn) 
                real_acc = tn / (tn + fp)
            logging.info(f'--------------- Overall ---------------')
            logging.info(f'Overall Recall: {recall:.2%}, F1: {f1:.2%}, AP: {ap:.2%}, ACC: {acc:.2%}, AUC: {auc:.2%}')
            logging.info(cm)
            logging.info(f'Real ACC: {real_acc:.2%}, Fake ACC: {fake_acc:.2%}')
        else:
            logging.info(f'--------------- Overall ---------------')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                acc = accuracy_score(true_labels, pred_labels)
                recall = recall_score(true_labels, pred_labels, average='macro')
                # auc = roc_auc_score(true_labels, outpred)
                cm = confusion_matrix(true_labels, pred_labels)
                true_labels_binarized = label_binarize(true_labels, classes=range(num_classes))
                ap = average_precision_score(true_labels_binarized, outpred, average='macro')
                # f1 = f1_score(true_labels, pred_labels)
                # tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
                # fake_acc = tp / (tp + fn) 
                # real_acc = tn / (tn + fp)
            
            logging.info(f'Overall Recall: {recall:.2%}, AP: {ap:.2%}, ACC: {acc:.2%}')
            logging.info(cm)

    return acc, ap
