"""Evaluation helpers for classification metrics."""

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


def evaluate_model(model, dataloader, device, class_names):
    """Compute top-1, top-5, macro metrics, and per-image predictions."""

    model.eval()
    all_labels = []
    all_preds_top1 = []
    all_preds_top5 = []
    all_probs_top1 = []
    start_time = time.time()

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            top1_probs, top1_preds = torch.max(probabilities, dim=1)
            _, top5_preds = torch.topk(probabilities, k=5, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds_top1.extend(top1_preds.cpu().numpy())
            all_preds_top5.extend(top5_preds.cpu().numpy())
            all_probs_top1.extend(top1_probs.cpu().numpy())

    total_time = time.time() - start_time

    all_labels = np.array(all_labels)
    all_preds_top1 = np.array(all_preds_top1)
    all_preds_top5 = np.array(all_preds_top5)
    all_probs_top1 = np.array(all_probs_top1)

    top1_acc = accuracy_score(all_labels, all_preds_top1)
    top5_acc = np.mean([true_label in predicted_top5 for true_label, predicted_top5 in zip(all_labels, all_preds_top5)])

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds_top1,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "num_samples": len(all_labels),
        "total_inference_time_sec": total_time,
        "inference_time_per_image_sec": total_time / len(all_labels),
    }

    predictions_df = pd.DataFrame(
        {
            "true_label": all_labels,
            "pred_label": all_preds_top1,
            "top1_confidence": all_probs_top1,
            "true_class_name": [class_names[i] for i in all_labels],
            "pred_class_name": [class_names[i] for i in all_preds_top1],
            "correct": all_labels == all_preds_top1,
        }
    )

    return metrics, predictions_df
