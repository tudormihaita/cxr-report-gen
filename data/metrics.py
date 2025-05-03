import torch
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, confusion_matrix

def compute_recall_at_k(image_embeds, text_embeds, k=1):
    """
    Compute recall at k for image-text retrieval.
    :param image_embeds: Image embeddings
    :param text_embeds: Text embeddings
    :param k: The number of top results to consider for recall
    :return: Recall@k
    """
    sims = image_embeds @ text_embeds.T
    ranks = sims.argsort(dim=-1, descending=True)
    labels = torch.arange(image_embeds.size(0), device=image_embeds.device)
    hits = (ranks[:, :k] == labels.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()

def compute_retrieval_accuracy(image_embeds, text_embeds):
    top_k = [1, 5]
    sims = image_embeds @ text_embeds.T
    ranks = sims.argsot(dim=-1, descending=True)

    labels = torch.arange(image_embeds.size(0), device=image_embeds.device)
    acc_at_k = {}

    for k in top_k:
        hits = (ranks[:, :k] == labels.unsqueeze(1)).any(dim=1).float()
        acc_at_k[f"acc@{k}"] = hits.mean().item()

    return acc_at_k

def compute_cosine_similarity(image_embeds, text_embeds):
    sims = F.cosine_similarity(image_embeds, text_embeds)
    return sims.mean().item()


def compute_auroc(y_pred, y_true, labels):
    return {
        label: roc_auc_score(y_true[:, i], y_pred[:, i])
        for i, label in enumerate(labels)
    }

def compute_auprc(y_pred, y_true, labels):
    return {
        label: average_precision_score(y_true[:, i], y_pred[:, i])
        for i, label in enumerate(labels)
    }

def compute_f1(y_pred, y_true, labels, thresholds):
    binarized = (y_pred >= thresholds).astype(int)

    return {
        label: f1_score(y_true[:, i], binarized[:, i])
        for i, label in enumerate(labels)
    }

def compute_mcc(y_pred, y_true, labels, thresholds):
    binarized = (y_pred >= thresholds).astype(int)

    return {
        label: matthews_corrcoef(y_true[:, i], binarized[:, i])
        for i, label in enumerate(labels)
    }

def compute_confusion_matrix(y_pred, y_true, labels, thresholds):
    binarized = (y_pred >= thresholds).astype(int)

    return {
        label: confusion_matrix(y_true[:, i], binarized[:, i])
        for i, label in enumerate(labels)
    }