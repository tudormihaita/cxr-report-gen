import numpy as np
import torch.nn.functional as F

from sklearn import metrics
from scipy.special import softmax
from typing import Dict, List, Union, Tuple


def compute_retrieval_metrics(image_embeddings, text_embeddings, text_list):
    """
    Compute image-to-text retrieval metrics as recall@1, recall@5, recall@10 and mean rank.
    :param image_embeddings: list of all image embeddings
    :param text_embeddings: list of all text embeddings
    :param text_list: list of all original text captions in the given data split
    :return: dictionary with recall@1, recall@5, recall@10 and mean rank
    """
    identical_text_set = []

    idx2label = {}
    identical_indexes = []
    for i, text in enumerate(text_list):
        if text not in identical_text_set:
            identical_text_set.append(text)
            identical_indexes.append(i)
            idx2label[i] = len(identical_text_set) - 1
        else:
            idx2label[i] = identical_text_set.index(text)

    identical_text_embedding = text_embeddings[identical_indexes]
    num_samples = image_embeddings.shape[0]
    n_text = len(identical_text_set)

    similarities = metrics.pairwise.cosine_similarity(image_embeddings, identical_text_embedding)
    recall_dict = {1: 0, 5: 0, 10: 0}
    mean_rank = 0
    for idx in range(num_samples):
        label = idx2label[idx]
        similarity = similarities[idx]
        similarity_args = similarity.argsort()

        rank = n_text - np.argwhere(similarity_args == label).ravel()[0]
        mean_rank += rank

        for k in recall_dict:
            if rank <= k:
                recall_dict[k] += 1

    eval_metrics = {
        'r_mean': mean_rank / num_samples,
        'recall@1': (recall_dict[1] / num_samples) * 100,
        'recall@5': (recall_dict[5] / num_samples) * 100,
        'recall@10': (recall_dict[10] / num_samples) * 100
    }
    return eval_metrics


def compute_multilabel_classification_metrics_from_logits(
        predictions: np.ndarray,
        labels: np.ndarray,
        class_list: List[str],
        thresholds: List[float] = None
) -> Dict:
    """
    Compute multilabel classification metrics directly from model predictions.

    Args:
        predictions: Model predictions after sigmoid of shape (N, C)
        labels: Ground truth labels of shape (N, C) where C is number of classes
        class_list: List of class names
        thresholds: Classification thresholds (defaulted to a series of values from 0.1 to 0.9)

    Returns:
        Dictionary with per-class metrics and average metrics
    """
    if thresholds is None:
        thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

    results = {}
    optimal_thresholds = {}

    # For each class, compute binary classification metrics
    for idx, class_name in enumerate(class_list):
        # Extract predictions and labels for this class
        class_preds = predictions[:, idx]
        class_labels = labels[:, idx]

        fpr, tpr, threshold_values = metrics.roc_curve(class_labels, class_preds)
        auroc = metrics.auc(fpr, tpr)

        youen_indices = tpr - fpr
        optimal_idx = np.argmax(youen_indices)
        optimal_threshold = threshold_values[optimal_idx]
        optimal_thresholds[class_name] = optimal_threshold

        threshold_metrics = {}
        for threshold in thresholds:
            # Convert probabilities to binary predictions
            binary_preds = (class_preds > threshold).astype(int)

            # Handle edge cases where a class might have all negative examples
            if np.sum(class_labels) == 0:
                auroc = 0.0 if np.sum(binary_preds) > 0 else 1.0
            else:
                auroc = metrics.roc_auc_score(class_labels, class_preds)

            # Calculate metrics
            accuracy = metrics.accuracy_score(class_labels, binary_preds)

            # Handle edge cases for precision, recall and f1
            if np.sum(binary_preds) == 0:
                precision = 1.0 if np.sum(class_labels) == 0 else 0.0
            else:
                precision = metrics.precision_score(class_labels, binary_preds, zero_division=0)

            recall = metrics.recall_score(class_labels, binary_preds, zero_division=0)
            f1 = metrics.f1_score(class_labels, binary_preds, zero_division=0)

            # Calculate confusion matrix elements
            tn, fp, fn, tp = metrics.confusion_matrix(class_labels, binary_preds, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            threshold_metrics[f"t{threshold}"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity
            }

        optimal_binary_preds = (class_preds > optimal_threshold).astype(int)

        accuracy = metrics.accuracy_score(class_labels, optimal_binary_preds)
        precision = metrics.precision_score(class_labels, optimal_binary_preds, zero_division=0)
        recall = metrics.recall_score(class_labels, optimal_binary_preds, zero_division=0)
        f1 = metrics.f1_score(class_labels, optimal_binary_preds, zero_division=0)

        tn, fp, fn, tp = metrics.confusion_matrix(class_labels, optimal_binary_preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Store metrics for this class
        results[class_name] = {
            "auroc": auroc,
            "optimal_threshold": optimal_threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }

    # Calculate average metrics across all classes
    avg_metrics = {
        'auroc_avg': np.mean([results[c]['auroc'] for c in class_list]),
        'accuracy_avg': np.mean([results[c]['accuracy'] for c in class_list]),
        'precision_avg': np.mean([results[c]['precision'] for c in class_list]),
        'recall_avg': np.mean([results[c]['recall'] for c in class_list]),
        'specificity_avg': np.mean([results[c]['specificity'] for c in class_list]),
        'f1_avg': np.mean([results[c]['f1'] for c in class_list])
    }

    results['average'] = avg_metrics
    return results


def compute_binary_classification_metrics(
        image_embeddings,
        condition_labels,
        prompt_embeddings,
        threshold=0.5,
) -> Dict[str, float]:
    """
      Compute binary classification metrics for a single condition.

      Args:
          image_embeddings: Image embeddings of shape (N, D)
          condition_labels: Binary labels of shape (N,)
          prompt_embeddings: Embeddings for [negative, positive] prompts of shape (2, D)
          threshold: Classification threshold (default: 0.5)

      Returns:
          Dictionary with evaluation metrics
      """
    similarities = metrics.pairwise.cosine_similarity(image_embeddings, prompt_embeddings)

    probs = softmax(similarities, axis=1)
    positive_probs = probs[:, 1]
    predictions = (positive_probs > threshold).astype(int)

    auroc = metrics.roc_auc_score(condition_labels, positive_probs)
    accuracy = metrics.accuracy_score(condition_labels, predictions)
    precision = metrics.precision_score(condition_labels, predictions)
    recall = metrics.recall_score(condition_labels, predictions)
    f1 = metrics.f1_score(condition_labels, predictions)

    tn, fp, fn, tp = metrics.confusion_matrix(condition_labels, predictions).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    eval_metrics = {
        "auroc": auroc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    return eval_metrics


def compute_multilabel_classification_metrics(
        image_embeddings: np.ndarray,
        labels: np.ndarray,
        class_prompt_embeddings: Dict[str, np.ndarray],
        class_list: List[str],
        threshold: float = 0.5
) -> Dict:
    """
    Compute multilabel classification metrics for multiple conditions.

    Args:
        image_embeddings: Image embeddings of shape (N, D)
        labels: Binary labels of shape (N, C) where C is number of classes
        class_prompt_embeddings: Dict mapping each class to its prompt embeddings [negative, positive]
        class_list: List of class names
        threshold: Classification threshold (default: 0.5)

    Returns:
        Dictionary with per-class metrics and average metrics
    """
    results = {}

    # for each class, compute binary classification metrics
    for idx, class_name in enumerate(class_list):
        # extract labels for this class
        class_labels = labels[:, idx]

        # get prompt embeddings for this class
        prompt_embeddings = class_prompt_embeddings[class_name]

        class_metrics = compute_binary_classification_metrics(
            image_embeddings=image_embeddings,
            condition_labels=class_labels,
            prompt_embeddings=prompt_embeddings,
            threshold=threshold
        )

        results[class_name] = class_metrics

    avg_metrics = {
        'auroc_avg': np.mean([results[c]['auroc'] for c in class_list]),
        'accuracy_avg': np.mean([results[c]['accuracy'] for c in class_list]),
        'precision_avg': np.mean([results[c]['precision'] for c in class_list]),
        'recall_avg': np.mean([results[c]['recall'] for c in class_list]),
        'specificity_avg': np.mean([results[c]['specificity'] for c in class_list]),
        'f1_avg': np.mean([results[c]['f1'] for c in class_list])
    }

    results['average'] = avg_metrics
    return results


def compute_zeroshot_classification_metrics(
        image_embeddings: np.ndarray,
        text_embeddings_by_class: Dict[str, np.ndarray],
        labels: np.ndarray,
        class_list: List[str]
) -> Dict:
    """
    Compute zero-shot classification metrics where each class has multiple prompts.

    Args:
        image_embeddings: Image embeddings of shape (N, D)
        text_embeddings_by_class: Dictionary mapping each class to prompt embeddings (multiple per class)
        labels: Binary labels of shape (N, C) where C is number of classes
        class_list: List of class names

    Returns:
        Dictionary with per-class metrics and average metrics
    """
    class_prompt_embeddings = {}

    for class_name in class_list:
        # for each class, prepare embeddings for negative and positive prompts
        negative_prompt_emb = text_embeddings_by_class[f"No {class_name}"]
        positive_prompt_emb = text_embeddings_by_class[class_name]

        # if multiple embeddings per class, take the mean
        if len(negative_prompt_emb.shape) > 1 and negative_prompt_emb.shape[0] > 1:
            negative_prompt_emb = np.mean(negative_prompt_emb, axis=0, keepdims=True)

        if len(positive_prompt_emb.shape) > 1 and positive_prompt_emb.shape[0] > 1:
            positive_prompt_emb = np.mean(positive_prompt_emb, axis=0, keepdims=True)

        # combine into [negative, positive] format
        class_prompt_embeddings[class_name] = np.vstack([negative_prompt_emb, positive_prompt_emb])

    return compute_multilabel_classification_metrics(
        image_embeddings=image_embeddings,
        labels=labels,
        class_prompt_embeddings=class_prompt_embeddings,
        class_list=class_list
    )


def compute_cosine_similarity(image_embeds, text_embeds):
    sims = F.cosine_similarity(image_embeds, text_embeds)
    return sims.mean().item()


