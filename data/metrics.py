import torch
import numpy as np
import torch.nn.functional as F

from sklearn import metrics
from scipy.special import softmax
from typing import Dict, List

from utils.logger import LoggerManager
log = LoggerManager.get_logger(__name__)


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

def compute_label_aware_retrieval_metrics(
        image_embeddings,
        text_embeddings,
        labels,
        threshold=0.5
):
    """
    Compute label-aware retrieval metrics that consider semantic similarity based on medical labels.
    :param image_embeddings: list of all image embeddings
    :param text_embeddings: list of all text embeddings
    :param labels: list of all labels (binary) for the images
    :param threshold: threshold for Jaccard similarity to consider a match
    :return: dictionary with label-aware recall@1, recall@5, recall@10 and mean rank
    """
    def jaccard_similarity(a, b):
        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return intersection / (union + 1e-8)

    num_samples = image_embeddings.shape[0]
    embedding_similarities = metrics.pairwise.cosine_similarity(image_embeddings, text_embeddings)
    label_similarities = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(num_samples):
            label_similarities[i, j] = jaccard_similarity(labels[i], labels[j])

    k_values = [1, 5, 10]
    recall_dict = {k: 0 for k in k_values}
    mean_rank = 0

    for idx in range(num_samples):
        combined_similarity = 0.7 * embedding_similarities[idx] + 0.3 * label_similarities[idx]
        combined_similarity[idx] = -1

        ranked_indices = np.argsort(-combined_similarity)
        relevant_ranks = []
        for rank, candidate_idx in enumerate(ranked_indices):
            if label_similarities[idx, candidate_idx] >= threshold:
                relevant_ranks.append(rank + 1)  # +1 for 1-based rank

        if not relevant_ranks:
            first_relevant_rank = num_samples
        else:
            first_relevant_rank = min(relevant_ranks)

        mean_rank += first_relevant_rank

        for k in k_values:
            top_k_indices = ranked_indices[:k]
            if any(label_similarities[idx, i] >= threshold for i in top_k_indices):
                recall_dict[k] += 1

    eval_metrics = {
        'label_aware_mean_rank': mean_rank / num_samples,
    }
    for k in k_values:
        eval_metrics[f'label_aware_recall@{k}'] = (recall_dict[k] / num_samples) * 100

    return eval_metrics


def compute_zeroshot_classification_metrics_from_embeddings(
        image_embeddings,
        class_embeddings,
        labels,
        class_list):
    """
    Compute zero-shot classification metrics using optimized thresholds per class.

    :param image_embeddings: Image embeddings of shape (n_samples, embedding_dim)
    :param class_embeddings: Dict mapping each class to prompt embeddings
    :param labels: Ground truth labels, shape (n_samples, n_classes)
    :param class_list: List of class names
    :return: Dictionary with per-class metrics and average metrics
    """
    if isinstance(class_embeddings, dict):
        class_emb_array = np.vstack([class_embeddings[c] for c in class_list])
    else:
        class_emb_array = class_embeddings

    similarities = metrics.pairwise.cosine_similarity(image_embeddings, class_emb_array)
    optimal_thresholds = compute_optimal_thresholds(similarities, labels, class_list)

    results = {}
    for i, class_name in enumerate(class_list):
        class_scores = similarities[:, i]
        class_labels = labels[:, i]
        threshold = optimal_thresholds[class_name]

        preds = (class_scores >= threshold).astype(int)

        accuracy = metrics.accuracy_score(class_labels, preds)
        precision = metrics.precision_score(class_labels, preds, zero_division=0)
        recall = metrics.recall_score(class_labels, preds, zero_division=0)
        f1 = metrics.f1_score(class_labels, preds, zero_division=0)
        auroc = metrics.roc_auc_score(class_labels, class_scores) if len(np.unique(class_labels)) > 1 else 0.5

        tn, fp, fn, tp = metrics.confusion_matrix(class_labels, preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        results[class_name] = {
            "threshold": threshold,
            "auroc": auroc,
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


def compute_zeroshot_classification_metrics_from_logits(
        image_embeddings: np.ndarray,
        text_embeddings_by_class: Dict[str, np.ndarray],
        labels: np.ndarray,
        class_list: List[str]
) -> Dict:
    """
    Compute zero-shot classification metrics where each class has multiple prompts.

    :param image_embeddings: Image embeddings of shape (N, D)
    :param text_embeddings_by_class: Dictionary mapping each class to prompt embeddings (multiple per class)
    :param labels: Binary labels of shape (N, C) where C is number of classes
    :param class_list: List of class names
    :return: Dictionary with per-class metrics and average metrics
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


def compute_multilabel_classification_metrics_from_logits(
        predictions: np.ndarray,
        labels: np.ndarray,
        class_list: List[str],
        thresholds: List[float] = None
) -> Dict:
    """
    Compute multilabel classification metrics directly from model predictions.

    :param predictions: Model predictions after sigmoid of shape (N, C)
    :param labels: Ground truth labels of shape (N, C) where C is number of classes
    :param class_list: List of class names
    :param thresholds: Classification thresholds (defaulted to a series of values from 0.1 to 0.9)
    :return: Dictionary with per-class metrics and average metrics
    """
    if thresholds is None:
        thresholds = np.arange(0.2, 0.6, 0.1)

    results = {}
    optimal_thresholds = compute_optimal_thresholds(predictions, labels, class_list)

    # for each class, compute binary classification metrics
    for idx, class_name in enumerate(class_list):
        # extract predictions and labels for this class
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
            binary_preds = (class_preds > threshold).astype(int)

            # handle edge cases where a class might have all negative examples
            if np.sum(class_labels) == 0:
                auroc = 0.0 if np.sum(binary_preds) > 0 else 1.0
            else:
                auroc = metrics.roc_auc_score(class_labels, class_preds)

            accuracy = metrics.accuracy_score(class_labels, binary_preds)

            # handle edge cases for precision, recall and f1
            if np.sum(binary_preds) == 0:
                precision = 1.0 if np.sum(class_labels) == 0 else 0.0
            else:
                precision = metrics.precision_score(class_labels, binary_preds, zero_division=0)

            recall = metrics.recall_score(class_labels, binary_preds, zero_division=0)
            f1 = metrics.f1_score(class_labels, binary_preds, zero_division=0)

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

    :param image_embeddings: Image embeddings of shape (N, D)
    :param condition_labels: Binary labels of shape (N,)
    :param prompt_embeddings: Embeddings for [negative, positive] prompts of shape (2, D)
    :param threshold: Classification threshold (default: 0.5)
    :return: Dictionary with evaluation metrics
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

    :param image_embeddings: Image embeddings of shape (N, D)
    :param labels: Binary labels of shape (N, C) where C is number of classes
    :param class_prompt_embeddings: Dict mapping each class to its prompt embeddings [negative, positive]
    :param class_list: List of class names
    :param threshold: Classification threshold (default: 0.5)
    :return: Dictionary with per-class metrics and average metrics
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


def compute_optimal_thresholds(predictions, labels, class_list, thresholds=None):
    """
    Find the optimal classification threshold for each class using Youden's J statistic.

    :param predictions: Model prediction scores, shape (n_samples, n_classes)
    :param labels: Ground truth labels, shape (n_samples, n_classes)
    :param class_list: List of class names
    :param thresholds: List of threshold values to try (default: range from 0.05 to 0.95)
    :return: Dictionary mapping class names to optimal thresholds
    """
    if thresholds is None:
        thresholds = np.arange(0.2, 1.0, 0.1)

    optimal_thresholds = {}

    for i, class_name in enumerate(class_list):
        class_scores = predictions[:, i]
        class_labels = labels[:, i]

        best_threshold = 0.5
        best_j_score = -1

        for threshold in thresholds:
            binary_preds = (class_scores > threshold).astype(int)

            tn, fp, fn, tp = metrics.confusion_matrix(class_labels, binary_preds, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # Youden's J statistic = sensitivity + specificity - 1
            j_score = sensitivity + specificity - 1

            if j_score > best_j_score:
                best_j_score = j_score
                best_threshold = threshold

        optimal_thresholds[class_name] = best_threshold

    return optimal_thresholds


def compute_cosine_similarity(image_embeds, text_embeds):
    sims = F.cosine_similarity(image_embeds, text_embeds)
    return sims.mean().item()


def generate_concept_prompt_embeddings(model, tokenizer, class_list, device, max_length=143):
    """
    Generate medical concept prompt embeddings for zero-shot classification.
    Uses carefully engineered medical prompts for each class.
    :param model: The CLIP model
    :param tokenizer: Corresponding tokenizer
    :param class_list: List of medical concept names to generate embeddings for
    :param device: Device to run the mo on
    :param max_length: Maximum length for tokenization
    :return: Dictionary mapping class names to embeddings for positive and negative prompts
    """
    class_embeddings = {}

    prompt_templates = [
        "Chest X-ray showing {}.",
        "Radiographic evidence of {}.",
        "This image demonstrates findings consistent with {}.",
        "X-ray with features of {}.",
        "Imaging study positive for {}."
    ]

    for class_name in class_list:
        positive_embeddings = []

        with torch.no_grad():
            for template in prompt_templates:
                prompt = template.format(class_name)
                tokens = tokenizer(prompt, return_tensors="pt", padding="max_length",
                                   truncation=True, max_length=max_length).to(device)

                text_features = model.encode_text(tokens)
                if hasattr(model, "text_projection") and model.projection:
                    text_features = model.text_projection(text_features)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                positive_embeddings.append(text_features.cpu().numpy())

        pos_embedding = np.mean(np.concatenate(positive_embeddings, axis=0), axis=0, keepdims=True)

        class_embeddings[class_name] = pos_embedding

    return class_embeddings

