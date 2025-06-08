import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from scipy.special import softmax
from typing import Dict, List, Union, Optional, Tuple

from data import load_prompts_from_json

from utils.logger import LoggerManager
log = LoggerManager.get_logger(__name__)


def compute_supervised_classification_metrics(
        predictions: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        class_list: List[str],
        threshold: float = 0.5,
) -> Dict:
    """
    Compute classification metrics from model predictions (probabilities).
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    results = {}

    for i, class_name in enumerate(class_list):
        class_labels = labels[:, i]
        class_probs = predictions[:, i]

        binary_predictions = (class_probs > threshold).astype(int)

        if len(np.unique(class_labels)) > 1:
            auroc = metrics.roc_auc_score(class_labels, class_probs)
        else:
            auroc = 0.5

        accuracy = metrics.accuracy_score(class_labels, binary_predictions)
        precision = metrics.precision_score(class_labels, binary_predictions, zero_division=0)
        recall = metrics.recall_score(class_labels, binary_predictions, zero_division=0)
        f1 = metrics.f1_score(class_labels, binary_predictions, zero_division=0)
        balanced_accuracy = metrics.balanced_accuracy_score(class_labels, binary_predictions)

        tn, fp, fn, tp = metrics.confusion_matrix(
            class_labels, binary_predictions, labels=[0, 1]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results[class_name] = {
            "auroc": auroc,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "threshold": threshold,
            "positive_samples": int(np.sum(class_labels)),
            "negative_samples": int(len(class_labels) - np.sum(class_labels))
        }

    avg_metrics = {
        'auroc_avg': np.mean([results[c]['auroc'] for c in class_list]),
        'accuracy_avg': np.mean([results[c]['accuracy'] for c in class_list]),
        'balanced_accuracy_avg': np.mean([results[c]['balanced_accuracy'] for c in class_list]),
        'precision_avg': np.mean([results[c]['precision'] for c in class_list]),
        'recall_avg': np.mean([results[c]['recall'] for c in class_list]),
        'specificity_avg': np.mean([results[c]['specificity'] for c in class_list]),
        'f1_avg': np.mean([results[c]['f1'] for c in class_list])
    }
    results['average'] = avg_metrics

    return results


def compute_supervised_classification_with_optimal_thresholds(
        predictions: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        class_list: List[str],
) -> Dict:
    """
    Compute supervised classification metrics with optimal thresholds per class.
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    results = {}
    optimal_thresholds = find_optimal_thresholds_per_class(
        predictions, labels, class_list
    )

    for i, class_name in enumerate(class_list):
        class_labels = labels[:, i]
        class_probs = predictions[:, i]
        threshold = optimal_thresholds[class_name]

        binary_predictions = (class_probs > threshold).astype(int)

        if len(np.unique(class_labels)) > 1:
            auroc = metrics.roc_auc_score(class_labels, class_probs)
        else:
            auroc = 0.5

        accuracy = metrics.accuracy_score(class_labels, binary_predictions)
        precision = metrics.precision_score(class_labels, binary_predictions, zero_division=0)
        recall = metrics.recall_score(class_labels, binary_predictions, zero_division=0)
        f1 = metrics.f1_score(class_labels, binary_predictions, zero_division=0)
        balanced_accuracy = metrics.balanced_accuracy_score(class_labels, binary_predictions)

        tn, fp, fn, tp = metrics.confusion_matrix(
            class_labels, binary_predictions, labels=[0, 1]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results[class_name] = {
            "auroc": auroc,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "optimal_threshold": threshold,
            "positive_samples": int(np.sum(class_labels)),
            "negative_samples": int(len(class_labels) - np.sum(class_labels))
        }

    avg_metrics = {
        'auroc_avg': np.mean([results[c]['auroc'] for c in class_list]),
        'accuracy_avg': np.mean([results[c]['accuracy'] for c in class_list]),
        'balanced_accuracy_avg': np.mean([results[c]['balanced_accuracy'] for c in class_list]),
        'precision_avg': np.mean([results[c]['precision'] for c in class_list]),
        'recall_avg': np.mean([results[c]['recall'] for c in class_list]),
        'specificity_avg': np.mean([results[c]['specificity'] for c in class_list]),
        'f1_avg': np.mean([results[c]['f1'] for c in class_list])
    }
    results['average'] = avg_metrics

    return results


def compute_zero_shot_classification_metrics(
        model,
        tokenizer,
        image_embeddings: np.ndarray,
        labels: np.ndarray,
        prompt_file_path: str,
        class_list: List[str],
        device: torch.device,
        threshold: float = 0.5,
        max_length: int = 77,
        random_seed: int = 42
) -> Dict:
    positive_probs = compute_zero_shot_classification_scores(
        model, tokenizer, image_embeddings, prompt_file_path,
        class_list, device, max_length, random_seed
    )

    results = {}
    for i, class_name in enumerate(class_list):
        class_labels = labels[:, i]
        class_probs = positive_probs[:, i]

        # convert probabilities to binary predictions using threshold
        predictions = (class_probs > threshold).astype(int)

        # compute AUROC (only if both classes present)
        if len(np.unique(class_labels)) > 1:
            auroc = metrics.roc_auc_score(class_labels, class_probs)
        else:
            auroc = 0.5

        accuracy = metrics.accuracy_score(class_labels, predictions)
        precision = metrics.precision_score(class_labels, predictions, zero_division=0)
        recall = metrics.recall_score(class_labels, predictions, zero_division=0)
        f1 = metrics.f1_score(class_labels, predictions, zero_division=0)

        tn, fp, fn, tp = metrics.confusion_matrix(
            class_labels, predictions, labels=[0, 1]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results[class_name] = {
            "auroc": auroc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "threshold": threshold,
            "positive_samples": int(np.sum(class_labels)),
            "negative_samples": int(len(class_labels) - np.sum(class_labels))
        }

    # compute average metrics across all classes
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


def compute_zero_shot_classification_with_optimal_thresholds(
        model,
        tokenizer,
        image_embeddings: np.ndarray,
        labels: np.ndarray,
        prompt_file_path: str,
        class_list: List[str],
        device: torch.device,
        max_length: int = 128,
        random_seed: int = 42
) -> Dict:
    # get positive class probabilities for all samples and classes
    positive_probs = compute_zero_shot_classification_scores(
        model, tokenizer, image_embeddings, prompt_file_path,
        class_list, device, max_length, random_seed
    )

    # find optimal thresholds for each class
    optimal_thresholds = find_optimal_thresholds_per_class(
        positive_probs, labels, class_list
    )

    results = {}

    # compute metrics for each class using optimal thresholds
    for i, class_name in enumerate(class_list):
        class_labels = labels[:, i]
        class_probs = positive_probs[:, i]
        threshold = optimal_thresholds[class_name]

        # convert probabilities to binary predictions using optimal threshold
        predictions = (class_probs > threshold).astype(int)

        # compute AUROC
        if len(np.unique(class_labels)) > 1:
            auroc = metrics.roc_auc_score(class_labels, class_probs)
        else:
            auroc = 0.5

        # compute other metrics
        accuracy = metrics.accuracy_score(class_labels, predictions)
        precision = metrics.precision_score(class_labels, predictions, zero_division=0)
        recall = metrics.recall_score(class_labels, predictions, zero_division=0)
        f1 = metrics.f1_score(class_labels, predictions, zero_division=0)
        balanced_accuracy = metrics.balanced_accuracy_score(class_labels, predictions)

        # compute confusion matrix
        tn, fp, fn, tp = metrics.confusion_matrix(
            class_labels, predictions, labels=[0, 1]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results[class_name] = {
            "auroc": auroc,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "optimal_threshold": threshold,
            "positive_samples": int(np.sum(class_labels)),
            "negative_samples": int(len(class_labels) - np.sum(class_labels))
        }

    # compute average metrics
    avg_metrics = {
        'auroc_avg': np.mean([results[c]['auroc'] for c in class_list]),
        'accuracy_avg': np.mean([results[c]['accuracy'] for c in class_list]),
        'balanced_accuracy_avg': np.mean([results[c]['balanced_accuracy'] for c in class_list]),
        'precision_avg': np.mean([results[c]['precision'] for c in class_list]),
        'recall_avg': np.mean([results[c]['recall'] for c in class_list]),
        'specificity_avg': np.mean([results[c]['specificity'] for c in class_list]),
        'f1_avg': np.mean([results[c]['f1'] for c in class_list])
    }
    results['average'] = avg_metrics

    log.info(f"Computed zero-shot classification with optimal thresholds")
    log.info(f"Average AUROC: {avg_metrics['auroc_avg']:.3f}")
    log.info(f"Average Accuracy: {avg_metrics['accuracy_avg']:.3f}")

    return results


def find_optimal_thresholds_per_class(
        positive_probabilities: np.ndarray,
        labels: np.ndarray,
        class_list: List[str],
) -> Dict[str, float]:
    """
    Find optimal decision thresholds for each class using Youden's J statistic.
    """
    optimal_thresholds = {}

    for i, class_name in enumerate(class_list):
        class_labels = labels[:, i]
        class_probs = positive_probabilities[:, i]

        if len(np.unique(class_labels)) <= 1:
            optimal_thresholds[class_name] = 0.5
            continue

        # find optimal threshold using ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(class_labels, class_probs)

        # Youden's J statistic: maximize (sensitivity + specificity - 1)
        youden_scores = tpr - fpr
        optimal_idx = np.argmax(youden_scores)

        optimal_thresholds[class_name] = thresholds[optimal_idx]

    return optimal_thresholds


def compute_zero_shot_classification_scores(
        model,
        tokenizer,
        image_embeddings: np.ndarray,
        prompt_file_path: str,
        class_list: List[str],
        device: torch.device,
        max_length: int = 128,
        random_seed: int = 42
) -> np.ndarray:
    random.seed(random_seed)
    np.random.seed(random_seed)

    n_samples = image_embeddings.shape[0]
    n_classes = len(class_list)

    # store positive class probabilities for each sample and class
    positive_probabilities = np.zeros((n_samples, n_classes))

    model.eval()
    with torch.no_grad():
        image_embeds_tensor = torch.from_numpy(image_embeddings).to(device)

        for i, class_name in enumerate(class_list):
            pos_prompts, neg_prompts = load_prompts_from_json(prompt_file_path, class_name)

            if not pos_prompts or not neg_prompts:
                log.warning(f"No prompts found for class '{class_name}'")
                continue

            # for each image, randomly sample one positive and one negative prompt
            for sample_idx in range(n_samples):
                pos_prompt = random.choice(pos_prompts)
                neg_prompt = random.choice(neg_prompts)

                pos_tokens = tokenizer(
                    pos_prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                ).to(device)

                neg_tokens = tokenizer(
                    neg_prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                ).to(device)

                pos_text_features = model.encode_text(pos_tokens)
                neg_text_features = model.encode_text(neg_tokens)

                if hasattr(model, "text_projection") and model.projection:
                    pos_text_features = model.text_projection(pos_text_features)
                    neg_text_features = model.text_projection(neg_text_features)

                pos_text_features = pos_text_features / pos_text_features.norm(dim=1, keepdim=True)
                neg_text_features = neg_text_features / neg_text_features.norm(dim=1, keepdim=True)

                # compute similarities with current image
                pos_similarity = torch.cosine_similarity(
                    image_embeds_tensor[sample_idx:sample_idx + 1],
                    pos_text_features
                ).item()
                neg_similarity = torch.cosine_similarity(
                    image_embeds_tensor[sample_idx:sample_idx + 1],
                    neg_text_features
                ).item()

                # apply softmax to get probability of positive class
                similarities = np.array([neg_similarity, pos_similarity])
                probabilities = softmax(similarities)
                positive_probabilities[sample_idx, i] = probabilities[1]
                del pos_tokens, neg_tokens, pos_text_features, neg_text_features

        del pos_prompts, neg_prompts
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return positive_probabilities


def plot_roc_curves(
        predictions: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        class_list: List[str],
        class_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 12),
        ncols: int = 3
) -> plt.Figure:
    """
    Plot ROC curves for specified classes in a grid layout.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if class_indices is None:
        # assume class_list corresponds to first len(class_list) columns
        if len(class_list) > predictions.shape[1]:
            raise ValueError(
                f"class_list length ({len(class_list)}) exceeds prediction dimensions ({predictions.shape[1]})")
        class_indices = list(range(len(class_list)))
    else:
        if len(class_indices) != len(class_list):
            raise ValueError("class_indices length must match class_list length")
        if max(class_indices) >= predictions.shape[1]:
            raise ValueError(
                f"class_indices contains index {max(class_indices)} but predictions only have {predictions.shape[1]} classes")

    n_classes = len(class_list)
    nrows = (n_classes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)
    plt.style.use('default')

    for i, (class_name, class_idx) in enumerate(zip(class_list, class_indices)):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        class_labels = labels[:, class_idx]
        class_probs = predictions[:, class_idx]

        # check if we have both classes
        if len(np.unique(class_labels)) <= 1:
            ax.text(0.5, 0.5, f'Only one class\npresent',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{class_name}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            continue

        fpr, tpr, thresholds = metrics.roc_curve(class_labels, class_probs)
        roc_auc = metrics.auc(fpr, tpr)

        # find optimal threshold using Youden's J statistic
        youden_scores = tpr - fpr
        optimal_idx = np.argmax(youden_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        # plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')

        # plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)

        # mark optimal point
        ax.plot(optimal_fpr, optimal_tpr, marker='o', color='red',
                markersize=8, label=f'Optimal (t={optimal_threshold:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{class_name}')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # hide empty subplots
    for i in range(n_classes, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"ROC curves saved to {save_path}")

    return fig
