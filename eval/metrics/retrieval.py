import numpy as np

from sklearn import metrics
from typing import Dict, List, Optional

from constants import CHEXPERT_LABELS
from eval.metrics import sample_balanced_pool, sample_balanced_category_pool

from utils.logger import LoggerManager
log = LoggerManager.get_logger(__name__)


def compute_retrieval_recall_metrics(
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        text_list: List[str],
) -> Dict:
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


def compute_retrieval_precision_metrics(
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        labels: np.ndarray,
        class_indices: Optional[List[int]] = None,
) -> Dict:
    n_samples = len(image_embeddings)
    similarities = metrics.pairwise.cosine_similarity(image_embeddings, text_embeddings)

    if class_indices is not None:
        if max(class_indices) >= labels.shape[1]:
            raise ValueError(
                f"class_indices contains index {max(class_indices)} but labels only have {labels.shape[1]} classes")
        selected_labels = labels[:, class_indices]
    else:
        selected_labels = labels

    precision_results = {1: [], 5: [], 10: []}

    # for each query image
    for i in range(n_samples):
        query_labels = selected_labels[i]
        similarity_scores = similarities[i]

        if not np.any(query_labels == 1):
            continue

        for k in precision_results.keys():
            top_k_indices = np.argsort(similarity_scores)[-k:][::-1]

            # count how many of the top-k texts share at least one positive class
            correct = 0
            for txt_idx in top_k_indices:
                retrieved_labels = selected_labels[txt_idx]

                # check if query and retrieved text share any positive class
                if np.any(np.logical_and(query_labels == 1, retrieved_labels == 1)):
                    correct += 1

            precision = correct / k
            precision_results[k].append(precision)

    results = {
        'precision@1': np.mean(precision_results[1]) * 100 if precision_results[1] else 0.0,
        'precision@5': np.mean(precision_results[5]) * 100 if precision_results[5] else 0.0,
        'precision@10': np.mean(precision_results[10]) * 100 if precision_results[10] else 0.0,
        'valid_queries': len(precision_results[1]),
        'total_samples': n_samples,
        'num_classes': selected_labels.shape[1]
    }

    return results


def compute_retrieval_precision_metrics_per_class(
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: List[str] = CHEXPERT_LABELS,
        class_indices: Optional[List[int]] = None
) -> Dict:
    n_samples, n_classes = labels.shape
    similarities = metrics.pairwise.cosine_similarity(image_embeddings, text_embeddings)

    if class_indices is None:
        if len(class_names) > labels.shape[1]:
            raise ValueError(f"class_names length ({len(class_names)}) exceeds label dimensions ({labels.shape[1]})")
        class_indices = list(range(len(class_names)))
    else:
        if len(class_indices) != len(class_names):
            raise ValueError("class_indices length must match class_names length")
        if max(class_indices) >= labels.shape[1]:
            raise ValueError(
                f"class_indices contains index {max(class_indices)} but labels only have {labels.shape[1]} classes")

    class_results = {}
    avg_precision_results = {1: [], 5: [], 10: []}

    for class_name, class_idx in zip(class_names, class_indices):
        positive_image_indices = np.where(labels[:, class_idx] == 1)[0]

        if len(positive_image_indices) == 0:
            class_results[class_name] = {
                'precision@1': 0.0,
                'precision@5': 0.0,
                'precision@10': 0.0,
                'num_queries': 0
            }
            continue

        precision_results = {1: [], 5: [], 10: []}

        # for each query image with this positive class
        for query_idx in positive_image_indices:
            similarity_scores = similarities[query_idx]

            for k in precision_results.keys():
                # get top-k most similar text indices
                top_k_indices = np.argsort(similarity_scores)[-k:][::-1]

                # count how many retrieved texts have this class as positive
                correct = 0
                for txt_idx in top_k_indices:
                    if labels[txt_idx, class_idx] == 1:
                        correct += 1

                precision = correct / k
                precision_results[k].append(precision)

        class_results[class_name] = {
            'precision@1': np.mean(precision_results[1]) * 100,
            'precision@5': np.mean(precision_results[5]) * 100,
            'precision@10': np.mean(precision_results[10]) * 100,
            'num_queries': len(positive_image_indices)
        }

        # add to overall average (weighted by number of queries)
        for k in avg_precision_results.keys():
            avg_precision_results[k].extend(precision_results[k])

    overall_results = {
        'precision@1_avg': np.mean(avg_precision_results[1]) * 100 if avg_precision_results[1] else 0.0,
        'precision@5_avg': np.mean(avg_precision_results[5]) * 100 if avg_precision_results[5] else 0.0,
        'precision@10_avg': np.mean(avg_precision_results[10]) * 100 if avg_precision_results[10] else 0.0,
        'total_queries': len(avg_precision_results[1])
    }

    results = {**overall_results, 'per_class': class_results}

    return results


def compute_retrieval_precision_metrics_fixed_pool(
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        text_list: List[str],
        labels: np.ndarray,
        class_names: List[str] = CHEXPERT_LABELS,
        class_indices: Optional[List[int]] = None,
        samples_per_class: int = 200,
        random_seed: int = 42
) -> Dict:
    if class_indices is not None:
        if max(class_indices) >= labels.shape[1]:
            raise ValueError(
                f"class_indices contains index {max(class_indices)} but labels only have {labels.shape[1]} classes")
        selected_labels = labels[:, class_indices]
    else:
        selected_labels = labels

    # sample the balanced gallery
    gallery_img_emb, gallery_txt_emb, gallery_texts, gallery_categories, gallery_indices = \
        sample_balanced_category_pool(
            image_embeddings, text_embeddings, text_list, selected_labels, class_names,
            samples_per_class, random_seed
        )

    # compute similarities between all gallery images and texts
    similarities = metrics.pairwise.cosine_similarity(gallery_img_emb, gallery_txt_emb)

    precision_results = {1: [], 5: [], 10: []}

    # for each image in the gallery
    for i, query_category in enumerate(gallery_categories):
        query_index = gallery_indices[i]
        query_labels = selected_labels[query_index]

        similarity_scores = similarities[i]  # similarities to all texts

        # for each k value
        for k in precision_results.keys():
            # get top-k most similar text indices
            top_k_indices = np.argsort(similarity_scores)[-k:][::-1]

            # count how many of the top-k texts belong to the same category
            correct = 0
            for txt_idx in top_k_indices:
                retrieved_index = gallery_indices[txt_idx]
                retrieved_labels = selected_labels[retrieved_index]

                if np.any(np.logical_and(query_labels.astype(bool), retrieved_labels.astype(bool))):
                    correct += 1

            precision = correct / k
            precision_results[k].append(precision)

    # calculate mean precision for each k
    results = {
        'precision@1': np.mean(precision_results[1]) * 100,
        'precision@5': np.mean(precision_results[5]) * 100,
        'precision@10': np.mean(precision_results[10]) * 100,
        'gallery_size': len(gallery_indices),
        'unique_samples': len(set(gallery_indices)),
        'num_classes': len(set(gallery_categories)),
        'classes': class_names,
    }

    return results


def compute_retrieval_recall_metrics_fixed_pool(image_embeddings: np.ndarray,
                                                text_embeddings: np.ndarray,
                                                text_list: List[str],
                                                labels: Optional[np.ndarray] = None,
                                                class_names: Optional[List[str]] = CHEXPERT_LABELS,
                                                pool_size: int = 1000,
                                                use_balanced_sampling: bool = True,
                                                random_seed: int = 42) -> Dict:
    if use_balanced_sampling and labels is not None and class_names is not None:
        pool_img_emb, pool_txt_emb, pool_texts, pool_indices = sample_balanced_pool(
            image_embeddings, text_embeddings, text_list, labels, class_names,
            pool_size, random_seed=random_seed
        )
    else:
        n_samples = len(image_embeddings)
        if n_samples > pool_size:
            np.random.seed(random_seed)
            pool_indices = np.random.choice(n_samples, pool_size, replace=False)
            pool_img_emb = image_embeddings[pool_indices]
            pool_txt_emb = text_embeddings[pool_indices]
            pool_texts = [text_list[i] for i in pool_indices]
        else:
            pool_img_emb = image_embeddings
            pool_txt_emb = text_embeddings
            pool_texts = text_list
            pool_indices = list(range(n_samples))

    pool_idx_to_pos = {idx: pos for pos, idx in enumerate(pool_indices)}

    identical_text_set = []
    idx2label = {}
    identical_indexes = []

    for i, text in enumerate(pool_texts):
        if text not in identical_text_set:
            identical_text_set.append(text)
            identical_indexes.append(i)
            idx2label[i] = len(identical_text_set) - 1
        else:
            idx2label[i] = identical_text_set.index(text)

    unique_text_embeddings = pool_txt_emb[identical_indexes]
    n_unique_texts = len(identical_text_set)

    similarities = metrics.pairwise.cosine_similarity(pool_img_emb, unique_text_embeddings)

    recall_dict = {1: 0, 5: 0, 10: 0}
    mean_rank = 0
    valid_queries = 0

    for idx in range(len(image_embeddings)):
        query_text = text_list[idx]

        if idx in pool_idx_to_pos:
            gallery_pos = pool_idx_to_pos[idx]
            query_label = idx2label[gallery_pos]
        else:
            if query_text in identical_text_set:
                query_label = identical_text_set.index(query_text)
            else:
                continue

        valid_queries += 1
        similarity = similarities[idx]
        similarity_args = similarity.argsort()[::-1]

        rank = np.where(similarity_args == query_label)[0][0] + 1  # 1-indexed rank
        mean_rank += rank

        for k in recall_dict:
            if rank <= k:
                recall_dict[k] += 1

    if valid_queries == 0:
        print("Warning: No valid queries found!")
        return {
            'r_mean': 0.0,
            'recall@1': 0.0,
            'recall@5': 0.0,
            'recall@10': 0.0,
            'valid_queries': 0,
            'pool_size': len(pool_indices),
            'unique_texts': n_unique_texts
        }

    eval_metrics = {
        'r_mean': mean_rank / valid_queries,
        'recall@1': (recall_dict[1] / valid_queries) * 100,
        'recall@5': (recall_dict[5] / valid_queries) * 100,
        'recall@10': (recall_dict[10] / valid_queries) * 100,
        'valid_queries': valid_queries,
        'pool_size': len(pool_indices),
        'unique_texts': n_unique_texts
    }

    return eval_metrics