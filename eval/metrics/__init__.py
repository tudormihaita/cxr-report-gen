import random
import numpy as np

from typing import List, Tuple
from collections import defaultdict

from utils.logger import LoggerManager
log = LoggerManager.get_logger(__name__)


def sample_balanced_pool(image_embeddings: np.ndarray,
                            text_embeddings: np.ndarray,
                            text_list: List[str],
                            labels: np.ndarray,
                            class_names: List[str],
                            pool_size: int = 1000,
                            min_samples_per_class: int = 10,
                            random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    random.seed(random_seed)
    np.random.seed(random_seed)
    n_samples = len(image_embeddings)

    # use all samples if pool size is larger than available samples
    if n_samples <= pool_size:
        indices = list(range(n_samples))
        return (image_embeddings, text_embeddings, text_list, indices)

    # group indices by positive classes
    class_indices = defaultdict(list)
    no_finding_indices = []

    for idx in range(n_samples):
        positive_classes = np.where(labels[idx] == 1)[0]
        if len(positive_classes) == 0:
            no_finding_indices.append(idx)
        else:
            for class_idx in positive_classes:
                class_indices[class_idx].append(idx)

    available_classes = list(class_indices.keys())
    n_available_classes = len(available_classes)

    if n_available_classes == 0:
        selected_indices = random.sample(range(n_samples), pool_size)
    else:
        target_per_class = max(min_samples_per_class, pool_size // (n_available_classes + 1))  # +1 for no finding

        selected_indices = []

        for class_idx in available_classes:
            available_samples = len(class_indices[class_idx])
            samples_to_take = min(target_per_class, available_samples)

            if available_samples <= samples_to_take:
                selected_indices.extend(class_indices[class_idx])
            else:
                selected_indices.extend(random.sample(class_indices[class_idx], samples_to_take))

        if no_finding_indices:
            no_finding_to_take = min(target_per_class, len(no_finding_indices))
            if len(no_finding_indices) <= no_finding_to_take:
                selected_indices.extend(no_finding_indices)
            else:
                selected_indices.extend(random.sample(no_finding_indices, no_finding_to_take))

        selected_indices = list(set(selected_indices))

        if len(selected_indices) < pool_size:
            remaining_indices = [i for i in range(n_samples) if i not in selected_indices]
            additional_needed = pool_size - len(selected_indices)

            if remaining_indices:
                additional_samples = random.sample(
                    remaining_indices,
                    min(additional_needed, len(remaining_indices))
                )
                selected_indices.extend(additional_samples)

        elif len(selected_indices) > pool_size:
            selected_indices = random.sample(selected_indices, pool_size)

    selected_indices.sort()

    sampled_image_embeddings = image_embeddings[selected_indices]
    sampled_text_embeddings = text_embeddings[selected_indices]
    sampled_texts = [text_list[i] for i in selected_indices]

    print(f"Sampled {len(selected_indices)} samples for pool from {n_samples} total samples")

    sampled_labels = labels[selected_indices]
    class_counts = np.sum(sampled_labels, axis=0)
    print("Class distribution in pool:")
    for i, (label_name, count) in enumerate(zip(class_names, class_counts)):
        print(f"  {label_name}: {count}")
    no_finding_count = np.sum(np.sum(sampled_labels, axis=1) == 0)
    print(f"  No Finding: {no_finding_count}")

    return sampled_image_embeddings, sampled_text_embeddings, sampled_texts, selected_indices


def sample_balanced_category_pool(
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        text_list: List[str],
        labels: np.ndarray,
        class_names: List[str],
        samples_per_class: int = 200,
        random_seed: int = 42
):
    np.random.seed(random_seed)

    # build mapping from each class to all samples that have that class
    class_to_samples = defaultdict(list)

    for i in range(len(labels)):
        # find all positive classes for this sample
        positive_classes = []
        for j, val in enumerate(labels[i]):
            if val == 1:
                positive_classes.append(class_names[j])

        # if no positive findings, assign to "No Finding"
        if not positive_classes:
            positive_classes = ["No Finding"]

        # add this sample to all its positive classes
        for class_name in positive_classes:
            class_to_samples[class_name].append(i)

    # sample from each class
    pool_img_emb = []
    pool_txt_emb = []
    pool_texts = []
    pool_categories = []  # which class this sample was selected for
    pool_indices = []

    print("Class sampling statistics:")
    for class_name, sample_indices in class_to_samples.items():
        n_available = len(sample_indices)
        n_samples = min(samples_per_class, n_available)

        if n_samples > 0:
            # sample without replacement from this class
            selected_indices = np.random.choice(sample_indices, n_samples, replace=False)

            for idx in selected_indices:
                pool_img_emb.append(image_embeddings[idx])
                pool_txt_emb.append(text_embeddings[idx])
                pool_texts.append(text_list[idx])
                pool_categories.append(class_name)  # remember which class this was sampled for
                pool_indices.append(idx)

        print(f" {class_name}: sampled {n_samples} from {n_available} available")

    print(f"\nTotal pool size: {len(pool_indices)}")
    print(f"Unique samples: {len(set(pool_indices))}")

    return (
        np.array(pool_img_emb),
        np.array(pool_txt_emb),
        pool_texts,
        pool_categories,
        pool_indices
    )
