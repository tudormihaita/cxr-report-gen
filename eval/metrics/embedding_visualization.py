import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from typing import Dict, List, Optional


def compute_tsne_embedding_visualization(
        image_embeddings: np.ndarray,
        labels: np.ndarray,
        class_labels: List[str],
        selected_classes: Optional[List[str]] = None,
        perplexity: int = 30,
        n_iter: int = 1000,
        save_path: Optional[str] = None,
        show_individual_plots: bool = False,
        random_seed: int = 42
) -> Dict:
    """Create t-SNE visualization of image embeddings."""
    if selected_classes is not None:
        class_indices = [i for i, label in enumerate(class_labels) if label in selected_classes]
        labels_filtered = labels[:, class_indices]
        class_labels_filtered = [class_labels[i] for i in class_indices]
    else:
        labels_filtered = labels
        class_labels_filtered = class_labels
        class_indices = list(range(len(class_labels)))

    print(f"Computing t-SNE for {len(class_labels_filtered)} findings: {class_labels_filtered}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_seed,
        verbose=1
    )

    tsne_coords = tsne.fit_transform(image_embeddings)

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_labels_filtered)))

    fig_main, main_ax = plt.subplots(1, 1, figsize=(12, 8))

    sample_colors = []
    sample_labels = []
    for i in range(len(tsne_coords)):
        positive_findings = np.where(labels_filtered[i] == 1)[0]
        if len(positive_findings) == 0:
            sample_colors.append('lightgray')
            sample_labels.append('Normal')
        else:
            finding_idx = positive_findings[0]
            sample_colors.append(colors[finding_idx])
            sample_labels.append(class_labels_filtered[finding_idx])

    scatter = main_ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                         c=sample_colors, alpha=0.6, s=20)
    main_ax.set_xticks([])
    main_ax.set_yticks([])
    main_ax.spines['top'].set_visible(False)
    main_ax.spines['right'].set_visible(False)
    main_ax.spines['bottom'].set_visible(False)
    main_ax.spines['left'].set_visible(False)

    legend_elements = []
    unique_labels = list(set(sample_labels))
    for label in unique_labels:
        if label == 'Normal':
            color = 'lightgray'
        else:
            color = colors[class_labels_filtered.index(label)]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=8, label=label))
    main_ax.legend(handles=legend_elements, loc='lower right',
                   frameon=True, fancybox=True, shadow=True,
                   fontsize=11, markerscale=1.2)

    plt.tight_layout()
    if save_path:
        if os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig_main.savefig(f"{save_path}_all_findings.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


    if show_individual_plots:
        num_individual_plots = min(5, len(class_labels_filtered))

        rows = (num_individual_plots + 1) // 2
        fig_individual, axes = plt.subplots(rows, 2, figsize=(16, 6 * rows))
        fig_individual.suptitle("t-SNE Visualization - Individual Findings",
                                fontsize=18)

        # flatten axes array for easier indexing
        if rows == 1:
            axes = [axes] if num_individual_plots == 1 else axes
        else:
            axes = axes.flatten()

        for idx in range(num_individual_plots):
            ax = axes[idx + 1]
            finding_name = class_labels_filtered[idx]

            # color points based on presence/absence of this specific finding
            colors_binary = ['red' if labels_filtered[i, idx] == 1 else 'lightblue'
                            for i in range(len(tsne_coords))]

            scatter = ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                                c=colors_binary, alpha=0.6, s=20)
            ax.set_title(f"{finding_name}")
            ax.set_xlabel("Visual feature space X", fontsize=12)
            ax.set_ylabel("Visual feature space Y", fontsize=12)

            positive_count = np.sum(labels_filtered[:, idx] == 1)
            negative_count = len(labels_filtered) - positive_count
            ax.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=8)],
                    [f'Positive ({positive_count})', f'Negative ({negative_count})'],
                    bbox_to_anchor=(1.05, 1), loc='upper left')

        for idx in range(num_individual_plots, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()

        if save_path:
            if os.path.isdir(save_path):
                os.makedirs(save_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(f"{save_path}_individual_findings.png", dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

    plt.show()

    clustering_metrics = compute_clustering_metrics(tsne_coords, labels_filtered, class_labels_filtered)

    return {
        "clustering_metrics": clustering_metrics,
        "visualization_path": save_path
    }


def compute_clustering_metrics(tsne_coords: np.ndarray, labels: np.ndarray, class_labels: List[str]) -> Dict:
    """
    Compute clustering quality metrics for the t-SNE embedding.
    """
    metrics = {}

    for i, finding_name in enumerate(class_labels):
        finding_labels = labels[:, i]

        if len(np.unique(finding_labels)) > 1:
            sil_score = silhouette_score(tsne_coords, finding_labels)
            metrics[f"silhouette_{finding_name}"] = sil_score

    dominant_findings = []
    for i in range(len(labels)):
        positive_findings = np.where(labels[i] == 1)[0]
        if len(positive_findings) == 0:
            dominant_findings.append(-1)
        else:
            dominant_findings.append(positive_findings[0])

    dominant_findings = np.array(dominant_findings)

    if len(np.unique(dominant_findings)) > 1:
        overall_silhouette = silhouette_score(tsne_coords, dominant_findings)
        metrics["overall_silhouette"] = overall_silhouette

    return metrics
