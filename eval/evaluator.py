import json
import torch
import numpy as np
from tqdm import tqdm

from data import load_tokenizer
from cxrclip import build_model
from configs import load_config_from_file

from constants import CHEXPERT_LABELS
from eval.metrics.generation import compute_text_generation_metrics
from eval.metrics.embedding_visualization import compute_tsne_embedding_visualization
from eval.metrics.retrieval import compute_retrieval_recall_metrics, compute_retrieval_precision_metrics, \
    compute_retrieval_precision_metrics_fixed_pool, compute_retrieval_recall_metrics_fixed_pool, \
    compute_retrieval_precision_metrics_per_class
from eval.metrics.classification import compute_zero_shot_classification_with_optimal_thresholds, \
    compute_supervised_classification_with_optimal_thresholds, plot_roc_curves


class Evaluator:
    def __init__(self, model_config_path, loss_config_path, tokenizer_config_path, prompts_path, ckpt_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prompts_path = prompts_path

        tokenizer_config = load_config_from_file(tokenizer_config_path)
        self.tokenizer = load_tokenizer(**tokenizer_config)

        model_config = load_config_from_file(model_config_path)
        loss_config = load_config_from_file(loss_config_path)
        self.model = build_model(model_config, loss_config, self.tokenizer)

        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        self.model.to(self.device)
        self.model.eval()


    @torch.no_grad()
    def _collect_embeddings(self, dataloader):
        all_texts = []
        all_labels = []
        all_text_embeddings = []
        all_image_embeddings = []

        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images = batch['images']
            texts = batch['texts']
            text_tokens = batch['text_tokens']
            labels = batch['labels']

            img_emb = self.model.encode_image(images.to(self.device))
            img_emb = self.model.image_projection(img_emb) if self.model.projection else img_emb
            img_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)
            img_emb = img_emb.detach().cpu().numpy()

            text_emb = self.model.encode_text(text_tokens.to(self.device))
            text_emb = self.model.text_projection(text_emb) if self.model.projection else text_emb
            text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
            text_emb = text_emb.detach().cpu().numpy()

            all_texts.extend(texts)
            all_text_embeddings.append(text_emb)
            all_image_embeddings.append(img_emb)
            all_labels.extend(labels.detach().cpu().numpy())

        all_image_embeddings = np.concatenate(all_image_embeddings)
        all_text_embeddings = np.concatenate(all_text_embeddings)
        all_labels = np.array(all_labels)

        return {
            "texts": all_texts,
            "text_embeddings": all_text_embeddings,
            "image_embeddings": all_image_embeddings,
            "labels": all_labels
        }

    @torch.no_grad()
    def _collect_predictions(self, dataloader):
        all_predictions, all_labels = [], []
        for batch in tqdm(dataloader, desc="Extracting predictions"):
            outputs = self.model(batch, self.device)
            predictions = torch.sigmoid(outputs["cls_pred"]).detach().cpu().numpy()
            labels = outputs["target_class"].detach().cpu().numpy()

            all_predictions.append(predictions)
            all_labels.append(labels)

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return {
            "predictions": all_predictions,
            "labels": all_labels
        }

    @torch.no_grad()
    def _collect_texts(self, dataloader):
        references, hypotheses = [], []

        for batch in tqdm(dataloader, desc="Extracting texts"):
            images = batch['images'].to(self.device)
            reports = batch['texts']
            findings = batch['labels']

            preds = self.model.generate(
                images=images,
                findings=findings,
                temperature=1.0,
                repetition_penalty=1.4,
                device=self.device
            )

            references.extend([[r] for r in reports])
            hypotheses.extend(preds)

        return {
            "references": references,
            "hypotheses": hypotheses
        }

    @torch.no_grad()
    def evaluate(self, dataloader, metrics):
        assert metrics in ["retrieval", "retrieval_fixed_pool", "classification_zero_shot", "classification_supervised", "generation"], \
            f"Unsupported performance metrics evaluation: {metrics}"

        results = {}

        if metrics == "retrieval":
            processed = self._collect_embeddings(dataloader)
            texts = processed["texts"]
            labels = processed["labels"]
            text_embeddings = processed["text_embeddings"]
            image_embeddings = processed["image_embeddings"]

            recall_metrics = compute_retrieval_recall_metrics(image_embeddings, text_embeddings, texts)
            precision_metrics = compute_retrieval_precision_metrics(image_embeddings, text_embeddings, labels)
            per_class_precision_metrics = compute_retrieval_precision_metrics_per_class(image_embeddings, text_embeddings, labels, CHEXPERT_LABELS)
            tsne = compute_tsne_embedding_visualization(image_embeddings, labels, CHEXPERT_LABELS, save_path='./output/plots/tsne_visualization.png')

            results.update(recall_metrics)
            results.update(precision_metrics)
            results.update(per_class_precision_metrics)
            results.update(tsne)
        elif metrics == "retrieval_fixed_pool":
            processed = self._collect_embeddings(dataloader)
            texts = processed["texts"]
            labels = processed["labels"]
            text_embeddings = processed["text_embeddings"]
            image_embeddings = processed["image_embeddings"]

            recall_metrics = compute_retrieval_recall_metrics_fixed_pool(image_embeddings, text_embeddings, texts, labels, CHEXPERT_LABELS)
            precision_metrics = compute_retrieval_precision_metrics_fixed_pool(image_embeddings, text_embeddings, texts, labels, CHEXPERT_LABELS)
            results.update(recall_metrics)
            results.update(precision_metrics)
        elif metrics == "classification_zero_shot":
            processed = self._collect_embeddings(dataloader)
            labels = processed["labels"]
            image_embeddings = processed["image_embeddings"]

            classification_metrics = compute_zero_shot_classification_with_optimal_thresholds(self.model, self.tokenizer, image_embeddings, labels, self.prompts_path, CHEXPERT_LABELS, self.device)
            results.update(classification_metrics)
        elif metrics == "classification_supervised":
            processed = self._collect_predictions(dataloader)
            predictions = processed["predictions"]
            class_labels = processed["labels"]
            classification_metrics = compute_supervised_classification_with_optimal_thresholds(predictions, class_labels, CHEXPERT_LABELS)
            plot_roc_curves(predictions, class_labels, CHEXPERT_LABELS, save_path='./output/plots/')
            results.update(classification_metrics)
        elif metrics == "generation":
            processed = self._collect_texts(dataloader)
            references = processed["references"]
            predictions = processed["hypotheses"]
            generation_metrics = compute_text_generation_metrics(predictions, references, self.device)
            results.update(generation_metrics)

        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return float(obj.item())
            elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        eval_metrics = convert_to_serializable(results)
        print(f"Evaluation metrics: {json.dumps(eval_metrics, indent=2)}")

        return eval_metrics
