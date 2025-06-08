import random
import torch
import torch.nn as nn
import numpy as np

from enum import Enum
from scipy.special import softmax
from typing import List, Optional, Dict

from constants import CHEXPERT_LABELS
from data import load_prompts_from_json


class PromptStrategy(Enum):
    GROUND_TRUTH = "ground_truth"
    SUPERVISED = "supervised"
    ZERO_SHOT = "zero_shot"


class PromptConstructor(nn.Module):
    def __init__(
            self,
            prompt_strategy: PromptStrategy,
            pretrained_model: nn.Module,
            prompt_file_path: str,
            use_diverse_templates: bool = False,
            optimal_thresholds: Optional[Dict[str, float]] = None,
            max_length: int = 77,
            random_seed: int = 42
    ):
        super().__init__()
        self.prompt_strategy = prompt_strategy

        self.use_diverse_templates = use_diverse_templates
        self.optimal_thresholds = optimal_thresholds or {}
        self.prompt_file_path = prompt_file_path
        self.max_length = max_length
        self.random_seed = random_seed

        self.default_threshold = 0.5

        if self.prompt_strategy == PromptStrategy.GROUND_TRUTH:
            self.image_encoder = pretrained_model.image_encoder
        elif self.prompt_strategy == PromptStrategy.SUPERVISED:
            self.image_encoder = pretrained_model.image_encoder
            self.classifier = pretrained_model.classifier
        elif self.prompt_strategy == PromptStrategy.ZERO_SHOT:
            self.image_encoder = pretrained_model.image_encoder
            self.clip_encoder = self.pretrained_model
        else:
            raise ValueError(f"Unsupported prompt strategy: {self.prompt_strategy}")

        self.finding_templates = [
            "Chest X-ray showing {findings}",

            "The chest radiograph shows {findings}",
            "Radiographic findings include {findings}",
            "Notable findings: {findings}",
            "Chest X-ray demonstrates {findings}",

            "Based on imaging analysis: {findings}",
            "Key radiological observations: {findings}",
            "Diagnostic findings present: {findings}",

            "Findings: {findings}",
            "Shows {findings}",
        ]

        self.normal_templates = [
            "No abnormal findings",
            "Normal chest radiograph",
            "No acute findings",
            "Clear lungs, normal heart size",
            "No radiographic abnormalities",

            "The chest radiograph appears normal",
            "No significant abnormalities detected",
            "Normal cardiopulmonary appearance",
            "No acute cardiopulmonary process",
            "Unremarkable chest X-ray",
        ]

    @staticmethod
    def get_findings_from_labels(labels: torch.Tensor):
        findings_list = []
        for label_tensor in labels:
            if label_tensor[8] == 1:  # label position 8 is 'No Finding' for CheXpert
                findings_list.append([])
            else:
                findings = [CHEXPERT_LABELS[i] for i in torch.where(label_tensor == 1)[0].detach().cpu().numpy()]
                findings_list.append(findings)

        return findings_list

    @staticmethod
    def _format_findings_text(findings: List[str]):
        if not findings:
            return ""

        if len(findings) == 1:
            return findings[0]
        elif len(findings) == 2:
            return f"{findings[0]} and {findings[1]}"
        else:
            return ", ".join(findings[:-1]) + f" and {findings[-1]}"

    def construct_prompts(self, labels: torch.Tensor):
        findings_list = self.get_findings_from_labels(labels)

        prompts = []
        for findings in findings_list:
            if findings:
                findings_text = self.format_findings_text(findings)
                if self.use_diverse_templates:
                    template = random.choice(self.finding_templates)
                else:
                    template = self.finding_templates[0]
                prompt = template.format(findings=findings_text)
            else:
                if self.use_diverse_templates:
                    prompt = random.choice(self.normal_templates)
                else:
                    prompt = self.normal_templates[0]
            prompts.append(prompt)

        return prompts

    def _predict_labels_supervised(self, images):
        with torch.no_grad():
            image_features = self.image_encoder(images)
            cls_pred = self.classifier(image_features)
            predictions = torch.sigmoid(cls_pred).detach().cpu().numpy()

            binary_labels = np.zeros_like(predictions)
            for i, class_name in enumerate(CHEXPERT_LABELS):
                threshold = self.optimal_thresholds.get(class_name, self.default_threshold)
                binary_labels[:, i] = (predictions[:, i] > threshold).astype(int)

            return torch.from_numpy(binary_labels).long()

    def _predict_labels_zero_shot(
            self,
            images,
            tokenizer,
            device: torch.device,
    ):
        if self.prompt_file_path is None:
            raise ValueError("prompt_file_path is required for zero-shot classification")

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        image_embeddings = self.clip_encoder.encode_image(images)
        image_embeddings = self.clip_encoder.image_projection(image_embeddings) if hasattr(self.clip_encoder, "image_projection") else image_embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        image_embeddings = image_embeddings.detach().cpu().numpy()

        batch_size = image_embeddings.shape[0]
        num_classes = len(CHEXPERT_LABELS)

        positive_probabilities = np.zeros((batch_size, num_classes))

        self.clip_encoder.eval()
        with torch.no_grad():
            for i, class_name in enumerate(CHEXPERT_LABELS):
                pos_prompts, neg_prompts = load_prompts_from_json(
                    self.prompt_file_path, class_name
                )

                if not pos_prompts or not neg_prompts:
                    positive_probabilities[:, i] = self.default_threshold
                    continue

                for sample_idx in range(batch_size):
                    pos_prompt = random.choice(pos_prompts)
                    neg_prompt = random.choice(neg_prompts)

                    pos_tokens = tokenizer(
                        pos_prompt,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length
                    ).to(device)

                    neg_tokens = tokenizer(
                        neg_prompt,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length
                    ).to(device)

                    pos_text_features = self.clip_encoder.encode_text(pos_tokens)
                    neg_text_features = self.clip_encoder.encode_text(neg_tokens)

                    if hasattr(self.clip_encoder, "text_projection") and self.clip_encoder.text_projection is not None:
                        pos_text_features = self.clip_encoder.text_projection(pos_text_features)
                        neg_text_features = self.clip_encoder.text_projection(neg_text_features)

                    pos_text_features = pos_text_features / pos_text_features.norm(dim=1, keepdim=True)
                    neg_text_features = neg_text_features / neg_text_features.norm(dim=1, keepdim=True)

                    current_image = image_embeddings[sample_idx:sample_idx + 1]
                    pos_similarity = torch.cosine_similarity(current_image, pos_text_features).item()
                    neg_similarity = torch.cosine_similarity(current_image, neg_text_features).item()

                    similarities = np.array([neg_similarity, pos_similarity])
                    probabilities = softmax(similarities)
                    positive_probabilities[sample_idx, i] = probabilities[1]

        binary_labels = np.zeros_like(positive_probabilities)
        for i, class_name in enumerate(CHEXPERT_LABELS):
            threshold = self.optimal_thresholds.get(class_name, self.default_threshold)
            binary_labels[:, i] = (positive_probabilities[:, i] > threshold).astype(int)

        return torch.from_numpy(binary_labels).long()

    def predict_findings(self, images, tokenizer=None, labels=None, device=None):
        if self.prompt_strategy == PromptStrategy.GROUND_TRUTH:
            return labels
        elif self.prompt_strategy == PromptStrategy.SUPERVISED:
            return self._predict_labels_supervised(images)
        elif self.prompt_strategy == PromptStrategy.ZERO_SHOT:
            if tokenizer is None:
                raise ValueError("tokenizer must be provided for zero-shot classification")
            device = device if device is not None else images.device
            return self._predict_labels_zero_shot(images, tokenizer, device)
        else:
            raise ValueError(f"Unsupported prompt strategy: {self.prompt_strategy}")

