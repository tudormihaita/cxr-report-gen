import time
import json
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from classifier.loss import FocalBCELoss


class ChexpertClassifierTrainer:
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            test_loader=None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            freeze_vision_encoder=False,
            learning_rate=1e-3,
            weight_decay=0.01,
            warmup_steps=1000,
            max_steps=10000,
            log_interval=100,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            threshold=0.5,
            weighted_loss=False
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.threshold = threshold

        self.logger = self.__setup_logger(__name__)

        self.pos_weights = self.__compute_pos_weights() if weighted_loss else None
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        # self.loss_fn = FocalBCELoss()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_steps, eta_min=0)

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epoch = 0

        self.model = self.model.to(self.device)

        if freeze_vision_encoder:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")

    @staticmethod
    def __setup_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()
        logger.propagate = False

        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def __compute_pos_weights(self):
        pos_count = torch.zeros(self.model.num_classes, dtype=torch.float)
        neg_count = torch.zeros(self.model.num_classes, dtype=torch.float)

        for batch in tqdm(self.train_loader, desc="Computing class counts"):
            labels = batch['labels']
            labels = labels.to(torch.float)

            labels = torch.where(labels == -1, torch.tensor(1.0), labels)

            pos_count += labels.sum(dim=0)
            neg_count += (1.0 - labels).sum(dim=0)

        pos_weights = neg_count / (pos_count + 1e-6)
        pos_weights = pos_weights.to(self.device)
        return pos_weights

    def train(self, epochs=None):
        self.logger.info("Start training")
        self.model.train()

        if epochs is None:
            steps_per_epoch = len(self.train_loader)
            epochs = (self.max_steps + steps_per_epoch - 1) // steps_per_epoch

        for epoch in range(epochs):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}/{epochs}")

            epoch_loss = 0.0
            epoch_start_time = time.time()

            for index, batch in enumerate(tqdm(self.train_loader)):
                if self.global_step >= self.max_steps:
                    self.logger.info(f"Reached maximum steps {self.max_steps}. Stopping training.")
                    break

                loss = self.train_step(batch)
                epoch_loss += loss

                if (self.global_step + 1) % self.log_interval == 0:
                    self.logger.info(f"Step {self.global_step + 1}: Loss = {loss:.4f}")

                self.global_step += 1

            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. Average loss: {epoch_loss / len(self.train_loader):.4f}")

            val_metrics = self.evaluate()
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']

        self.logger.info("Training finished")

    def train_step(self, batch):
        inputs, labels = batch['image'].to(self.device), batch['labels'].to(self.device)

        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        outputs = self.model(inputs)

        logits = outputs['logits']
        loss = self.loss_fn(logits, labels)
        original_loss = loss

        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()

        return original_loss

    @torch.no_grad()
    def evaluate(self, test=False):
        self.model.eval()
        dataloader = self.test_loader if test else self.val_loader
        phase = 'test' if test else 'val'

        total_loss = 0.0
        all_labels_pred, all_labels_gt = [], []

        self.logger.info(f"Running evaluation on {phase} set")

        for batch in tqdm(dataloader, desc=f"{phase.capitalize()} Evaluation"):
            embeds, labels = batch['image'].to(self.device), batch['labels'].to(self.device)

            outputs = self.model(embeds)
            logits = outputs['logits']
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item()

            labels_pred = torch.sigmoid(logits).cpu().numpy()
            all_labels_pred.append(labels_pred)

            labels_gt = labels.clone().detach().cpu().numpy()
            labels_gt[labels_gt == -1] = 0.0
            mask = (labels_gt != -1)
            all_labels_gt.append((labels_gt, mask))

        avg_loss = total_loss / len(dataloader)
        all_labels_pred = np.concatenate(all_labels_pred, axis=0)

        metrics = {
            f"{phase}_loss": avg_loss
        }

        uncertain_cases = ((all_labels_pred >= 0.45) & (all_labels_pred <= 0.55)).sum()
        self.logger.info(f"Number of uncertain cases predicted near 0.5: {uncertain_cases}")

        if len(all_labels_gt) > 0:
            all_gt = np.concatenate([gt for gt, _ in all_labels_gt], axis=0)
            all_masks = np.concatenate([mask for _, mask in all_labels_gt], axis=0)

            aucs = []
            for i in range(self.model.num_classes):
                concept_mask = all_masks[:, i]
                if concept_mask.sum() > 0:
                    try:
                        auc = roc_auc_score(all_gt[concept_mask, i], all_labels_pred[concept_mask, i])
                        aucs.append(auc)
                    except ValueError:
                        self.logger.warning(f"AUC calculation failed for concept {i}")
                        continue

            test_thresholds = [0.4, 0.5, 0.7]
            best_metrics = {}
            best_f1 = 0.0

            for t in test_thresholds:
                f1s, precisions, recalls = [], [], []
                for i in range(self.model.num_classes):
                    concept_mask = all_masks[:, i]
                    if concept_mask.sum() > 0:
                        pred_labels = (all_labels_pred[concept_mask, i] > t).astype(int)
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            all_gt[concept_mask, i],
                            pred_labels,
                            average='binary',
                            zero_division=0
                        )
                        precisions.append(precision)
                        recalls.append(recall)
                        f1s.append(f1)

                        # print(pred_labels[:5])

                mean_f1 = np.mean(f1s)
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_metrics = {
                        f"{phase}_best_threshold": t,
                        f"{phase}_mean_f1": mean_f1,
                        f"{phase}_mean_precision": np.mean(precisions),
                        f"{phase}_mean_recall": np.mean(recalls)
                    }

            metrics.update(best_metrics)
            if aucs:
                metrics[f"{phase}_auc"] = np.mean(aucs)

        self.logger.info(
            f"{phase.capitalize()} metrics: {json.dumps({k: v if isinstance(v, list) else float(v) for k, v in metrics.items()}, indent=2)}")

        self.model.train()
        return metrics
