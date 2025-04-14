import os
import time
import json
import torch
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from constants import CHEXPERT_LABELS
from clip.loss import ContrastiveConceptualAlignmentLoss
from clip.tokenizer import CLIPTokenizer


class CLIPTrainer:
    """
    Trainer for fine-tuning a foundational CLIP model on language-vision contrastive pre-training for radiology image datasets.
    """

    def __init__(
            self,
            model,
            train_loader,
            val_loader=None,
            test_loader=None,
            num_epochs=None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            learning_rate=1e-5,
            weight_decay=0.2,
            warmup_steps=500,
            max_steps=30000,
            output_dir='./output',
            log_interval=100,
            save_interval=1000,
            gradient_accumulation_steps=1,
            mixed_precision=True,
            max_grad_norm=1.0
    ):
        """
        Initializes the trainer.

        :param model: CLIP foundational model
        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :param test_loader: DataLoader for test data (optional)
        :param device: device to train on ('cuda' or 'cpu')
        :param num_epochs: number of epochs to train for (if None, max_steps is used)
        :param learning_rate: learning rate for optimizer
        :param weight_decay: weight decay for regularization
        :param warmup_steps: number of warmup steps for learning rate scheduler
        :param max_steps: maximum number of training steps
        :param output_dir: directory to save checkpoints and logs
        :param log_interval: how often to log training metrics
        :param save_interval: how often to save model checkpoints
        :param gradient_accumulation_steps: number of steps to accumulate gradients
        :param mixed_precision: whether to use mixed precision training
        :param max_grad_norm: maximum gradient norm for gradient clipping
        """
        self.model = model
        self.tokenizer = CLIPTokenizer()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm

        os.makedirs(output_dir, exist_ok=True)

        self.logger = self.__setup_logger(__name__)

        self.loss_fn = ContrastiveConceptualAlignmentLoss(use_soft_concept_loss=False)

        concept_params = [p for n, p in model.named_parameters() if
                          ("medical_concept" in n or "concept_embedding" in n) and p.requires_grad]
        clip_params = [p for n, p in model.named_parameters() if
                       not ("medical_concept" in n or "concept_embedding" in n) and p.requires_grad]

        param_groups = [
            {"params": concept_params, "lr": learning_rate * 0.5},
            {"params": clip_params, "lr": learning_rate}
        ]

        # freeze non-used auxiliary classification head
        for param in concept_params:
            param.requires_grad = False

        steps_per_epoch = len(train_loader)
        if num_epochs is not None:
            self.num_epochs = num_epochs
            self.max_steps = steps_per_epoch * num_epochs
        else:
            self.max_steps = max_steps
            self.num_epochs = (max_steps + steps_per_epoch - 1) // steps_per_epoch

        # set up optimizer and scheduler
        self.optimizer = AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_steps)

        # set up gradient scaler for mixed precision training
        self.scaler = GradScaler('cuda') if mixed_precision and device == 'cuda' else None

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epoch = 0

        self.model = self.model.to(self.device)

        # log trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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

    @staticmethod
    def __compute_pos_weights(dataloader, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        total_pos = torch.zeros(num_classes, dtype=torch.float32).to(device)
        total_neg = torch.zeros(num_classes, dtype=torch.float32).to(device)
        total_samples = 0

        for batch in tqdm(dataloader, desc="Computing pos weights"):
            labels = batch['labels'].to(device)
            mask = labels != -1

            pos = (labels == 1) & mask
            neg = (labels == 0) & mask

            total_pos += pos.sum(dim=0).float()
            total_neg += neg.sum(dim=0).float()
            total_samples += labels.size(0)

        pos_weights = total_neg / (total_pos + 1e-8)
        return pos_weights

    def train(self):
        self.logger.info("Starting training")
        self.model.train()

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")

            epoch_loss = 0.0
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                if self.global_step >= self.max_steps:
                    self.logger.info(f"Reached maximum steps {self.max_steps}. Stopping training.")
                    break

                loss = self.train_step(batch)
                epoch_loss += loss

                # log loss
                if (self.global_step + 1) % self.log_interval == 0:
                    self.logger.info(f"Step {self.global_step + 1}: Loss = {loss:.4f}")

                # save checkpoint
                if (self.global_step + 1) % self.save_interval == 0:
                    self.save_checkpoint(step=self.global_step + 1)
                    # run validation
                    val_metrics = self.evaluate()
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint(is_best=True)

                self.global_step += 1

            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. Average loss: {epoch_loss / len(self.train_loader):.4f}")

            # run validation at the end of each epoch
            val_metrics = self.evaluate()
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(is_best=True)

        if self.test_loader is not None:
            self.logger.info("Running final evaluation on test set")
            test_metrics = self.evaluate(test=True)
            self.logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")

        self.logger.info("Training completed")

    def train_step(self, batch):
        images = batch['image'].to(self.device)
        text_tokens = self.tokenizer.encode(batch['report'], truncate=True, extended_context=self.model.extended_context).to(self.device)
        labels = batch['labels'].to(self.device)

        # batch attention mask not used in this context for pretraining CLIP, due to using pre-computed causal mask

        # zero gradients at the beginning of the accumulation steps
        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        # forward pass with mixed precision if enabled
        if self.mixed_precision and self.device == 'cuda':
            with autocast('cuda'):
                outputs = self.model(images, text_tokens)

                loss, loss_dict = self.loss_fn(outputs, labels)

                original_loss = loss_dict['clip_loss']
                loss = loss / self.gradient_accumulation_steps

            # backward pass with scaled gradients
            self.scaler.scale(loss).backward()

            # update weights if we've accumulated enough gradients
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision and self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
        else:
            outputs = self.model(images, text_tokens)

            loss, loss_dict = self.loss_fn(outputs, labels)

            original_loss = loss_dict['clip_loss']
            # scale loss by gradient accumulation steps
            loss = loss / self.gradient_accumulation_steps

            # backward pass
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
        phase = "test" if test else "val"

        total_loss = 0.0
        all_img_embeds, all_text_embeds, all_labels = [], [], []

        for batch in tqdm(dataloader, desc=f"{phase.capitalize()} Evaluation", position=0, leave=True, dynamic_ncols=True):
            images = batch['image'].to(self.device)
            text_tokens = self.tokenizer.encode(batch['report'], truncate=True, extended_context=self.model.extended_context).to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(images, text_tokens)

            loss, loss_dict = self.loss_fn(outputs, labels)

            # store predictions and ground truth
            total_loss += loss_dict["clip_loss"]

            all_labels.append(labels.cpu())
            all_img_embeds.append(outputs['image_features'].cpu())
            all_text_embeds.append(outputs['text_features'].cpu())

        # calculate average loss
        avg_loss = total_loss / len(dataloader)
        metrics = {f"{phase}_loss": avg_loss}

        # calculate retrieval metrics
        if len(all_img_embeds) > 0 and len(all_text_embeds) > 0:
            # concatenate all embeddings
            img_embeds = torch.cat(all_img_embeds, dim=0)
            text_embeds = torch.cat(all_text_embeds, dim=0)
            labels_tensor = torch.cat(all_labels, dim=0)

            # normalize embeddings for cosine similarity
            # img_embeds = img_embeds / img_embeds.norm(dim=1, keepdim=True)
            # text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

            img_embeds = img_embeds.to(self.device)
            text_embeds = text_embeds.to(self.device)

            logit_scale = self.model.logit_scale.exp().clamp(max=100)
            # calculate similarity scores
            similarity = (logit_scale * (img_embeds @ text_embeds.T)).cpu().numpy()

            # calculate retrieval metrics
            metrics.update(self._calculate_retrieval_metrics(similarity, phase))

            # calculate zero-shot accuracy
            metrics.update(self._calculate_zero_shot_accuracy(img_embeds, labels_tensor, phase))

            # save embeddings for further analysis
            # self.save_embeddings(img_embeds, text_embeds, phase)

        # log metrics
        formatted_metrics = {k: round(float(v), 10) for k, v in metrics.items() }
        self.logger.info(
            f"{phase.capitalize()} metrics: {json.dumps(formatted_metrics, indent=2)}")

        self.model.train()
        return metrics

    @staticmethod
    def _calculate_retrieval_metrics(similarity, phase):
        """
        Calculate retrieval metrics (recall@k) from similarity matrix.
        :param similarity: similarity matrix of shape (batch_size, batch_size) where each entry is the cosine similarity between image and text pairs.
        :param phase: phase of evaluation ('val', 'test')
        """
        metrics = {}
        batch_size = similarity.shape[0]

        # for each image, rank all texts
        ranked_sim = np.argsort(similarity, axis=1)[:, ::-1]
        ranks = np.array([
            np.where(ranked_sim[i] == i)[0][0]
            for i in range(batch_size)
        ])

        # calculate recall@k for k in [1, 5, 10]
        for k in [1, 5, 10]:
            recall = (ranks < k).mean()
            metrics[f"{phase}_recall@{k}"] = recall

        median_rank = np.median(ranks) + 1
        mrr = np.mean(1.0 / (ranks + 1))
        metrics[f"{phase}_median_rank"] = median_rank
        metrics[f"{phase}_mrr"] = mrr

        return metrics

    def _calculate_zero_shot_accuracy(self, image_embeds, labels, phase):
        image_embeds = image_embeds.to(self.device)
        labels = labels.to(self.device)

        accuracy_per_class = []
        total_valid = 0

        for index, classname in enumerate(CHEXPERT_LABELS):
            # handle "no finding" class separately
            if classname.lower() == "no finding":
                prompts = [f"No Finding", "Abnormal"]
            else:
                prompts = [f"{classname}", f"No {classname}"]

            text_tokens = self.tokenizer.encode(prompts, truncate=True, extended_context=self.model.extended_context).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            logit_scale = self.model.logit_scale.exp().clamp(max=100)
            logits = logit_scale * image_embeds @ text_features.T

            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            class_labels = labels[:, index].cpu().numpy()
            mask = class_labels != -1
            if mask.sum() > 0:
                y_true = class_labels[mask]
                y_score = probs[:, 0][mask]

                try:
                    auc = roc_auc_score(y_true, y_score)
                    accuracy_per_class.append(auc)
                except:
                    accuracy_per_class.append(0.0)

                total_valid += mask.sum()

        mean_auc = np.mean(accuracy_per_class) if accuracy_per_class else 0.0
        per_class_accuracy = {
            f"{phase}_zs_auc_{classname.lower().replace(' ', '_')}": acc
            for classname, acc in zip(CHEXPERT_LABELS, accuracy_per_class)
        }

        return {
            f"{phase}_mean_auc": mean_auc,
            **per_class_accuracy
        }

    def save_embeddings(self, img_embeds, text_embeds, phase):
        embedding_data = []
        for i in range(img_embeds.shape[0]):
            embedding_data.append({
                "uid": f"img_{i}",
                "type": "image",
                "embedding": img_embeds[i].cpu().numpy()
            })

        for i in range(text_embeds.shape[0]):
            embedding_data.append({
                "uid": f"text_{i}",
                "type": "text",
                "embedding": text_embeds[i].cpu().numpy()
            })

        emb_df = pd.DataFrame(embedding_data)
        emb_df.to_pickle(os.path.join(self.output_dir, f"{phase}_embeddings.pkl"))


    def save_checkpoint(self, step=None, is_best=False):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss
        }

        if step is not None:
            checkpoint_path = os.path.join(self.output_dir, f'checkpoint-{step}.pt')
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        if is_best:
            best_path = os.path.join(self.output_dir, 'clip_cxr_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path):
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        self.logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")