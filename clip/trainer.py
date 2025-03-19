import os
import time
import json
import torch
import numpy as np
import logging
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from clip.loss import CCALoss


class CLIPTrainer:
    """
    Trainer for fine-tuning CLIP models on the ARRG (Aligned Radiology Report Generation) task.
    Optimizes for image-text alignment while leveraging medical concept labels.
    """

    def __init__(
            self,
            model,
            train_loader,
            val_loader=None,
            test_loader=None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_steps=1000,
            max_steps=100000,
            output_dir='./output',
            log_interval=100,
            save_interval=1000,
            gradient_accumulation_steps=1,
            mixed_precision=True,
            freeze_vision_encoder=False,
            freeze_text_encoder=False,
            max_grad_norm=1.0
    ):
        """
        Initialize the CLIP Medical Trainer.

        :param model: CLIP model with medical concept prediction capabilities
        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :param test_loader: DataLoader for test data (optional)
        :param device: device to train on ('cuda' or 'cpu')
        :param learning_rate: learning rate for optimizer
        :param weight_decay: weight decay for regularization
        :param warmup_steps: number of warmup steps for learning rate scheduler
        :param max_steps: maximum number of training steps
        :param output_dir: directory to save checkpoints and logs
        :param log_interval: how often to log training metrics
        :param save_interval: how often to save model checkpoints
        :param gradient_accumulation_steps: number of steps to accumulate gradients
        :param mixed_precision: whether to use mixed precision training
        :param freeze_vision_encoder: whether to freeze the vision encoder weights
        :param freeze_text_encoder: whether to freeze the text encoder weights
        :param max_grad_norm: maximum gradient norm for gradient clipping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm

        os.makedirs(output_dir, exist_ok=True)

        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.loss_fn = CCALoss(
            temperature=0.07,
            concept_weight=0.5,
            concept_sim_weight=0.3
        )

        # TODO: set up optimizer with parameter groups
        param_groups = []

        # vision encoder parameters
        if not freeze_vision_encoder:
            vision_params = [p for n, p in model.named_parameters()
                             if "visual" in n and p.requires_grad]
            param_groups.append({"params": vision_params, "lr": learning_rate})
        else:
            for n, p in model.named_parameters():
                if "visual" in n:
                    p.requires_grad = False

        # text encoder parameters
        if not freeze_text_encoder:
            text_params = [p for n, p in model.named_parameters()
                           if ("transformer" in n or "token_embedding" in n or
                               "positional_embedding" in n or "ln_final" in n or
                               "text_projection" in n) and p.requires_grad]
            param_groups.append({"params": text_params, "lr": learning_rate})
        else:
            for n, p in model.named_parameters():
                if ("transformer" in n or "token_embedding" in n or
                        "positional_embedding" in n or "ln_final" in n or
                        "text_projection" in n):
                    p.requires_grad = False

        # medical concept related parameters (always trainable)
        concept_params = [p for n, p in model.named_parameters()
                          if ("medical_concept" in n or "concept_embedding" in n) and p.requires_grad]
        param_groups.append({"params": concept_params, "lr": learning_rate * 5})  # higher LR for new parameters

        # set up optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_steps)

        # set up gradient scaler for mixed precision training
        self.scaler = GradScaler() if mixed_precision and device == 'cuda' else None

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epoch = 0

        self.model = self.model.to(self.device)

        # log trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")

    def train(self, epochs=None):
        self.logger.info("Starting training")
        self.model.train()

        # step_counter = 0
        if epochs is None:
            steps_per_epoch = len(self.train_loader)
            epochs = (self.max_steps + steps_per_epoch - 1) // steps_per_epoch

        for epoch in range(epochs):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}/{epochs}")

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
        text_tokens = batch['text_tokens'].to(self.device)
        medical_concepts = batch['labels'].to(self.device)

        # get attention mask if available
        # attention_mask = batch['attn_mask']
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(self.device)

        # zero gradients at the beginning of the accumulation steps
        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        # forward pass with mixed precision if enabled
        if self.mixed_precision and self.device == 'cuda':
            with autocast():
                outputs = self.model(images, text_tokens, medical_concepts)

                loss, loss_dict = self.loss_fn(outputs, medical_concepts)

                original_loss = loss_dict['total_loss']
                loss = loss / self.gradient_accumulation_steps

            # backward pass with scaled gradients
            self.scaler.scale(loss).backward()

            # update weights if we've accumulated enough gradients
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
        else:
            outputs = self.model(images, text_tokens, medical_concepts)

            loss, loss_dict = self.loss_fn(outputs, medical_concepts)

            original_loss = loss_dict['total_loss']
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
        all_concepts_pred = []
        all_concepts_true = []
        all_img_embeds = []
        all_text_embeds = []
        all_concept_embeds = []

        self.logger.info(f"Running evaluation on {phase} set")

        for batch in tqdm(dataloader, desc=f"{phase.capitalize()} Evaluation"):
            if batch is None:
                continue

            images = batch['image'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            medical_concepts = batch['labels'].to(self.device)

            # attention_mask = batch['attn_mask']
            # if attention_mask is not None:
            #     attention_mask = attention_mask.to(self.device)

            outputs = self.model(images, text_tokens, medical_concepts)

            loss, loss_dict = self.loss_fn(outputs, medical_concepts)

            # store predictions and ground truth
            total_loss += loss_dict["total_loss"]

            # store embeddings and predictions for later analysis
            all_img_embeds.append(outputs['image_features'].cpu())
            all_text_embeds.append(outputs['text_features'].cpu())
            if 'concepts_embeddings' in outputs:
                all_concept_embeds.append(outputs['concepts_embeddings'].cpu())

            # store concept predictions for metrics calculation
            concepts_pred = torch.sigmoid(outputs['concepts_logits']).cpu().numpy()
            all_concepts_pred.append(concepts_pred)

            # convert uncertain (-1) to masked values for metrics calculation
            concepts_true = medical_concepts.clone().cpu().numpy()
            mask = (concepts_true != -1)
            concepts_true[concepts_true == -1] = 0  # replace uncertain with 0 for metrics
            all_concepts_true.append((concepts_true, mask))

        # calculate average loss
        avg_loss = total_loss / len(dataloader)

        # concatenate predictions and ground truth
        all_concepts_pred = np.concatenate(all_concepts_pred, axis=0)

        # calculate metrics only on non-uncertain labels
        metrics = {f"{phase}_loss": avg_loss}

        # calculate AUC and other metrics for concept prediction
        if len(all_concepts_true) > 0:
            # combine all masks and ground truth
            all_gt = np.concatenate([gt for gt, _ in all_concepts_true], axis=0)
            all_masks = np.concatenate([mask for _, mask in all_concepts_true], axis=0)

            # calculate AUC per concept
            aucs = []
            precision_list = []
            recall_list = []
            f1_list = []

            # get per-concept metrics
            for i in range(self.model.num_medical_concepts):
                concept_mask = all_masks[:, i]
                if concept_mask.sum() > 0:  # only calculate if we have non-masked values
                    # AUC
                    try:
                        auc = roc_auc_score(all_gt[concept_mask, i], all_concepts_pred[concept_mask, i])
                        aucs.append(auc)
                    except ValueError:
                        # this can happen if there's only one class in the ground truth
                        pass

                    # precision, recall, F1 at threshold 0.5
                    pred_binary = (all_concepts_pred[concept_mask, i] > 0.5).astype(int)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        all_gt[concept_mask, i],
                        pred_binary,
                        average='binary'
                    )
                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1)

            # add metrics to the result dict
            if aucs:
                metrics[f"{phase}_concept_auc_avg"] = np.mean(aucs)
            if precision_list:
                metrics[f"{phase}_concept_precision_avg"] = np.mean(precision_list)
            if recall_list:
                metrics[f"{phase}_concept_recall_avg"] = np.mean(recall_list)
            if f1_list:
                metrics[f"{phase}_concept_f1_avg"] = np.mean(f1_list)

        # calculate retrieval metrics
        if len(all_img_embeds) > 0 and len(all_text_embeds) > 0:
            # concatenate all embeddings
            img_embeds = torch.cat(all_img_embeds, dim=0)
            text_embeds = torch.cat(all_text_embeds, dim=0)

            # normalize embeddings for cosine similarity
            img_embeds = img_embeds / img_embeds.norm(dim=1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

            # calculate similarity scores
            similarity = (img_embeds @ text_embeds.T).numpy()

            # calculate retrieval metrics
            metrics.update(self._calculate_retrieval_metrics(similarity, phase))

        # log metrics
        self.logger.info(
            f"{phase.capitalize()} metrics: {json.dumps({k: float(v) for k, v in metrics.items()}, indent=2)}")

        # set model back to training mode
        self.model.train()

        return metrics

    def _calculate_retrieval_metrics(self, similarity, phase):
        """Calculate retrieval metrics (recall@k) from similarity matrix."""
        metrics = {}
        batch_size = similarity.shape[0]

        # get indices of highest similarities
        ranks = {}
        for retrieval_type in ['i2t', 't2i']:  # image-to-text and text-to-image
            if retrieval_type == 'i2t':
                # for each image, find the rank of the correct text
                ranked_sim = np.argsort(similarity, axis=1)[:, ::-1]
                ranks[retrieval_type] = np.where(ranked_sim == np.arange(batch_size)[:, np.newaxis])[1]
            else:
                # for each text, find the rank of the correct image
                ranked_sim = np.argsort(similarity.T, axis=1)[:, ::-1]
                ranks[retrieval_type] = np.where(ranked_sim == np.arange(batch_size)[:, np.newaxis])[1]

        # calculate recall@k
        for k in [1, 5, 10]:
            for retrieval_type in ['i2t', 't2i']:
                metrics[f"{phase}_{retrieval_type}_recall@{k}"] = (ranks[retrieval_type] < k).mean()

        # calculate median rank
        for retrieval_type in ['i2t', 't2i']:
            metrics[f"{phase}_{retrieval_type}_median_rank"] = np.median(ranks[retrieval_type]) + 1

        return metrics

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