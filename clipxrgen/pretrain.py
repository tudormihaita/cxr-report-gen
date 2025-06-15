import os
import json
import time
import torch
import numpy as np

from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from utils.logger import LoggerManager
from utils.training_monitor import TrainingMonitor
from eval.metrics.retrieval import compute_retrieval_recall_metrics


class ImageTextAlignmentTrainer:
    def __init__(
            self,
            model,
            config,
            loss_fn,
            train_loader,
            val_loader=None,
            test_loader=None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            output_dir='./output/pretrain',
            mixed_precision=False,
    ):
        self.device = device
        self.config = config
        self.model = model.to(self.device)
        self.loss_fn = loss_fn

        self.max_length = self.config.get('max_length', 143)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.init_lr = float(self.config.get('init_lr', 1e-5))
        self.min_lr = float(self.config.get('min_lr', 1e-6))
        self.warmup_lr = float(self.config.get('warmup_lr', 1e-7))
        self.weight_decay = self.config.get('weight_decay', 0.05)

        self.early_stopping_patience = self.config.get('early_stopping_patience', 5)
        self.warmup_steps = self.config.get('warmup_steps', 1000)
        self.scheduler_type = self.config.get('scheduler_type', 'cosine')
        self.restarts_t_mult = self.config.get('restarts_t_mult', 1)

        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.log_interval = self.config.get('log_interval', 100)
        self.save_interval = self.config.get('save_interval', 1000)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        steps_per_epoch = len(train_loader)
        if self.config.get('max_epochs') is None:
            self.max_steps = self.config.get('max_steps', 10000)
            self.num_epochs = (self.max_steps + steps_per_epoch - 1) // steps_per_epoch
        else:
            self.num_epochs = self.config.get('max_epochs', 10)
            self.max_steps = steps_per_epoch * self.num_epochs

        # visual encoder typically learns slower than other components
        self.optimizer = AdamW(
            [
                {'params': [p for n, p in model.named_parameters() if 'image_encoder' in n], 'lr': self.init_lr * 0.1},
                {'params': [p for n, p in model.named_parameters() if 'image_encoder' not in n]}
            ],
            lr=self.init_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        # initialize learning rate scheduler
        if self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_steps,
                eta_min=self.min_lr
            )
        elif self.scheduler_type == 'cosine_restarts':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.max_steps,
                T_mult=self.restarts_t_mult,
                eta_min=self.min_lr
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

        self.scaler = GradScaler('cuda') if torch.cuda.is_available() and mixed_precision else None

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_r_mean = float('inf')

        self.logger = LoggerManager.get_logger(__name__)
        self.train_monitor = TrainingMonitor(
            output_dir=self.output_dir,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_metric='val_loss'
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")

    def warmup_lr_schedule(self, step):
        lr = self.warmup_lr + (self.init_lr - self.warmup_lr) * step / self.warmup_steps
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        self.model.train()
        self.logger.info("Start training")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")

            epoch_loss = 0.0
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")):
                if (self.global_step + 1) >= self.max_steps:
                    self.logger.info(f"Reached maximum steps {self.max_steps}. Stopping training.")
                    break

                if self.global_step % self.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()

                loss = self.train_step(batch)
                epoch_loss += loss

                if (self.global_step + 1) % self.log_interval == 0:
                    self.train_monitor.log_metric('train_loss', loss, self.global_step + 1)
                    self.logger.info(f"Step {self.global_step + 1}: Loss = {loss:.4f}")

                if (self.global_step + 1) % self.save_interval == 0:
                    self.save_checkpoint(step=self.global_step + 1)

                self.global_step += 1

            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / len(self.train_loader)
            self.logger.info(
                f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. Average loss: {avg_loss:.4f}"
            )

            if self.val_loader is not None:
                val_metrics = self.evaluate()
                val_loss = val_metrics['val_loss']

                early_stop = self.train_monitor.log_metric('val_loss', val_loss, self.global_step + 1)

                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(is_best=True, suffix='best_loss')

                if val_metrics['r_mean'] < self.best_r_mean:
                    self.best_r_mean = val_metrics['r_mean']
                    self.save_checkpoint(is_best=True, suffix='best_retrieval')

                if early_stop:
                    self.logger.info(
                        f"No improvement in validation loss for {self.early_stopping_patience} epochs. Stopping training.")
                    try:
                        self.train_monitor.plot_all_metrics()
                    except Exception as e:
                        self.logger.warning(f"Error generating plots: {e}")
                    break

        if self.test_loader is not None:
            self.logger.info("Running final evaluation on test set")
            self.evaluate(test=True)

        try:
            self.train_monitor.plot_all_metrics()
        except Exception as e:
            self.logger.warning(f"Error generating plots: {e}")
        self.logger.info("Training completed")

    def train_step(self, batch):
        if self.global_step < self.warmup_steps:
            self.warmup_lr_schedule(self.global_step)

        if self.scaler is not None:
            with autocast('cuda'):
                outputs = self.model(batch, self.device)
                loss_dict = self.loss_fn(**outputs, is_train=True)
                loss = loss_dict['total']
                scaled_loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(scaled_loss).backward()

            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.global_step >= self.warmup_steps:
                    self.scheduler.step()
        else:
            outputs = self.model(batch, self.device)
            loss_dict = self.loss_fn(**outputs, is_train=True)
            loss = loss_dict['total']
            scaled_loss = loss / self.gradient_accumulation_steps

            scaled_loss.backward()

            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()

                if self.global_step >= self.warmup_steps:
                    self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, test=False):
        self.model.eval()
        dataloader = self.test_loader if test else self.val_loader
        phase = "test" if test else "val"

        total_loss = 0.0
        uids = []
        all_texts = []
        all_labels = []
        all_image_embeddings = []
        all_text_embeddings = []

        for batch in tqdm(dataloader, desc=f"{phase.capitalize()} Evaluation"):
            images = batch['images']
            texts = batch['texts']
            tokens = batch['text_tokens']
            labels = batch['labels']

            if 'uid' in batch:
                uids.extend(batch['uid'])

            outputs = self.model(batch, self.device)
            loss_dict = self.loss_fn(**outputs)
            loss = loss_dict['total']
            total_loss += loss

            img_emb = self.model.encode_image(images.to(self.device))
            img_emb = self.model.image_projection(img_emb) if self.model.projection else img_emb
            img_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)
            img_emb = img_emb.detach().cpu().numpy()

            text_emb = self.model.encode_text(tokens.to(self.device))
            text_emb = self.model.text_projection(text_emb) if self.model.projection else text_emb
            text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
            text_emb = text_emb.detach().cpu().numpy()

            all_image_embeddings.append(img_emb)
            all_text_embeddings.append(text_emb)
            all_texts.extend(texts)
            all_labels.append(labels.detach().cpu().numpy())

        all_image_embeddings = np.concatenate(all_image_embeddings)
        all_text_embeddings = np.concatenate(all_text_embeddings)

        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        retrieval_metrics = compute_retrieval_recall_metrics(all_image_embeddings, all_text_embeddings, all_texts)

        eval_metrics = {
            f"{phase}_loss": avg_loss,
        }
        eval_metrics.update(retrieval_metrics)

        if not test:
            for k, v in retrieval_metrics.items():
                self.train_monitor.log_metric(k, v, self.global_step + 1)

        eval_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in eval_metrics.items()}
        self.logger.info(f"{phase.capitalize()} metrics: {json.dumps(eval_metrics, indent=2)}")

        self.model.train()
        return eval_metrics

    def save_checkpoint(self, step=None, is_best=False, suffix=None):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config
        }

        if step is not None:
            checkpoint_path = os.path.join(self.output_dir, f'clip-xrgen-pt_ckpt-{step}.pt')
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        if is_best:
            if suffix:
                best_path = os.path.join(self.output_dir, f'clip-xrgen-pt_{suffix}.pt')
            else:
                best_path = os.path.join(self.output_dir, 'clip-xrgen-pt_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint {checkpoint_path} does not exist. Starting from scratch.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.epoch}, step {self.global_step})")
