import os
import time
import json
import torch
import evaluate
import numpy as np

from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_scheduler
from torch.amp import GradScaler, autocast

from utils.logger import LoggerManager
from utils.training_monitor import TrainingMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


class RadiologyReportDecoderTrainer:
    def __init__(
            self,
            model,
            config,
            loss_fn,
            train_loader,
            val_loader=None,
            test_loader=None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            output_dir='./output/generation',
            mixed_precision=False,
    ):
        self.device = device
        self.config = config
        self.loss_fn = loss_fn
        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.max_length = config.get('max_length', 256)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        self.init_lr = float(config.get('init_lr', 5e-6))
        self.min_lr = float(config.get('min_lr', 1e-6))
        self.warmup_lr = float(config.get('warmup_lr', 1e-7))
        self.weight_decay = config.get('weight_decay', 0.02)
        self.warmup_steps = config.get('warmup_steps', 500)

        self.log_interval = config.get('log_interval', 10)
        self.save_interval = config.get('save_interval', 500)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.run_validation = config.get('run_validation', False)

        steps_per_epoch = len(train_loader)
        if config.get('max_epochs') is None:
            self.max_steps = config.get('max_steps', 10000)
            self.num_epochs = (self.max_steps + steps_per_epoch - 1) // steps_per_epoch
        else:
            self.num_epochs = config.get('max_epochs', 10)
            self.max_steps = steps_per_epoch * self.num_epochs

        self.model.train()
        param_groups = [{'params': [p for n, p in model.named_parameters() if p.requires_grad]}]
        self.optimizer = AdamW(
            param_groups,
            lr=self.init_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        scheduler_type = config.get('scheduler_type', 'linear')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_steps,
                eta_min=self.min_lr
            )
        elif scheduler_type == 'linear':
            self.scheduler = get_scheduler(
                name='linear',
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        self.scaler = GradScaler('cuda', init_scale=0.2) if torch.cuda.is_available() and mixed_precision else None

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_bleu = 0.0

        self.logger = LoggerManager.get_logger(__name__)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 3)
        self.train_monitor = TrainingMonitor(
            self.output_dir,
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
        self.logger.info("Starting training")

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


                if early_stop:
                    self.logger.info(
                        f"No improvement in validation loss for {self.early_stopping_patience} epochs. Stopping training.")
                    try:
                        self.train_monitor.plot_all_metrics()
                    except Exception as e:
                        self.logger.warning(f"Error generating plots: {e}")
                    break

        self.logger.info("Training completed")

        if self.test_loader is not None:
            self.logger.info("Running final evaluation on test set")
            self.evaluate(test=True)

        try:
            self.train_monitor.plot_all_metrics()
        except Exception as e:
            self.logger.error(f"Error generating loss plots: {e}")
        self.logger.info("Training and evaluation completed")

    def train_step(self, batch):
        if self.global_step < self.warmup_steps:
            self.warmup_lr_schedule(self.global_step)

        if self.scaler is not None:
            with autocast('cuda'):
                outputs = self.model(batch, self.device)
                loss_dict = self.loss_fn(**outputs)
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
            loss_dict = self.loss_fn(**outputs)
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

        references, hypotheses = [], []
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"{phase.capitalize()} Evaluation"):
            images = batch['images'].to(self.device)
            reports = batch['texts']

            outputs = self.model(batch, self.device)
            loss_dict = self.loss_fn(**outputs)
            loss = loss_dict['total']
            total_loss += loss

            if phase == 'val' and not self.run_validation:
                continue

            captions = self.model.generate(
                images=images,
                findings=batch['labels'],
                temperature=1.0,
                repetition_penalty=1.4,
                device=self.device,
            )

            references.extend([[r] for r in reports])
            hypotheses.extend(captions)

        if phase == "test":
            rouge_scores = rouge.compute(predictions=hypotheses, references=references)
            bleu_result = bleu.compute(predictions=hypotheses, references=references, max_order=4, smooth=True)
            bleu_scores = {
                f"{phase}_bleu": bleu_result["bleu"],
                f"{phase}_bleu1": bleu_result.get("precisions", [0] * 4)[0],
                f"{phase}_bleu2": bleu_result.get("precisions", [0] * 4)[1],
                f"{phase}_bleu3": bleu_result.get("precisions", [0] * 4)[2],
                f"{phase}_bleu4": bleu_result.get("precisions", [0] * 4)[3],
            }

            metrics = {
                f"{phase}_loss": total_loss / len(dataloader),
                **bleu_scores,
                f"{phase}_rouge1": rouge_scores["rouge1"],
                f"{phase}_rouge2": rouge_scores["rouge2"],
                f"{phase}_rougeL": rouge_scores["rougeL"],
            }
        else:
            # bleu_score = bleu.compute(predictions=hypotheses, references=references)
            metrics = {
                f"{phase}_loss": total_loss / len(dataloader),
                # f"{phase}_bleu": bleu_score["bleu"],
            }

        if phase == 'test' or self.run_validation:
            self.logger.info(f"{phase.capitalize()} sample:")
            samples = np.random.randint(0, len(hypotheses), size=3)
            for i in samples:
                self.logger.info(f"Reference: {references[i][0]}")
                self.logger.info(f"Prediction: {hypotheses[i]}")
                self.logger.info("-" * 50)

        eval_metrics = {
            k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v)
            for k, v in metrics.items()
        }
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
            checkpoint_path = os.path.join(self.output_dir, f'clip-xrgen-{step}.pt')
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        if is_best:
            if suffix:
                best_path = os.path.join(self.output_dir, f'clip-xrgen_{suffix}.pt')
            else:
                best_path = os.path.join(self.output_dir, 'clip-xrgen_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint {checkpoint_path} does not exist. Starting from scratch.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']

        self.logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.epoch}, step {self.global_step})")
