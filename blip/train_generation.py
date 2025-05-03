import os
import time
import json
import math
import torch
import evaluate

from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from utils.logger import LoggerManager
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
# cider = evaluate.load("cider")

def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Linear warmup of learning rate"""
    lr = init_lr + (max_lr - init_lr) * step / max_step
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class BLIPReportGenTrainer:
    def __init__(self,
                 model,
                 config,
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 output_dir='output/generation',
                 mixed_precision=False
                 ):
        self.device = device
        self.config = config
        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.max_length = config['max_length']
        self.min_length = config['min_length']
        self.grad_accumulation_steps = config['grad_accumulation_steps']
        self.max_grad_norm = config['max_grad_norm']

        self.init_lr = float(config.get('init_lr', 1e-5))
        self.min_lr = float(config.get('min_lr', 1e-6))
        self.warmup_lr = float(config.get('warmup_lr', 1e-7))
        self.weight_decay = config.get('weight_decay', 0.01)

        self.log_interval = config['log_interval']
        self.save_interval = config['save_interval']
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        steps_per_epoch = len(train_loader)
        if config['max_epochs'] is None:
            self.max_steps = config['max_steps']
            self.num_epochs = (config['max_steps'] + steps_per_epoch - 1) // steps_per_epoch
        else:
            self.max_steps = steps_per_epoch * config['max_epochs']
            self.num_epochs = config['max_epochs']

        self.optimizer = AdamW(self.model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.max_steps - self.warmup_steps,
            T_mult=1,
            eta_min=self.min_lr
        )
        self.scaler = GradScaler('cuda') if torch.cuda.is_available() and mixed_precision else None

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        self.logger = LoggerManager.get_logger(__name__)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")

    def train(self):
        self.model.train()
        self.logger.info(f"Starting training for {self.num_epochs} epochs")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            # cosine_lr_schedule(self.optimizer, epoch, self.num_epochs, self.config['init_lr'], self.config['min_lr'])

            epoch_loss = 0.0
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Train")):
                if self.global_step >= self.max_steps:
                    self.logger.info(f"Reached maximum steps {self.max_steps}. Stopping training.")
                    break

                if (self.global_step + 1) % self.grad_accumulation_steps == 0:
                    self.optimizer.zero_grad()

                loss = self.train_step(batch)
                epoch_loss += loss

                if (self.global_step + 1) % self.log_interval == 0:
                    self.logger.info(f"Step {self.global_step + 1}: Loss = {loss:.4f}")

                if (self.global_step + 1) % self.save_interval == 0:
                    self.save_checkpoint(step=self.global_step + 1)

                self.global_step += 1

            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. Average loss: {epoch_loss / len(self.train_loader):.4f}")

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
        reports = batch['report']

        if self.global_step < self.warmup_steps:
            warmup_lr_schedule(self.optimizer, self.global_step, self.warmup_steps, self.warmup_lr, self.init_lr)

        if self.scaler is not None:
            with autocast('cuda'):
                loss = self.model(images, reports)

                original_loss = loss.clone()
                loss = loss / self.grad_accumulation_steps

            self.scaler.scale(loss).backward()

            if self.global_step % self.grad_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.global_step >= self.warmup_steps:
                    self.scheduler.step()
        else:
            loss = self.model(images, reports)

            original_loss = loss.clone()
            loss = loss / self.grad_accumulation_steps
            loss.backward()

            if (self.global_step + 1) % self.grad_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                if self.global_step >=  self.warmup_steps:
                    self.scheduler.step()

        return original_loss

    @torch.no_grad()
    def evaluate(self, test=False):
        self.model.eval()
        dataloader = self.test_loader if test else self.val_loader
        phase = 'test' if test else 'val'

        references, hypotheses = [], []
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Evaluation"):
            images = batch['image'].to(self.device)
            reports = batch['report']

            loss = self.model(images, reports)
            total_loss += loss.item()

            preds = self.model.generate(images, sample=False, num_beams=self.config['num_beams'])

            references.extend(reports)
            hypotheses.extend(preds)

            num_batches += 1

        avg_loss = total_loss / num_batches
        metrics = self._compute_nlp_metrics(references, hypotheses)
        metrics.update({
            f"{phase}_loss": avg_loss,
        })

        sample_pred = hypotheses[0]
        sample_ref = references[0]

        self.logger.info(f"Sample prediction: {sample_pred}")
        self.logger.info(f"Sample reference: {sample_ref}")

        self.logger.info(f"{phase.capitalize()} metrics: {json.dumps(metrics, indent=2)}")
        self.model.train()
        return metrics

    @staticmethod
    def _compute_nlp_metrics(references, hypotheses):
        eval_metrics = {}

        bleu_score = bleu.compute(predictions=hypotheses, references=references)
        rouge_score = rouge.compute(predictions=hypotheses, references=[r[0] for r in references])
        # cider_score = cider.compute(predictions=hypotheses, references=[r[0] for r in references])

        eval_metrics.update({
            "bleu": bleu_score,
            "rouge": rouge_score,
            # "cider": cider_score
        })
        return eval_metrics

    def save_checkpoint(self, step=None, is_best=False):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss
        }

        if step is not None:
            checkpoint_path = os.path.join(self.output_dir, f'checkpoint-{step}.pt')
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        if is_best:
            best_path = os.path.join(self.output_dir, 'blip_cxr_pretrain.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")