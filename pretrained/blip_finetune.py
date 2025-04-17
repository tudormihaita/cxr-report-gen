import os
import torch
import logging

from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from torch.cuda.amp import GradScaler, autocast
from transformers import BlipProcessor, BlipForConditionalGeneration


class BLIPTrainer:
    def __init__(
        self,
        model: BlipForConditionalGeneration,
        processor: BlipProcessor,
        train_loader,
        val_loader=None,
        test_loader=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir='./blip_output',
        learning_rate=5e-5,
        warmup_steps=1000,
        decay_steps=9000,
        max_steps=30000,
        max_epochs=None,
        log_interval=100,
        save_interval=2000,
        gradient_accumulation_steps=1,
        mixed_precision=True,
        max_grad_norm=1.0,
    ):
        self.model = model.to(device)
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        steps_per_epoch = len(train_loader)
        if max_epochs is not None:
            self.max_epochs = max_epochs
            self.max_steps = steps_per_epoch * max_epochs
        else:
            self.max_steps = max_steps
            self.max_epochs = (max_steps + steps_per_epoch - 1) // steps_per_epoch

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm

        self.logger = self._setup_logger()
        self.scaler = GradScaler() if mixed_precision else None
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        decay = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=decay_steps)
        self.scheduler = SequentialLR(self.optimizer, [warmup, decay], milestones=[warmup_steps])

    @staticmethod
    def _setup_logger():
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        return logger

    def train(self):
        self.logger.info("Starting training")
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}")
            epoch_loss = 0.0
            self.model.train()

            for batch in tqdm(self.train_loader, desc="Training"):
                if self.global_step >= self.max_steps:
                    self.logger.info("Max steps reached, stopping.")
                    return

                loss = self.train_step(batch)
                epoch_loss += loss

                if (self.global_step + 1) % self.log_interval == 0:
                    self.logger.info(f"Step {self.global_step + 1}, Loss: {loss:.4f}")

                if (self.global_step + 1) % self.save_interval == 0:
                    self.save_checkpoint(self.global_step + 1)
                    if self.val_loader:
                        val_loss = self.evaluate()
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(is_best=True)

                self.global_step += 1

            if self.val_loader:
                val_loss = self.evaluate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)

    def train_step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        pixel_values = batch['pixel_values'].to(self.device)

        self.optimizer.zero_grad()

        if self.mixed_precision:
            with autocast():
                outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
                loss = outputs.loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
        else:
            outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

        return loss.item() * self.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.logger.info("Evaluating...")
        total_loss = 0.0

        for batch in tqdm(self.val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(self.device)
            pixel_values = batch['pixel_values'].to(self.device)

            outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, step=None, is_best=False):
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.global_step,
            'epoch': self.epoch
        }

        if step:
            path = os.path.join(self.output_dir, f'checkpoint-{step}.pt')
            torch.save(ckpt, path)
            self.logger.info(f"Saved checkpoint at {path}")

        if is_best:
            best_path = os.path.join(self.output_dir, 'blip_best_model.pt')
            torch.save(ckpt, best_path)
            self.logger.info(f"Saved best model at {best_path}")