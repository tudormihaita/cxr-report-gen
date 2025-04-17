import os
import time
import json
import torch
import logging
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR


class BLIPTrainer:
    def __init__(self,
                 model,
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 num_epochs=None,
                 max_steps=30000,
                 warmup_steps=300,
                 log_interval=300,
                 save_interval=1000,
                 learning_rate=1e-5,
                 output_dir='./output',
                 gradient_accumulation_steps=1,
                 weight_decay=0.2,
                 max_grad_norm=1.0,
                 mixed_precision=True,
                 ):
        self.device = device
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        steps_per_epoch = len(train_loader)
        if num_epochs is None:
            self.max_steps = max_steps
            self.num_epochs = (max_steps + steps_per_epoch - warmup_steps) // warmup_steps
        else:
            self.max_steps = steps_per_epoch * num_epochs
            self.num_epochs = num_epochs

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_steps)
        self.scaler = GradScaler('cuda') if torch.cuda.is_available() and mixed_precision else None

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        self.logger = self.__setup_logger(__name__)
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

    def train(self):
        self.model.train()
        self.logger.info("Start training")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")

            epoch_loss = 0.0
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Train")):
                if self.global_step >= self.max_steps:
                    self.logger.info(f"Reached maximum steps {self.max_steps}. Stopping training.")
                    break

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
        reports = batch['report'].to(self.device)

        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        if self.scaler is not None:
            with autocast('cuda'):
                loss_ita, loss_itm, loss_lm = self.model(images, reports, alpha=0.4)

                loss = loss_ita + loss_itm + loss_lm
                original_loss = loss.clone()
                loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if self.global_step % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
        else:
            loss_ita, loss_itm, loss_lm = self.model(images, reports, alpha=0.4)

            loss = loss_ita + loss_itm + loss_lm
            original_loss = loss.clone()
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
        phase = "test" if test else "val"

        total_loss = 0.0
        image_feats, image_embeds = [], []
        text_ids, text_embeds, text_atts = [], [], []

        for batch in tqdm(dataloader, desc=f"Evaluation"):
            images = batch['image'].to(self.device)
            reports = batch['report'].to(self.device)

            loss_ita, loss_itm, loss_lm = self.model(images, reports, alpha=0.4)
            total_loss += (loss_ita + loss_itm + loss_lm).item()

            text_input = self.model.tokenizer(reports, padding='max_length', truncation=True, max_length=256, return_tensors='pt').to(self.device)
            text_output = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
            text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]))

            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

            image_feat = self.model.visual_encoder(images)
            image_embed = self.model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feats.append(image_feat)
            image_feats.append(image_embed)

        text_embeds = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_atts = torch.cat(text_atts, dim=0)
        text_ids[:, 0] = self.model.tokenizer.enc_token_id

        image_feats = torch.cat(image_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        sims_matrix = image_embeds @ text_embeds.t()
        score_matrix_i2t = torch.full((len(image_embeds), len(text_embeds)), -100.0).to(self.device)

        for i, sims in enumerate(sims_matrix):
            topk_sim, topk_idx = sims.topk(k=5, dim=0)

            encoder_output = image_feats[i].repeat(5, 1, 1).to(self.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)

            output = self.model.text_encoder(
                text_ids[topk_idx],
                attention_mask=text_atts[topk_idx],
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True
            )
            score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_i2t[i, topk_idx] = score + topk_sim

        sims_matrix_t = sims_matrix.t()
        score_matrix_t2i = torch.full((len(text_embeds), len(image_embeds)), -100.0).to(self.device)

        for i, sims in enumerate(sims_matrix_t):
            topk_sim, topk_idx = sims.topk(k=5, dim=0)

            encoder_output = image_feats[topk_idx].to(self.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)

            output = self.model.text_encoder(
                text_ids[i].repeat(5, 1),
                attention_mask=text_atts[i].repeat(5, 1),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[i, topk_idx] = score + topk_sim

        avg_loss = total_loss / len(dataloader)
        eval_metrics = {
            f"{phase}_loss": avg_loss,
            f"{phase}_i2t_score": score_matrix_i2t,
            f"{phase}_t2i_score": score_matrix_t2i
        }

        self.logger.info(f"{phase.capitalize()} loss: {avg_loss:.4f}")

        self.model.train()
        return eval_metrics



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
            best_path = os.path.join(self.output_dir, 'blip_cxr_pretrain_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")


