import os
import time
import json
import math
import torch
import logging
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    lr = max(min_lr, init_lr * (decay_rate ** epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class BLIPTrainer:
    def __init__(self,
                 model,
                 config,
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 output_dir='./output/pretrain',
                 mixed_precision=True,
                 ):
        self.device = device
        self.config = config
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.alpha = config['alpha']
        self.init_lr, self.min_lr, self.warmup_lr = float(config['init_lr']), float(config['min_lr']), float(
            config['warmup_lr'])
        self.lr_decay_rate = config['lr_decay_rate']
        self.weight_decay = config['weight_decay']
        self.warmup_steps = config['warmup_steps']
        self.gradient_accumulation_steps = config['gradient_accumulation_steps']
        self.max_grad_norm = config['max_grad_norm']

        self.log_interval = config['log_interval']
        self.save_interval = config['save_interval']
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        steps_per_epoch = len(train_loader)
        if config['max_epochs'] is None:
            self.max_steps = config['max_steps']
            self.num_epochs = (config['max_steps'] + steps_per_epoch - config['warmup_steps']) // steps_per_epoch
        else:
            self.max_steps = steps_per_epoch * config['max_epochs']
            self.num_epochs = config['max_epochs']

        self.optimizer = AdamW(self.model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_steps)
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
            step_lr_schedule(self.optimizer, epoch, self.init_lr, self.min_lr, self.lr_decay_rate)

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
        reports = batch['report']

        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()
        if self.global_step < self.warmup_steps:
            warmup_lr_schedule(self.optimizer, self.global_step, self.warmup_steps, self.warmup_lr, self.init_lr)

        if self.scaler is not None:
            with autocast('cuda'):
                loss_ita, loss_itm, loss_lm = self.model(images, reports, alpha=self.alpha)

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
                # self.scheduler.step()
        else:
            loss_ita, loss_itm, loss_lm = self.model(images, reports, alpha=self.alpha)

            loss = loss_ita + loss_itm + loss_lm
            original_loss = loss.clone()
            loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                # self.scheduler.step()

        return original_loss

    @torch.no_grad()
    def evaluate_step(self):
        self.model.eval()
        total_loss = 0.0
        image_embeds, text_embeds = [], []
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            reports = batch['report']

            loss_ita, loss_itm, loss_lm = self.model(images, reports, alpha=self.alpha)
            total_loss += (loss_ita + loss_itm + loss_lm).item()

            text_input = self.model.tokenizer(reports, padding='max_length', truncation=True, max_length=248,
                                              return_tensors='pt').to(self.device)
            text_output = self.model.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                mode='text'
            )
            text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]))
            text_embeds.append(text_embed.detach().cpu())

            image_feat = self.model.visual_encoder(images)
            image_embed = self.model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)
            image_embeds.append(image_embed.detach().cpu())

        text_embeds = torch.cat(text_embeds, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        similarity = image_embeds @ text_embeds.t()
        similarity = similarity.numpy()
        ranks = np.array([
            np.where(np.argsort(sim)[::-1] == i)[0][0] for i, sim in enumerate(similarity)
        ])

        avg_loss = total_loss / len(self.val_loader)
        r1 = 100.0 * np.mean(ranks < 1)
        r5 = 100.0 * np.mean(ranks < 5)
        r10 = 100.0 * np.mean(ranks < 10)
        median_rank = np.median(ranks) + 1

        eval_metrics = {
            'val_loss': avg_loss,
            'val_r1': r1,
            'val_r5': r5,
            'val_r10': r10,
            'val_median_rank': median_rank,
        }

        self.logger.info(f"Val metrics: {json.dumps(eval_metrics, indent=2)}")

        self.model.train()
        return eval_metrics

    @torch.no_grad()
    def evaluate(self, test=False, k_candidates=128):
        if not test:
            self.logger.info("Evaluating retrieval metrics on val set")
            return self.evaluate_step()

        self.model.eval()
        dataloader = self.test_loader if test else self.val_loader
        phase = "test" if test else "val"

        total_loss = 0.0
        image_feats, image_embeds = [], []
        text_ids, text_embeds, text_atts = [], [], []

        self.logger.info(f"Evaluating all performance metrics on {phase} set")
        for batch in tqdm(dataloader, desc=f"Evaluation"):
            images = batch['image'].to(self.device)
            reports = batch['report']

            loss_ita, loss_itm, loss_lm = self.model(images, reports, alpha=self.alpha)
            total_loss += (loss_ita + loss_itm + loss_lm).item()

            text_input = self.model.tokenizer(reports, padding='max_length', truncation=True, max_length=248,
                                              return_tensors='pt').to(self.device)
            text_output = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                                  mode='text')
            text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]))

            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

            image_feat = self.model.visual_encoder(images)
            image_embed = self.model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feats.append(image_feat)
            image_embeds.append(image_embed)

        text_embeds = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_atts = torch.cat(text_atts, dim=0)
        text_ids[:, 0] = self.model.tokenizer.enc_token_id

        image_feats = torch.cat(image_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        sims_matrix = image_embeds @ text_embeds.t()
        score_matrix_i2t = torch.full((len(image_embeds), len(text_embeds)), -100.0)

        for i, sims in enumerate(sims_matrix):
            topk_sim, topk_idx = sims.topk(k=k_candidates, dim=0)

            encoder_output = image_feats[i].repeat(k_candidates, 1, 1)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long)

            output = self.model.text_encoder(
                text_ids[topk_idx].to(self.device),
                attention_mask=text_atts[topk_idx].to(self.device),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True
            )
            score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_i2t[i, topk_idx] = score + topk_sim

        sims_matrix_t = sims_matrix.t()
        score_matrix_t2i = torch.full((len(text_embeds), len(image_embeds)), -100.0)

        for i, sims in enumerate(sims_matrix_t):
            topk_sim, topk_idx = sims.topk(k=k_candidates, dim=0)

            encoder_output = image_feats[topk_idx].to(self.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)

            output = self.model.text_encoder(
                text_ids[i].repeat(k_candidates, 1).to(self.device),
                attention_mask=text_atts[i].repeat(k_candidates, 1).to(self.device),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[i, topk_idx] = score + topk_sim

        avg_loss = total_loss / len(dataloader)

        txt2img = {i: i for i in range(len(text_embeds))}
        img2txt = {i: [i] for i in range(len(score_matrix_i2t))}

        ranks_i2t = np.zeros(score_matrix_i2t.shape[0])
        for index, score in enumerate(score_matrix_i2t):
            inds = torch.argsort(score, descending=True)
            gt = img2txt[index]
            rank = min([torch.where(inds == g)[0].item() for g in gt])
            ranks_i2t[index] = rank

        tr1 = 100.0 * np.mean(ranks_i2t < 1)
        tr5 = 100.0 * np.mean(ranks_i2t < 5)
        tr10 = 100.0 * np.mean(ranks_i2t < 10)

        ranks_t2i = np.zeros(score_matrix_t2i.shape[0])
        for index, score in enumerate(score_matrix_t2i):
            inds = torch.argsort(score, descending=True)
            ranks_t2i[index] = torch.where(inds == txt2img[index])[0].item()

        ir1 = 100.0 * np.mean(ranks_t2i < 1)
        ir5 = 100.0 * np.mean(ranks_t2i < 5)
        ir10 = 100.0 * np.mean(ranks_t2i < 10)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        eval_metrics = {
            f"{phase}_loss": avg_loss,
            f"{phase}_txt_r1": tr1,
            f"{phase}_txt_r5": tr5,
            f"{phase}_txt_r10": tr10,
            f"{phase}_img_r1": ir1,
            f"{phase}_img_r5": ir5,
            f"{phase}_img_r10": ir10,
            f"{phase}_txt_r_mean": tr_mean,
            f"{phase}_img_r_mean": ir_mean,
            f"{phase}_r_mean": r_mean,
        }

        self.logger.info(f"{phase.capitalize()} metrics: {json.dumps(eval_metrics, indent=2)}")

        self.model.train()
        return eval_metrics

    def save_checkpoint(self, step=None, is_best=False):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
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
