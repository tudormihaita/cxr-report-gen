import os
import time
import json
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from utils.logger import LoggerManager
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Linear warmup of learning rate"""
    lr = init_lr + (max_lr - init_lr) * step / max_step
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class BLIPRetrievalTrainer:
    def __init__(self,
                 model,
                 config,
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 output_dir='./output/retrieval',
                 mixed_precision=True,
                 ):
        self.device = device
        self.config = config
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.batch_size = config['batch_size']
        self.max_length = config['max_length']
        self.alpha = config['alpha']

        self.init_lr = float(config.get('init_lr', 1e-5))
        self.min_lr = float(config.get('min_lr', 1e-6))
        self.warmup_lr = float(config.get('warmup_lr', 1e-7))
        self.lr_decay_rate = config.get('lr_decay_rate', 0.9)

        self.weight_decay = config.get('weight_decay', 0.01)
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.grad_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        self.log_interval = config['log_interval']
        self.save_interval = config['save_interval']
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        steps_per_epoch = len(train_loader)
        if config['max_epochs'] is None:
            self.max_steps = config['max_steps']
            self.num_epochs = (config['max_steps'] + steps_per_epoch - 1) // steps_per_epoch
        else:
            self.max_steps = steps_per_epoch * config['max_epochs']
            self.num_epochs = config['max_epochs']

        self.optimizer = AdamW(
            [
                {'params': [p for n, p in model.named_parameters() if 'visual_encoder' in n], 'lr': self.init_lr * 0.1},
                {'params': [p for n, p in model.named_parameters() if 'visual_encoder' not in n]}
            ],
            lr=self.init_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
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
        self.best_val_r_mean = 0.0

        self.logger = LoggerManager.get_logger(__name__)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")

    def train(self):
        self.model.train()
        self.logger.info("Start training")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")

            epoch_loss = 0.0
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Train")):
                if (self.global_step + 1) >= self.max_steps:
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
            avg_loss = epoch_loss / len(self.train_loader)
            self.logger.info(
                f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. Average loss: {avg_loss:.4f}")

            val_metrics = self.evaluate()
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(is_best=True, suffix='best_loss')

            if val_metrics['r_mean'] > self.best_val_r_mean:
                self.best_val_r_mean = val_metrics['r_mean']
                self.save_checkpoint(is_best=True, suffix='best_retrieval')

        if self.test_loader is not None:
            self.logger.info("Running final evaluation on test set")
            test_metrics = self.evaluate(test=True)
            self.logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")

        self.logger.info("Training completed")

    def train_step(self, batch):
        images = batch['image'].to(self.device)
        captions = batch['report']
        idxs = batch['idx']

        if self.global_step < self.warmup_steps:
            warmup_lr_schedule(self.optimizer, self.global_step, self.warmup_steps, self.warmup_lr, self.init_lr)

        if self.scaler is not None:
            with autocast('cuda'):
                loss_ita, loss_itm = self.model(images, captions, idxs, alpha=self.alpha)

                loss = 1.0 * loss_ita + 1.0 * loss_itm
                scaled_loss = loss / self.grad_accumulation_steps

            self.scaler.scale(scaled_loss).backward()

            if (self.global_step + 1) % self.grad_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.global_step >=  self.warmup_steps:
                    self.scheduler.step()
        else:
            loss_ita, loss_itm = self.model(images, captions, idxs, alpha=self.alpha)

            loss = 1.0 * loss_ita + 1.0 * loss_itm
            scaled_loss = loss / self.grad_accumulation_steps

            scaled_loss.backward()

            if (self.global_step + 1) % self.grad_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                if self.global_step >=  self.warmup_steps:
                    self.scheduler.step()

        return loss

    @torch.no_grad()
    def evaluate(self, test=False):
        self.model.eval()
        dataloader = self.test_loader if test else self.val_loader
        phase = "test" if test else "val"

        total_loss, ita_loss, itm_loss = 0.0, 0.0, 0.0
        uids = []
        image_embeds, text_embeds = [], []

        for batch in tqdm(dataloader, desc="Evaluation"):
            images = batch['image'].to(self.device)
            captions = batch['caption']
            idxs = batch['idx']
            uids.extend(batch['uid'])

            loss_ita, loss_itm = self.model(images, captions, idxs, alpha=self.alpha)
            total_loss += (1.0 * loss_ita + 1.0 * loss_itm).item()
            ita_loss += 1.0 * loss_ita.item()
            itm_loss += 1.0 * loss_itm.item()

            text_input = self.model.tokenizer(captions, padding='max_length', truncation=True,
                                              max_length=self.max_length,
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

        uids = np.array(uids)
        text_embeds = torch.cat(text_embeds, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        similarity = image_embeds @ text_embeds.t()
        similarity = similarity.numpy()

        img2txt = dataloader.dataset.img2txt
        retrieval_metrics = self.compute_retrieval_metrics(similarity, img2txt, uids)

        avg_loss = total_loss / len(dataloader)
        eval_metrics = {
            f"{phase}_loss": avg_loss,
            f"{phase}_ita_loss": ita_loss / len(dataloader),
            f"{phase}_itm_loss": itm_loss / len(dataloader),
        }
        eval_metrics.update(retrieval_metrics)

        self.logger.info(f"{phase.capitalize()} metrics: {json.dumps(eval_metrics, indent=2)}")

        self.model.train()
        return eval_metrics

    @staticmethod
    def compute_retrieval_metrics(similarity_matrix, img2txt, uids):
        """
        Compute retrieval metrics using the img2txt mapping
        which groups images with the same caption (medical findings).
        :param similarity_matrix: numpy array of similarity scores (images x texts)
        :param img2txt: dict mapping image uid to list of image uids with same caption
        :param uids: list of image unique identifiers
        :return: dict of retrieval metrics
        """
        uid_to_idx = {uid: idx for idx, uid in enumerate(uids)}
        # Image -> Text retrieval
        i2t_ranks = np.zeros(similarity_matrix.shape[0])
        for idx, scores in enumerate(similarity_matrix):
            query_uid = uids[idx]
            inds = np.argsort(scores)[::-1]  # sort indices by descending similarity

            # get indices of valid matches (images with same caption as query)
            valid_uids = [uid for uid in img2txt.get(query_uid, []) if uid != query_uid]
            valid_indices = [uid_to_idx[uid] for uid in valid_uids if uid in uid_to_idx]

            # find the highest rank of any valid match
            if valid_indices:
                # get the positions of all valid matches in the sorted list
                positions = [np.where(inds == i)[0][0] for i in valid_indices]
                i2t_ranks[idx] = min(positions)  # take the best rank
            else:
                i2t_ranks[idx] = similarity_matrix.shape[1]  # no matches found

        # compute image-to-text metrics
        tr1 = 100.0 * len(np.where(i2t_ranks < 1)[0]) / len(i2t_ranks)
        tr5 = 100.0 * len(np.where(i2t_ranks < 5)[0]) / len(i2t_ranks)
        tr10 = 100.0 * len(np.where(i2t_ranks < 10)[0]) / len(i2t_ranks)

        # Text -> Image retrieval (using transposed similarity matrix)
        t2i_matrix = similarity_matrix.T
        t2i_ranks = np.zeros(t2i_matrix.shape[0])

        for idx, scores in enumerate(t2i_matrix):
            query_uid = uids[idx]
            inds = np.argsort(scores)[::-1]

            valid_uids = [uid for uid in img2txt.get(query_uid, []) if uid != query_uid]
            valid_indices = [uid_to_idx[uid] for uid in valid_uids if uid in uid_to_idx]

            if valid_indices:
                positions = [np.where(inds == i)[0][0] for i in valid_indices]
                t2i_ranks[idx] = min(positions)
            else:
                t2i_ranks[idx] = t2i_matrix.shape[1]

        # compute text-to-image metrics
        ir1 = 100.0 * len(np.where(t2i_ranks < 1)[0]) / len(t2i_ranks)
        ir5 = 100.0 * len(np.where(t2i_ranks < 5)[0]) / len(t2i_ranks)
        ir10 = 100.0 * len(np.where(t2i_ranks < 10)[0]) / len(t2i_ranks)

        # compute mean metrics
        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        return {
            'i2t_recall@1': tr1,
            'i2t_recall@5': tr5,
            'i2t_recall@10': tr10,
            'i2t_r_mean': tr_mean,
            't2i_recall@1': ir1,
            't2i_recall@5': ir5,
            't2i_recall@10': ir10,
            't2i_r_mean': ir_mean,
            'r_mean': r_mean
        }

    def save_checkpoint(self, step=None, is_best=False, suffix=None):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_r_mean': self.best_val_r_mean
        }

        if step is not None:
            checkpoint_path = os.path.join(self.output_dir, f'checkpoint-{step}.pt')
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        if is_best:
            if suffix:
                best_path = os.path.join(self.output_dir, f'blip_cxr_retrieval_{suffix}.pt')
            else:
                best_path = os.path.join(self.output_dir, 'blip_cxr_retrieval.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
