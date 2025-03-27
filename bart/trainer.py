import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class VisionTextTrainer:
    """
    Trainer class for end-to-end Vision-Text model
    """

    def __init__(
            self,
            model,
            train_dataloader,
            val_dataloader,
            tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            output_dir="./checkpoints",
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=1000,
            max_grad_norm=1.0,
            num_epochs=10,
            logging_steps=100,
            save_steps=1000,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.save_steps = save_steps

        self.model.to(self.device)

        os.makedirs(output_dir, exist_ok=True)

    def train(self):
        # prepare optimizer and scheduler
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        total_steps = len(self.train_dataloader) * self.num_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

            for batch in progress_bar:
                images = batch["images"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # forward pass
                outputs = self.model(
                    image=images,
                    decoder_input_ids=input_ids[:, :-1],  # remove EOS for input
                    decoder_attention_mask=attention_mask[:, :-1],
                )

                # calculate loss
                # shift labels (teacher forcing)
                shifted_labels = labels[:, 1:]  # remove BOS for target

                # reshape logits and labels for loss calculation
                logits_flat = outputs.view(-1, outputs.size(-1))
                labels_flat = shifted_labels.view(-1)

                loss = loss_fn(logits_flat, labels_flat)

                # backward pass
                loss.backward()

                # CLIP gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (global_step % len(self.train_dataloader) + 1)})

                global_step += 1
                if global_step % self.logging_steps == 0:
                    print(f"Epoch: {epoch + 1}/{self.num_epochs}, Step: {global_step}, Loss: {loss.item()}")

                # save checkpoint
                if global_step % self.save_steps == 0:
                    self.save_model(os.path.join(self.output_dir, f"checkpoint-{global_step}"))

            # validation
            val_loss, val_metrics = self.evaluate()
            print(f"Validation Loss: {val_loss}, Metrics: {val_metrics}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(self.output_dir, "best_model"))

            self.save_model(os.path.join(self.output_dir, f"epoch-{epoch + 1}"))

        self.save_model(os.path.join(self.output_dir, "final_model"))

        return global_step, epoch_loss / len(self.train_dataloader)

    def evaluate(self):
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                images = batch["images"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    image=images,
                    decoder_input_ids=input_ids[:, :-1],
                    decoder_attention_mask=attention_mask[:, :-1],
                )

                shifted_labels = labels[:, 1:]
                logits_flat = outputs.view(-1, outputs.size(-1))
                labels_flat = shifted_labels.view(-1)

                loss = loss_fn(logits_flat, labels_flat)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=-1)

                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_labels.extend(shifted_labels.detach().cpu().numpy().tolist())

        # calculate metrics
        # TODO: improve metrics
        metrics = {
            "accuracy": accuracy_score(
                [l for l in all_labels if l != self.tokenizer.pad_token_id],
                [p for p, l in zip(all_preds, all_labels) if l != self.tokenizer.pad_token_id]
            ),
        }

        return val_loss / len(self.val_dataloader), metrics

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        self.tokenizer.save_pretrained(path)
        config = {
            "hidden_size": self.model.config.hidden_size,
            "vocab_size": self.model.config.vocab_size,
            "decoder_layers": self.model.config.decoder_layers,
            "decoder_attention_heads": self.model.config.decoder_attention_heads,
            "max_position_embeddings": self.model.config.max_position_embeddings,
        }

        import json
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    def load_model(self, path):
        # load model from checkpoint
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        return self.model