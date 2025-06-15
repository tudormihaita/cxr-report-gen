import torch
import torch.optim as optim

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AutoTokenizer


class VisionBartTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=3e-5,
        weight_decay=0.01,
        num_epochs=5,
        warmup_ratio=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_position_embeddings=128,
        max_grad_norm=1.0,
        model_name="facebook/bart-base",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.device = device
        self.max_position_embeddings = max_position_embeddings
        self.max_grad_norm = max_grad_norm

        self.model = self.model.to(self.device)
        self.model.train()

        total_steps = len(self.train_loader) * self.num_epochs

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self):
        global_step = 0

        for epoch in range(self.num_epochs):
            print(f"\nEpoch: {epoch+1}/{self.num_epochs}")
            total_loss = 0.0

            self.model.train()
            for step, batch in enumerate(tqdm(self.train_loader)):

                pixel_values = batch["image"].to(self.device)
                textual_reports = list(batch["report"])  # raw text

                if max(len(self.tokenizer.encode(text)) for text in textual_reports) > self.max_position_embeddings:
                    print(f"Warning: Some reports exceed {self.max_position_embeddings} tokens and will be truncated")

                encoded_text = self.tokenizer(
                    textual_reports,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_position_embeddings,
                    return_tensors="pt"
                ).to(self.device)

                decoder_input_ids = encoded_text["input_ids"]
                decoder_attention_mask = encoded_text["attention_mask"]

                # forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=decoder_input_ids
                )
                loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                global_step += 1

                if (step + 1) % 50 == 0:
                    avg_loss = total_loss / (step + 1)
                    print(f" Step [{step+1}/{len(self.train_loader)}], Loss: {avg_loss:.4f}")

            epoch_loss = total_loss / (len(self.train_loader) if len(self.train_loader) > 0 else 1)
            print(f"Epoch {epoch+1} finished. Average Training Loss = {epoch_loss:.4f}")

            if self.val_loader is not None:
                self.evaluate()

        print("Training complete!")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_val_loss = 0.0
        steps = 0

        for batch in self.val_loader:
            if batch is None:
                continue

            pixel_values = batch["image"].to(self.device)
            textual_reports = list(batch["report"])

            encoded_text = self.tokenizer(
                textual_reports,
                padding=True,
                truncation=True,
                max_length=self.max_position_embeddings,
                return_tensors="pt"
            ).to(self.device)

            input_ids = encoded_text["input_ids"]
            attention_mask = encoded_text["attention_mask"]

            outputs = self.model(
                pixel_values=pixel_values,
                labels=input_ids
            )
            total_val_loss += outputs.loss.item()
            steps += 1

        avg_val_loss = total_val_loss / steps if steps > 0 else 0.0
        print(f"[Validation] Average Loss = {avg_val_loss:.4f}")
        self.model.train()