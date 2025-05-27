import os
import torch
import torch.nn as nn

from typing import Dict, TypeVar
from transformers.modeling_outputs import BaseModelOutput

from constants import CHEXPERT_LABELS
from utils.logger import LoggerManager
from .modules import load_image_encoder, load_text_decoder

T = TypeVar("T", bound="Module")
log = LoggerManager.get_logger(__name__)


class CxrReportDecoder(nn.Module):
    def __init__(
            self,
            model_config: Dict,
            tokenizer,
            use_custom: bool = False,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_input_length = model_config["text_decoder"]["max_input_length"]
        self.max_output_length = model_config["text_decoder"]["max_output_length"]

        self.prompt_template = model_config["text_decoder"]["prompt_template"]
        self.instruction_prefix = model_config["text_decoder"].get("instruction_prefix", "")
        self.beam_size = model_config["text_decoder"]["beam_size"]

        if model_config["load_backbone_weights"] is None:
            self.image_encoder = load_image_encoder(model_config["image_encoder"])
        else:
            log.info("Loading pre-trained image encoder from checkpoint")
            if not os.path.isfile(model_config["load_backbone_weights"]):
                raise ValueError(f"Cannot find a weight file: {model_config['load_backbone_weights']}")
            ckpt = torch.load(model_config["load_backbone_weights"], map_location="cpu", weights_only=False)
            # TODO: fix with using only model_state_dict after correcting the training script
            model_key = "model_state_dict" if use_custom else "model_state_dict"
            self.image_encoder = load_image_encoder(model_config["image_encoder"])
            image_encoder_weights = {}
            for k in ckpt[model_key].keys():
                if k.startswith("image_encoder."):
                    image_encoder_weights[".".join(k.split(".")[1:])] = ckpt[model_key][k]
            self.image_encoder.load_state_dict(image_encoder_weights, strict=True)

        if model_config["freeze_backbone_weights"]:
            log.info("Freezing image encoder to not be re-trained")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.text_decoder = load_text_decoder(model_config["text_decoder"], len(tokenizer), mode="conditional")

        encoder_hidden_size = model_config["image_encoder"]["hidden_size"]
        decoder_hidden_size = self.text_decoder.config.d_model

        self.image_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.image_pos_embedding = nn.Parameter(torch.randn(1, 1, decoder_hidden_size) * 0.02)


    def train(self: T, mode: bool = True) -> T:
        if not isinstance(mode, bool):
            raise ValueError("Training mode is expected to be boolean")

        if mode:
            self.image_encoder.eval()
            self.text_decoder.train()
        else:
            self.image_encoder.eval()
            self.text_decoder.eval()

        return self

    def encode_image(self, image):
        image_features = self.image_encoder(image)
        # use [CLS] token as global image descriptor
        global_features = image_features[:, 0]
        return global_features

    def _prepare_encoder_inputs(self, concept_labels):
        prompts = []
        for labels in concept_labels:
            findings = [CHEXPERT_LABELS[i] for i in torch.where(labels == 1)[0].cpu().numpy()]
            findings_caption = ', '.join(findings) if findings else "No Finding"
            prompt = self.prompt_template.format(findings_caption)
            prompts.append(prompt)

        encoder_texts = [f"{self.instruction_prefix}{prompt}" for prompt in prompts]
        tokenized_texts = self.tokenizer(
            encoder_texts,
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        return tokenized_texts

    def _prepare_decoder_inputs(self, texts, is_train=False):
        if not is_train:
            return None

        tokenized_texts = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_output_length,
            return_tensors="pt"
        )
        labels = tokenized_texts["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        decoder_start_token_id = getattr(self.text_decoder.config, 'decoder_start_token_id',
                                         self.tokenizer.bos_token_id)
        if decoder_start_token_id is None:
            # Fallback to pad_token_id or 0 if no start token is defined
            decoder_start_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        decoder_input_ids = torch.zeros_like(labels)
        decoder_input_ids[:, 1:] = labels[:, :-1].clone()
        decoder_input_ids[:, 0] = decoder_start_token_id

        vocab_size = len(self.tokenizer)
        decoder_input_ids = torch.clamp(decoder_input_ids, 0, vocab_size - 1)

        return {
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": tokenized_texts["attention_mask"],
            "labels": labels,
        }

    def _prepare_decoder_forward(self, encoder_inputs, image_embeds, device):
        encoder_outputs = self.text_decoder.model.encoder(
            input_ids=encoder_inputs["input_ids"],
            attention_mask=encoder_inputs["attention_mask"],
        )

        image_features = self.image_proj(image_embeds)
        image_features_expanded = image_features.unsqueeze(1) + self.image_pos_embedding

        combined_encoder_outputs = torch.cat([
            image_features_expanded,
            encoder_outputs.last_hidden_state,
        ], dim=1)

        batch_size = encoder_inputs["attention_mask"].size(0)
        image_attention_mask = torch.ones(batch_size, 1, device=device, dtype=encoder_inputs["attention_mask"].dtype)
        combined_attention_mask = torch.cat([
            image_attention_mask,
            encoder_inputs["attention_mask"]
        ], dim=1)

        return combined_encoder_outputs, combined_attention_mask, encoder_outputs

    def forward(self, batch, device=None):
        device = batch["images"].device if device is None else device

        encoder_inputs = self._prepare_encoder_inputs(batch["labels"])
        encoder_inputs = {k: v.to(device) for k, v in encoder_inputs.items()}

        decoder_inputs = self._prepare_decoder_inputs(texts=batch["texts"], is_train=True,)
        decoder_inputs = {k: v.to(device) for k, v in decoder_inputs.items()}

        image_embeds = self.encode_image(batch["images"].to(device))
        combined_encoder_outputs, combined_attention_mask, encoder_outputs = self._prepare_decoder_forward(
            encoder_inputs=encoder_inputs,
            image_embeds=image_embeds,
            device=device
        )

        outputs = self.text_decoder(
            encoder_outputs=BaseModelOutput(last_hidden_state=combined_encoder_outputs),
            attention_mask=combined_attention_mask,
            decoder_input_ids=decoder_inputs["decoder_input_ids"],
            decoder_attention_mask=decoder_inputs["decoder_attention_mask"],
            labels=decoder_inputs["labels"],
        )

        return {
            "logits": outputs.logits,
            "labels": decoder_inputs["labels"],
            "loss": outputs.loss,
        }

    def generate(self, image_embeds, findings=None,
                 num_beams=None, top_p=0.9, temperature=1.0, repetition_penalty=1.0,
                 sample=False, device=None
                 ):
        if findings is None:
            raise NotImplementedError("Prompt construction is not implemented yet. Please provide findings.")
        if num_beams is None:
            num_beams = self.beam_size

        self.text_decoder.eval()
        with torch.no_grad():
            encoder_inputs = self._prepare_encoder_inputs(concept_labels=findings)
            encoder_inputs = {k: v.to(device) for k, v in encoder_inputs.items()}

            combined_encoder_outputs, combined_attention_mask, encoder_outputs = self._prepare_decoder_forward(
                encoder_inputs=encoder_inputs,
                image_embeds=image_embeds.to(device),
                device=device
            )

            generation_kwargs = {
                "max_length": self.max_output_length,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "repetition_penalty": repetition_penalty,
            }

            if sample:
                generation_kwargs.update({
                    "do_sample": True,
                    "top_p": top_p,
                    "temperature": temperature,
                    "num_return_sequences": 1,
                })
            else:
                generation_kwargs.update({
                    "num_beams": num_beams,
                    "early_stopping": True,
                })

            outputs = self.text_decoder.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=combined_encoder_outputs),
                attention_mask=combined_attention_mask,
                **generation_kwargs
            )

            generated_texts = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            generated_texts = [text.strip() for text in generated_texts]
        return generated_texts
