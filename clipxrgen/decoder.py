import os
import torch
import torch.nn as nn
from typing import Dict, TypeVar

from constants import CHEXPERT_LABELS
from utils.logger import LoggerManager
from .modules import load_text_decoder, load_image_encoder, \
    load_pretrained_image_encoder_weights, load_hf_checkpoint

T = TypeVar("T", bound="Module")
log = LoggerManager.get_logger(__name__)

class CLIPXRGen(nn.Module):
    def __init__(
            self,
            model_config: Dict,
            tokenizer,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self._setup_tokenizer()
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids("<sep>")
        self.findings_token_id = self.tokenizer.convert_tokens_to_ids("<findings>")

        self.beam_size = model_config["text_decoder"]["beam_size"]
        self.max_length = model_config["text_decoder"]["max_length"]
        self.min_length = model_config["text_decoder"]["min_length"]
        self.max_prompt_length = model_config["text_decoder"]["max_prompt_length"]
        assert (self.min_length - self.max_prompt_length >= 0), "Minimum length must be greater than or equal to max prompt length"

        self.image_encoder = load_image_encoder(model_config["image_encoder"])
        encoder_hidden_size = model_config["image_encoder"].get("hidden_size", 768)
        self.text_decoder = load_text_decoder(model_config["text_decoder"], len(self.tokenizer), encoder_hidden_size)

        decoder_hidden_size = self.text_decoder.config.hidden_size
        self.image_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)

        if model_config["load_pretrained_weights"]:
            log.info("Loading pretrained weights for encoder-decoder setup")

            ckpt_path = model_config["load_pretrained_weights"]
            ckpt = load_hf_checkpoint(ckpt_path, map_location="cpu")
            if "model_state_dict" not in ckpt:
                raise KeyError(f"Checkpoint does not contain key 'model_state_dict'")

            self.load_state_dict(ckpt["model_state_dict"], strict=True)
            if model_config.get("freeze_backbone_weights", False):
                log.info("Freezing model weights")
                for param in self.parameters():
                    param.requires_grad = False
        elif model_config["load_backbone_weights"]:
            self.image_encoder = load_pretrained_image_encoder_weights(model_config, "model_state_dict")

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

    def _setup_tokenizer(self):
        special_tokens_dict = {
            'additional_special_tokens': ['<findings>', '<sep>', '[ENC]', '[DEC]']
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'

        medical_vocab = CHEXPERT_LABELS + ['infiltrate', 'nodule', 'mass']
        self.tokenizer.add_tokens(medical_vocab)

    def encode_image(self, image):
        image_features = self.image_encoder(image)
        global_features = image_features[:, 0]  # use [CLS] token as global image descriptor
        global_features = self.image_proj(global_features)

        return global_features

    @staticmethod
    def _post_process_generated_text(text):
        if "<sep>" in text:
            parts = text.split("<sep>", 1)
            if len(parts) > 1:
                text = parts[1].strip()
            else:
                text = text.strip()
        elif "<findings>" in text:
            parts = text.split("<findings>", 1)
            if len(parts) > 1:
                text = parts[1].strip()

        text = text.replace("<findings>", "").replace("<sep>", "").strip()
        text = ' '.join(text.split())

        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]

        if text and text[-1] not in '.!?':
            text += '.'

        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace('  ', ' ')

        return text

    @staticmethod
    def _prepare_findings_prompts(concept_prompts):
        input_prompts = []
        for prompt in concept_prompts:
            input_prompt = f"<findings> {prompt} <sep>"
            input_prompts.append(input_prompt)

        return input_prompts

    @staticmethod
    def _create_training_inputs(prompts, reports):
        full_inputs = []
        for prompt, report in zip(prompts, reports):
            full_input = f"{prompt} {report.strip()}"
            full_inputs.append(full_input)
        return full_inputs

    def _create_label_mask(self, input_ids, prompt_lengths):
        labels = input_ids.clone()

        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100

            padding_mask = input_ids[i] == self.tokenizer.pad_token_id
            labels[i][padding_mask] = -100

        return labels

    def forward(self, batch, prompts=None, device=None):
        device = batch["images"].device if device is None else device

        image_embeds = self.encode_image(batch["images"].to(device))
        image_embeds = image_embeds.unsqueeze(1)  # add sequence dimension
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

        texts = batch["texts"]
        if prompts is None:
            prompts = ["<findings> <sep>"] * len(texts)
        else:
            prompts = self._prepare_findings_prompts(prompts)

        full_texts = self._create_training_inputs(prompts, texts)

        text_inputs = self.tokenizer(
            full_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(device)

        prompt_lengths = []
        for prompt in prompts:
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_lengths.append(len(prompt_tokens['input_ids']))

        decoder_targets = self._create_label_mask(text_inputs.input_ids, prompt_lengths)

        outputs = self.text_decoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
            is_decoder=True,
            mode='multimodal'
        )

        return {
            "logits": outputs.logits,
            "labels": decoder_targets,
            "output_loss": outputs.loss,
        }

    def generate(self, images, prompts=None,
                 top_p=0.9, temperature=1.0, repetition_penalty=1.2,
                 num_beams=None, sample=False, device=None
                 ):
        if num_beams is None:
            num_beams = self.beam_size
        max_new_tokens = self.max_length - self.max_prompt_length

        with torch.no_grad():
            image_embeds = self.encode_image(images)
            image_embeds = image_embeds.unsqueeze(1)  # add sequence dimension
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

            if prompts is None:
                prompts = ["<findings> <sep>"] * images.size(0)  # batch size
            else:
                prompts = self._prepare_findings_prompts(prompts)

            prompt_inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            if not sample and num_beams > 1:
                image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
                image_atts = image_atts.repeat_interleave(num_beams, dim=0)

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "min_length": prompt_inputs.input_ids.shape[1] + 30,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "repetition_penalty": repetition_penalty,
                "encoder_hidden_states": image_embeds,
                "encoder_attention_mask": image_atts,
                "no_repeat_ngram_size": 3,
                "length_penalty": 1.2,
                "early_stopping": True,
            }

            if sample:
                generation_kwargs.update({
                    "do_sample": True,
                    "top_p": top_p,
                    "top_k": 50,
                    "temperature": max(temperature, 0.7),
                    "num_return_sequences": 1,
                })
            else:
                generation_kwargs.update({
                    "num_beams": num_beams,
                    "early_stopping": True,
                    "num_return_sequences": 1,
                })

            outputs = self.text_decoder.generate(
                input_ids=prompt_inputs.input_ids,
                attention_mask=prompt_inputs.attention_mask,
                **generation_kwargs
            )

            generated_texts = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            cleaned_texts = []
            for i, text in enumerate(generated_texts):
                text = self._post_process_generated_text(text)
                cleaned_texts.append(text)

        return cleaned_texts
