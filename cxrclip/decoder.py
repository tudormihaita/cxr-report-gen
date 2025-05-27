import os
import torch
import torch.nn as nn

from typing import Dict, TypeVar
from transformers import GPT2LMHeadModel, GPT2Config

from constants import CHEXPERT_LABELS
from utils.logger import LoggerManager
from .modules import load_image_encoder, load_text_decoder

T = TypeVar("T", bound="Module")
log = LoggerManager.get_logger(__name__)


class XRGenModel(nn.Module):
    def __init__(
            self,
            model_config: Dict,
            tokenizer,
            use_custom: bool = False,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        special_tokens_dict = { 'additional_special_tokens': ['<findings>'] }
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.max_seq_length = model_config["text_decoder"]["max_seq_length"]
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

        cache_dir = model_config["text_decoder"].get("cache_dir", "~/.cache/huggingface/hub")
        decoder_name = model_config["text_decoder"]["name"].lower()
        decoder_config = GPT2Config.from_pretrained(
            decoder_name,
            cache_dir=cache_dir,
        )
        decoder_config.add_cross_attention = True
        self.text_decoder = GPT2LMHeadModel.from_pretrained(
            decoder_name,
            config=decoder_config,
            cache_dir=cache_dir,
        )
        self.text_decoder.resize_token_embeddings(len(tokenizer))

        encoder_hidden_size = model_config["image_encoder"]["hidden_size"]
        decoder_hidden_size = self.text_decoder.config.hidden_size

        self.image_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)

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
        global_features = self.image_proj(global_features)
        return global_features

    def _prepare_prompt_with_findings(self, concept_labels):
        prompts = []
        for labels in concept_labels:
            findings = [CHEXPERT_LABELS[i] for i in torch.where(labels == 1)[0].cpu().numpy()]
            if findings:
                findings_text = ', '.join(findings)
                prompt = self.prompt_template.format(findings_text)
            else:
                prompt = self.prompt_template.format("No abnormal findings")
            prompts.append(prompt)

        return prompts

    def forward(self, batch, device=None):
        device = batch["images"].device if device is None else device

        image_embeds = self.encode_image(batch["images"].to(device))
        image_embeds = image_embeds.unsqueeze(1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

        texts = batch["texts"]
        prompts = self._prepare_prompt_with_findings(batch["labels"])
        full_texts = [f"{self.instruction_prefix}{p} <findings> {t}" for p, t in zip(prompts, texts)]

        text_inputs = self.tokenizer(
            full_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        ).to(device)

        decoder_targets = text_inputs.input_ids.clone()
        findings_token = self.tokenizer.convert_tokens_to_ids("<findings>")
        for i in range(decoder_targets.size(0)):
            idx = (text_inputs.input_ids[i] == findings_token).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                decoder_targets[i, :idx[0] + 1] = -100
            else:
                decoder_targets[i, :] = -100

        outputs = self.text_decoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True
        )

        return {
            "logits": outputs.logits,
            "labels": decoder_targets,
            "loss": outputs.loss,
        }

    def generate(self, images, findings,
                 num_beams=None, top_p=0.9, temperature=1.0, repetition_penalty=1.0,
                 sample=False, device=None
                 ):
        if num_beams is None:
            num_beams = self.beam_size

        with torch.no_grad():
            image_embeds = self.encode_image(images)
            image_embeds = image_embeds.unsqueeze(1)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

            prompts = self._prepare_prompt_with_findings(findings)
            full_prompts = [f"{self.instruction_prefix}{p} <findings>" for p in prompts]
            prompt_inputs = self.tokenizer(
                full_prompts,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)

            max_new_tokens = self.max_seq_length - prompt_inputs.input_ids.shape[1]
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
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
                input_ids=prompt_inputs.input_ids,
                attention_mask=prompt_inputs.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                **generation_kwargs
            )

            generated_texts = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            generated_texts = [text.strip() for text in generated_texts]
        return generated_texts
