import os
import torch
import torch.nn as nn

from typing import Dict, TypeVar
from constants import CHEXPERT_LABELS
from utils.logger import LoggerManager
from .modules import load_image_encoder, load_text_decoder

T = TypeVar("T", bound="Module")
log = LoggerManager.get_logger(__name__)


class CxrCLIPDecoder(nn.Module):
    def __init__(
            self,
            model_config: Dict,
            tokenizer,
            use_custom: bool = False,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = model_config["text_decoder"]["max_length"]
        self.min_length = model_config["text_decoder"]["min_length"]

        self.prompt_template = model_config["text_decoder"]["prompt_template"]
        self.beam_size = model_config["text_decoder"]["beam_size"]

        if model_config["load_backbone_weights"] is None:
            self.image_encoder = load_image_encoder(model_config["image_encoder"])
        else:
            log.info("Loading pre-trained image encoder from checkpoint")
            if not os.path.isfile(model_config["load_backbone_weights"]):
                raise ValueError(f"Cannot find a weight file: {model_config['load_backbone_weights']}")
            ckpt = torch.load(model_config["load_backbone_weights"], map_location="cpu", weights_only=False)
            model_key = "model_state_dict"
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

        self.text_decoder = load_text_decoder(model_config["text_decoder"], len(tokenizer))

        encoder_hidden_size = model_config["image_encoder"].get("hidden_size", 768)
        decoder_hidden_size = self.text_decoder.config.d_model
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

        global_features = image_features[:, 0]
        return global_features

    def forward(self, batch, device=None):
        device = batch["images"].device if device is None else device
        image_embeds = self.encode_image(batch["images"].to(device))
        image_embeds = self.image_proj(image_embeds)
        image_atts = torch.ones((image_embeds.shape[0], 1), dtype=torch.long).to(device)

        concept_labels = batch["labels"]
        prompts = []
        for labels in concept_labels:
            findings = [CHEXPERT_LABELS[i] for i in torch.where(labels == 1)[0].cpu().numpy()]
            findings_caption = ', '.join(findings) if findings else "No Finding"
            prompt = self.prompt_template.format(findings_caption)
            prompts.append(prompt)


        gt_texts = [p + " " + t for p, t in zip(prompts, batch["texts"])]
        text_inputs = self.tokenizer(
            gt_texts,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        input_ids = text_inputs.input_ids.to(device)
        text_attention_mask = text_inputs.attention_mask.to(device)

        labels = input_ids.clone()
        prompt_lengths = []
        for prompt in prompts:
            prompt_tokens = self.tokenizer.encode(
                prompt,
                add_special_tokens=False
            )
            prompt_lengths.append(len(prompt_tokens))

        for i, prompt_length in enumerate(prompt_lengths):
            if self.tokenizer.bos_token_id is not None and input_ids[i, 0] == self.tokenizer.bos_token_id:
                prompt_length += 1
            labels[i, :prompt_length] = -100
        labels = labels.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=text_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=labels,
            return_dict=True,
        )

        return {
            "logits": outputs.logits,
            "labels": labels
        }

    def generate(self, image_embeds, image_atts, prompt=None, findings=None,
                 num_beams=None, top_p=0.9, temperature=1.0, repetition_penalty=1.0,
                 sample=False
                 ):
        if num_beams is None:
            num_beams = self.beam_size

        prompts = []
        for i in range(image_embeds.size(0)):
            if findings is not None:
                pos_labels = (findings[i] == 1).nonzero(as_tuple=False).view(-1).tolist()
                findings_caption = ', '.join([CHEXPERT_LABELS[j] for j in pos_labels]) if pos_labels else "No Finding"
            else:
                findings_caption = ""

            prompt_str = self.prompt_template.format(findings_caption)
            prompts.append(prompt_str)

        prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = prompt_inputs.input_ids.to(image_embeds.device)

        if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
            input_ids[:, 0] = self.tokenizer.bos_token_id

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        if sample:
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=self.max_length,
                min_length=self.min_length,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=3,
                **model_kwargs
            )
        else:
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=3,
                **model_kwargs
            )

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            if prompt and caption.startswith(prompt):
                caption = caption[len(prompt):].strip()
            captions.append(caption)

        return captions




