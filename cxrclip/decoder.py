import torch
import torch.nn as nn
from typing import Dict, TypeVar

from constants import CHEXPERT_LABELS
from .modules.prompt_constructor import PromptStrategy
from .modules import load_image_encoder, load_text_decoder, load_image_classifier, \
    load_pretrained_image_encoder_weights, load_prompt_constructor

T = TypeVar("T", bound="Module")


class XRGenModel(nn.Module):
    def __init__(
            self,
            model_config: Dict,
            tokenizer,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self._setup_tokenizer()

        self.findings_token_id = self.tokenizer.convert_tokens_to_ids("<findings>")
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids("<report>")

        self.beam_size = model_config["text_decoder"]["beam_size"]
        self.max_length = model_config["text_decoder"]["max_length"]
        self.min_length = model_config["text_decoder"]["min_length"]
        self.max_prompt_length = model_config["text_decoder"]["max_prompt_length"]
        assert(self.min_length - self.max_prompt_length >= 0), "Minimum length must be greater than or equal to max prompt length"

        if model_config["load_backbone_weights"] is None:
            self.image_encoder = load_image_encoder(model_config["image_encoder"])
        else:
            self.image_encoder = load_pretrained_image_encoder_weights(model_config, "model_state_dict")

        self.classifier = load_text_decoder(model_config["classifier"]["config"], self.image_encoder.out_dim)

        encoder_hidden_size = model_config["image_encoder"].get("hidden_size", 768)
        self.text_decoder = load_text_decoder(model_config["text_decoder"], len(self.tokenizer), encoder_hidden_size)

        decoder_hidden_size = self.text_decoder.config.hidden_size
        self.image_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)

        self.prompt_strategy = model_config["text_decoder"]["prompt_strategy"]
        self.prompt_constructor = load_prompt_constructor(model_config["text_decoder"].get("use_diverse_templates", False))

    def _setup_tokenizer(self):
        special_tokens_dict = {
            'additional_special_tokens': ['<findings>', '<report>']
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'

        medical_vocab = CHEXPERT_LABELS + ['infiltrate', 'nodule', 'mass']
        self.tokenizer.add_tokens(medical_vocab)

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
        global_features = image_features[:, 0]  # use [CLS] token as global image descriptor
        global_features = self.image_proj(global_features)

        return global_features

    @staticmethod
    def _post_process_generated_text(text):
        if "<report>" in text:
            parts = text.split("<report>", 1)
            if len(parts) > 1:
                text = parts[1].strip()
            else:
                text = text.strip()
        elif "<findings>" in text:
            parts = text.split("<findings>", 1)
            if len(parts) > 1:
                text = parts[1].strip()

        text = text.replace("<findings>", "").replace("<report>", "").strip()
        text = ' '.join(text.split())

        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]

        if text and text[-1] not in '.!?':
            text += '.'

        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace('  ', ' ')

        return text

    def _prepare_findings_prompts(self, concept_labels):
        generated_prompts = self.prompt_constructor.construct_prompts(concept_labels)
        input_prompts = []
        for prompt in generated_prompts:
            input_prompt = f"<findings> {prompt} <report>"
            input_prompts.append(input_prompt)

        return input_prompts

    def _create_training_inputs(self, prompts, reports):
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

    def forward(self, batch, device=None):
        device = batch["images"].device if device is None else device

        image_embeds = self.encode_image(batch["images"].to(device))
        image_embeds = image_embeds.unsqueeze(1)  # add sequence dimension
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

        texts = batch["texts"]
        labels = batch.get("labels", None)

        if labels is not None:
            prompts = self._prepare_findings_prompts(labels)
        else:
            prompts = ["<findings> <report>"] * len(texts)

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
            "model_loss": outputs.loss,
            "full_texts": full_texts,
            "prompts": prompts,
        }

    def generate(self, images, findings=None,
                 top_p=0.9, temperature=1.0, repetition_penalty=1.2,
                 num_beams=None, sample=False, device=None
        ):
        max_new_tokens = self.max_length - self.max_prompt_length

        with torch.no_grad():
            image_embeds = self.encode_image(images)
            image_embeds = image_embeds.unsqueeze(1) # add sequence dimension
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

            if findings is not None:
                prompts = self._prepare_findings_prompts(findings)
            else:
                prompts = ["<findings> <report>"] * len(images)

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
                "min_length": self.min_length,
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
                    "temperature": temperature,
                    "num_return_sequences": 1,
                })
            else:
                generation_kwargs.update({
                    "num_beams": num_beams if num_beams is not None else self.beam_size,
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

        if not sample and num_beams > 1:
            final_texts = []
            for i in range(0, len(cleaned_texts), num_beams):
                final_texts.append(cleaned_texts[i])
            return final_texts

        return cleaned_texts
