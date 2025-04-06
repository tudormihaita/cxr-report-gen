import torch
import torch.nn as nn

from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput
from constants import BART_TYPE

class CLIP2BARTModel(nn.Module):
    def __init__(
            self,
            vision_encoder,
            decoder_model_name=BART_TYPE,
            freeze_clip_encoder=True,
            dropout_rate=0.1
    ):
        super(CLIP2BARTModel, self).__init__()
        self.vision_encoder = vision_encoder

        self.tokenizer = BartTokenizer.from_pretrained(decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.bart_decoder = BartForConditionalGeneration.from_pretrained(decoder_model_name)
        self.bart_decoder.config.encoder_layers = 0

        self.image_embed_dim = vision_encoder.embed_dim
        self.bart_hidden_dim = self.bart_decoder.config.d_model

        if freeze_clip_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(self.image_embed_dim, self.bart_hidden_dim),
            nn.LayerNorm(self.bart_hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

    def forward(self, pixel_values, labels=None, mode='forward'):
        with torch.no_grad():
            _, vision_embeds = self.vision_encoder(pixel_values)
            vision_embeds = vision_embeds.unsqueeze(1)

        vision_embeds = self.projection(vision_embeds)

        if mode == 'forward':
            outputs = self.bart_decoder(
                encoder_outputs=BaseModelOutput(last_hidden_state=vision_embeds),
                labels=labels,
                return_dict=True
            )

            return outputs
        else:
            generated_ids = self.bart_decoder.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=vision_embeds),
                max_length=256,
                num_beams=4,
            )

            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)