import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from transformers import BartConfig, BeamSearchScorer


class PositionalEncoding(nn.Module):
    """
    Classic transformer positional encoding
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, d_model, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CrossAttention(nn.Module):
    """
    Cross-Attention module for attending over encoder outputs from image modality
    """
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.output(context_layer)
        return output

class DecoderLayer(nn.Module):
    """
    Single decoder layer with Self-Attention, Cross-Attention and Feed-Forward Network
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        # self-attention for decoder
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size)

        # cross-attention between decoder and encoder
        self.cross_attn = CrossAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout
        )
        self.cross_attn_layer_norm = nn.LayerNorm(self.hidden_size)

        # feed-forward network
        self.fc1 = nn.Linear(config.hidden_size, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)
        self.activation_fn = F.gelu

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            self_attn_mask=None,
            cross_attn_mask=None
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states_transposed = hidden_states.transpose(0, 1)
        hidden_states_transposed, _ = self.self_attn(
            query=hidden_states_transposed,
            key=hidden_states_transposed,
            value=hidden_states_transposed,
            attn_mask=self_attn_mask
        )
        hidden_states = hidden_states_transposed.transpose(0, 1)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=cross_attn_mask
        )
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        return hidden_states


class VisionTextDecoderModel(nn.Module):
    """
    Main model class that connects CLIP vision encoder with BART-inspired text decoder
    """

    def __init__(
            self,
            vision_encoder,
            vocab_size,  # size of text vocabulary
            hidden_size=768,
            decoder_layers=6,
            decoder_attention_heads=12,
            decoder_ffn_dim=3072,
            max_position_embeddings=512,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
    ):
        super().__init__()

        self.vision_encoder = vision_encoder

        # lock vision encoder weights (optional - depends on the training strategy)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # create config for decoder (similar to BART config)
        decoder_config = BartConfig(
            vocab_size=vocab_size,
            d_model=hidden_size,
            encoder_layers=1,  # not used, but needed for config
            decoder_layers=decoder_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        self.config = decoder_config

        # projection layer to align CLIP visual features with decoder dimensions
        self.vision_projection = nn.Linear(
            self.vision_encoder.output_dim,  # TODO: replace with actual dimension of CLIP encoder output
            hidden_size
        )

        # token embeddings for decoder
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = PositionalEncoding(
            d_model=hidden_size,
            max_seq_length=max_position_embeddings,
            dropout=dropout
        )

        # decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(decoder_config) for _ in range(decoder_layers)
        ])

        # final layer norm
        self.decoder_layernorm = nn.LayerNorm(hidden_size)

        # output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to BART initialization"""
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        """Get token embedding layer"""
        return self.embed_tokens

    def forward(
            self,
            image,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            use_cache=None,
            return_dict=None,
    ):
        """
        Forward pass for the entire model

        :param image: input image for the vision encoder
        :param decoder_input_ids: input token IDs for the text decoder
        :param decoder_attention_mask: attention mask for decoder inputs
        :param encoder_outputs: pre-computed encoder outputs (if available)
        :param past_key_values: cached key values for faster inference
        :param use_cache: whether to use cached key values
        :param return_dict: whether to return a dictionary or tuple
        """

        # get vision features if not provided
        if encoder_outputs is None:
            vision_features = self.vision_encoder(image)

            # project vision features to match decoder dimensions
            encoder_hidden_states = self.vision_projection(vision_features)

            # add batch dimension if needed (depending on CLIP encoder output shape)
            if len(encoder_hidden_states.shape) == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
        else:
            encoder_hidden_states = encoder_outputs

        # embed decoder tokens and add positional embeddings
        decoder_embeddings = self.embed_tokens(decoder_input_ids)

        decoder_hidden_states = self.embed_positions(decoder_embeddings)

        # create causal attention mask
        batch_size, seq_length = decoder_input_ids.size()
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=decoder_input_ids.device) * float('-inf'),
            diagonal=1
        )

        # process decoder layers
        for decoder_layer in self.decoder_layers:
            decoder_hidden_states = decoder_layer(
                hidden_states=decoder_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                self_attn_mask=causal_mask,
                cross_attn_mask=None,  # all encoder outputs can be attended to
            )

        decoder_hidden_states = self.decoder_layernorm(decoder_hidden_states)

        # project to vocabulary
        logits = self.output_projection(decoder_hidden_states)

        return logits


class VisionTextGenerator:
    """
    Helper class to generate text reports from input X-Ray images
    """

    def __init__(self, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate(
            self,
            image,
            max_length=50,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            temperature=1.0,
    ):
        """
        Generate text from image using beam search

        :param image: input image tensor
        :param max_length: maximum length of generated text
        :param num_beams: number of beams for beam search
        :param early_stopping: whether to stop when all beams reach EOS
        :param no_repeat_ngram_size: size of n-grams to avoid repeating
        :param temperature: temperature for sampling
        """
        image = image.to(self.device)

        # encode image with vision encoder
        with torch.no_grad():
            vision_features = self.model.vision_encoder(image)
            encoder_hidden_states = self.model.vision_projection(vision_features)

            # add sequence dimension if needed
            if len(encoder_hidden_states.shape) == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)

        batch_size = image.size(0)

        # start with BOS token
        input_ids = torch.tensor([[self.model.config.bos_token_id]] * batch_size).to(self.device)

        # setup for beam search
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=1.0,
        )

        # expand encoder outputs for beam search
        expanded_encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_beams, dim=0)

        # initialize beam search
        beam_inputs = {
            "input_ids": input_ids.repeat_interleave(num_beams, dim=0),
            "encoder_outputs": expanded_encoder_hidden_states,
            "past_key_values": None,
            "attention_mask": None,
        }

        # TODO: Implement beam search
        # this is a simplified version for greedy decoding
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(
                    image=None,  # not needed, using cached encoder outputs
                    decoder_input_ids=input_ids,
                    encoder_outputs=encoder_hidden_states,
                )

            next_token_logits = outputs[:, -1, :] / temperature
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            # stop if all sequences have EOS
            if (next_token_id == self.model.config.eos_token_id).all():
                break

        # decode to text
        generated_text = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )

        return generated_text