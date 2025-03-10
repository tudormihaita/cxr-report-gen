from transformers import BartConfig

class MultiModalBartConfig(BartConfig):
    def __init__(self,
                 vocab_size=50320,
                 d_model=1024,
                 image_feature_size=512,
                 encoder_ffn_dim=4096,
                 encoder_layers=12,
                 decoder_attention_heads=16,
                 max_position_embeddings=1024,
                 dropout=0.1,
                 attention_dropout=0.0,
                 init_std=0.02,
                 is_encoder_decoder=True,
                 pad_token_id=1,
                 bos_token_id=0,
                 eos_token_id=2,
                 decoder_start_token_id=0,
                 image_projection_dim=1024, # project image embeddings to BART hidden size
                 lm_loss_refactor=1.0,
                 image_loss_refactor=1.0,
                 **kwargs
                 ):
        super().__init__(

        )