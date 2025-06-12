import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from typing import Tuple, Union
from collections import OrderedDict

from constants import *


class LayerNorm(nn.LayerNorm):
    """
    Subclass of LayerNorm that casts the input tensor to float32 before performing the normalization, then casts it back to the original type.
    Avoids numerical instability in the normalization process when working with lower precision data types.
    """

    def forward(self, x: torch.Tensor):
        original_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(original_type)


class QuickGELU(nn.Module):
    """
    A faster version of GELU activation function that approximates the original function with a polynomial.
    """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        """
        Initializes a form of attention mechanism that pools spatial data into a fixed-size representation.

        :param spacial_dim: spatial dimension of the input (for example, height and width of a 2d image).
        :param embed_dim: embedding dimension of the input features. represents the dimensionality of the feature space for each spatial location in the input.
        :param num_heads: number of attention heads in MHA mechanism. allows the model to focus on different parts of the input simultaneously.
        :param output_dim: output dimension. if not provided, defaults to embed_dim, therefore no pooling is performed.
        """
        super().__init__()
        # tensor that encodes the positions of the input features. the +1 accounts for an additional embedding for the mean of the input features
        # division by embed_dim ** 0.5 normalizes the variance of the embeddings
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # Projection layers for the key, query, and value vectors in the attention mechanism
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # output projection layer that projects the output of the attention mechanism to the desired output dimension
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # input tensor is flattened starting from 2nd dimension, then permuted to have the spatial dimension as the first dimension
        # e.g.: if the input shape is (N, C, H, W), flatten will combine height and width into a single dimension, then permute will result in (HW, N, C),
        # where N = batch size, C = number of channels, H = height, W = width
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        # concatenates the mean of the spatial positions across the batch to the input tensor
        # extra token that represents the global average of the input features, helping the attention mechanism to attend to a global context
        x = x + self.positional_embedding[:, None, :].to(x.dtype)

        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        # after the attention mechanism, the resulting vector is reshaped to remove the first token corresponding to the global mean
        return x.squeeze(0)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        """
        Initializes a residual block with the following components:
         - MHA: multi-head self-attention mechanism
         - FFN: feed-forward-network, represented by a 2-layer MLP with QuickGELU activation function

        :param d_model: dimensionality of the model (embedding size)
        :param n_head: number of attention heads used in the MHA mechanism
        :param attn_mask: attention mask used to prevent attention from certain tokens (e.g., padding tokens or future tokens in autoregressive models)
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(d_model, d_model * 4)),
            ('gelu', QuickGELU()),
            ('c_proj', nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # follows the Pre-LN transformer design, where the layer normalization is applied before the attention mechanism
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        """
        Initializes a Vision Transformer model, similar to the one introduced by Dosovitskiy et al., but with a series of adaptations for the CLIP framework:
            - Instead of flattening the raw image patches and applying a linear projection, the model uses a convolutional layer with kernel size equal to patch size and takes advantage of weight sharing in filters.
            - The model uses a special learnable class token (CLS) which will act as a global representation of the image after being processed by the transformer
            - The entire sequence undergoes layer normalization before being fed into the transformer.
            - The CLS token embedding is finally linearly projected to output_dim using a projection matrix, which is a latent space for downstream contrastive learning

        :param input_resolution: resolution of the input image (e.g., 224 for a 224x224 image)
        :param patch_size: size of the patches in which the image is divided
        :param width: number of channels in the hidden layers of the transformer
        :param layers: number of transformer layers
        :param heads: number of attention heads
        :param output_dim: final output dimension, for mapping the embeddings in a latent representation space
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.embed_dim = width
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # apply convolution to divide the image into patches; shape is (N, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # reshape to (N, width, grid ** 2), flattening the images patches to a single sequence for the transformer to process
        x = x.permute(0, 2, 1)  # permute to (N, grid ** 2, width)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # add CLS token to the sequence of patches
        x += self.positional_embedding.to(x.dtype)  # Add positional embeddings
        x = self.ln_pre(x)

        x = x.permute(1, 0,
                      2)  # NLD -> LND: permute to match the transformer's input shape (batch_size, num_patches, embedding_dim)
        x = self.transformer(x)  # apply the transformer
        x = x.permute(1, 0, 2)  # permute back to the original shape

        x_cls = self.ln_post(x[:, 0, :])  # apply layer normalization to the CLS token
        if self.proj is not None:
            x_proj = x_cls @ self.proj  # dot product for mapping the embeddings to the output dimension in the projection layer
            return x_proj, x_cls

        return x_cls, x_cls


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # visual encoder
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 # can be either a tuple, defining a ResNet with 4 stages, or an integer, defining the layers of the ViT
                 vision_width: int,  # feature width: number of channels in the hidden layers
                 vision_patch_size: int,  # size of patches used in ViT
                 # text encoder
                 context_length: int,  # maximum number of tokens in the input text
                 vocab_size: int,
                 transformer_width: int,  # hidden dimension of the transformer model for text
                 transformer_heads: int,  # number of attention heads in the transformer model for text
                 transformer_layers: int,
                 num_medical_concepts: int = 14,
                 extended_context_length: int = 248,
                 extended_context: bool = False,
                 load_from_clip: bool = True,
                 ):
        super().__init__()

        self.vision_width = vision_width
        self.transformer_width = transformer_width

        self.extended_context = extended_context
        self.context_length = context_length
        self.extended_context_length = extended_context_length
        self.num_classes = num_medical_concepts

        if isinstance(vision_layers, int):
            # ViT initialization
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
        else:
            # ResNet vision encoder initialization: will be used for future experiments
            raise NotImplementedError("This version of CLIP only supports ViT for visual encoder")

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        if not load_from_clip and self.extended_context:
            self.positional_embedding = nn.Parameter(torch.empty(self.extended_context_length, transformer_width))
            self.positional_embedding_res = nn.Parameter(torch.empty(self.extended_context_length, transformer_width))
        else:
            self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))

        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self.medical_concept_classifier = ChexpertConceptClassifier(self.vision_width, self.transformer_width, num_medical_concepts)
        self.medical_concept_classifier = nn.Sequential(
            nn.Linear(self.vision_width, self.transformer_width),
            nn.ReLU(),
            nn.Linear(self.transformer_width, num_medical_concepts)
        )
        self.concept_embedding = nn.Parameter(torch.empty(num_medical_concepts, embed_dim))

        self.initialize_parameters()

        self.mask1 = torch.zeros([self.extended_context_length, 1])
        self.mask1[:20, :] = 1
        self.mask2 = torch.zeros([self.extended_context_length, 1])
        self.mask2[20:, :] = 1

    def initialize_parameters(self):
        """
        Initializes the learnable parameters of the model, using values drawn from a normal (Gaussian) distribution with a mean of 0 and a given standard deviation.
        Chosen initial standard deviation values for the text encoder weights are small to ensure embeddings remain well scaled in the first steps of training.
        For the visual encoder weights, standard deviation values are based on scaling rules from DL research, specifically:
            - Xavier Initialization: scales weights based on layer width
            - Root Mean Square Scaling: scales weights based on layer depth

        """
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        nn.init.normal_(self.concept_embedding, std=0.01)
        for m in self.medical_concept_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """
        Causal attention mask to prevent attending to future tokens. Fills the upper triangular part of the matrix with -inf for masked positions'
        """
        context_length = self.extended_context_length if self.extended_context else self.context_length
        mask = torch.empty(context_length, context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        """
        Encode the image and return both the projected embeddings and raw features from the CLS token.
        """
        proj_features, cls_features = self.visual(image.type(self.dtype))
        return proj_features, cls_features

    def encode_text(self, text, full: bool = False):
        """
        Encode the text and return the projected embeddings.
        """
        x = self.token_embedding(text).type(self.dtype)  # initial shape of (batch_size, context_length, d_model)

        if hasattr(self, 'positional_embedding_res'):
            x = x + self.positional_embedding.type(self.dtype) * self.mask1.to(x.device).type(self.dtype).to(
                x.device) + (self.positional_embedding_res.to(x.device) * self.mask2.to(x.device)).type(self.dtype).to(
                x.device)
        else:
            x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND: shape of (context_length, batch_size, d_model)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD: reorder back to original shape
        x = self.ln_final(x).type(self.dtype)

        # take the features from the EOS (end of sequence) token embedding and project them to the shared multimodal space output dimension
        # the last token is used as it captures the entire sequence meaning
        if not full:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def predict_medical_concepts(self, cls_features):
        """
        Predict medical concepts from the visual features.
        """
        return self.medical_concept_classifier(cls_features)

    def get_concepts_embeddings(self, medical_concepts):
        """
        Get concept-aware embeddings based on the medical concept labels from CheXpert.
        :param medical_concepts: tensor of shape [batch_size, num_medical_concepts] containing 0/1/-1 labels
                                ( 0: not present, 1: present, -1: uncertain)
        :return: concept_embeddings: tensor of shape [batch_size, embed_dim]
        """
        concept_weights = medical_concepts.clone()
        concept_weights[concept_weights == -1] = 0.0  # uncertain labels are treated as absent to address uncertainty

        weighted_concepts = torch.matmul(concept_weights,
                                         self.concept_embedding)  # [batch_size, num_concepts] @ [num_concepts, embed_dim] -> [batch_size, embed_dim]
        weighted_concepts = weighted_concepts / (
                weighted_concepts.norm(dim=-1, keepdim=True) + 1e-8)  # normalize to ensure they have unit length
        return weighted_concepts.to(self.dtype)

    def forward(self, image, text, medical_concepts=None):
        image_features, cls_features = self.encode_image(image)
        text_features = self.encode_text(text)
        # concepts_logits = self.predict_medical_concepts(cls_features)['logits']

        # feature vectors normalized to have a unit length (L2 normalization)
        # ensures cosine similarity is calculated purely based on the angle between the vectors and not their magnitudes
        image_features_norm = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
        text_features_norm = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)

        # learnable parameters that adjusts the scale of the logits (similarity scores)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        # similarity matrix where each entry represents the cosine similarity between an image and a text pair
        logits_per_image = logit_scale * image_features_norm @ text_features_norm.t()
        # transpose the similarity matrix to get the similarity scores for each text compared to an image in the batch
        logits_per_text = logits_per_image.t()

        outputs = {
            "logits_per_image": logits_per_image,  # image-to-text similarity
            "logits_per_text": logits_per_text,  # text-to-image similarity
            "image_features": image_features,  # projected image features
            "text_features": text_features,  # projected text features
            # "concepts_logits": concepts_logits,  # predicted medical concepts
        }

        if medical_concepts is not None:
            concepts_embeddings_norm = self.get_concepts_embeddings(medical_concepts)

            concepts_image_similarity = logit_scale * concepts_embeddings_norm @ image_features_norm.t()
            outputs["concepts_embeddings"] = concepts_embeddings_norm
            outputs["concepts_image_similarity"] = concepts_image_similarity

        return outputs


def convert_weights(model: nn.Module):
    """
    Converts the model weights to half precision (float16) for faster computation on hardware that supports it.
    :param model: model to convert
    """

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict = None, extended_context: bool = False, load_from_clip = False):
    if state_dict is None:
        model = CLIP(
            embed_dim=CLIP_EMBED_DIM,
            image_resolution=CLIP_IMAGE_RESOLUTION,
            vision_layers=CLIP_VISION_LAYERS,
            vision_width=CLIP_VISION_WIDTH,
            vision_patch_size=CLIP_PATCH_SIZE,
            context_length=CLIP_CONTEXT_LENGTH,
            vocab_size=CLIP_VOCAB_SIZE,
            transformer_width=CLIP_TRANSFORMER_WIDTH,
            transformer_heads=CLIP_TRANSFORMER_HEADS,
            transformer_layers=CLIP_TRANSFORMER_LAYERS,
            extended_context=extended_context,
            load_from_clip=load_from_clip,
        )
        return model.train()

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith("attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        raise NotImplementedError("This version of CLIP only supports ViT for visual encoder")

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        extended_context=extended_context, load_from_clip=load_from_clip,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if not load_from_clip:
        if missing_keys:
            print(f"[Warning] Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"[Warning] Unexpected keys in state_dict: {unexpected_keys}")

    return model.train()
