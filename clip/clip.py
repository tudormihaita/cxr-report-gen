import os
import torch
import hashlib
import warnings
import urllib.request

from torch import nn
from PIL import Image
from tqdm import tqdm
from packaging import version
from typing import Union, List

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from .model import build_model
from .tokenizer import BpeTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from torchvision.transforms import InterpolationMode

    BICUBIC = Image.Resampling.BICUBIC

if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

__all__ = ["load", "tokenize", "available_models"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    return list(_MODELS.keys())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False, extended_context: bool = True) -> torch.Tensor:
    """
    Returns the tokenized representation of given input string(s).

    :param texts: input string or a list of input string to tokenize
    :param context_length: context length to use; CLIP models use 77 as default
    :param truncate: whether to truncate the text in case its encoding is longer than the context length
    :param extended_context: whether to extend the positional embeddings to support longer text sequences
    :return: two-dimensional tensor containing the resulting tokens with the shape = [number of input strings, context_length]
    """
    if extended_context:
        context_length = 248

    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder['<|startoftext|>']
    eot_token = _tokenizer.encoder['<|endoftext|>']
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def load(
        name: str,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        jit: bool = False,
        download_root: str = None,
        load_from_clip: bool = True
):
    """
    Loads a pretrained CLIP model with the chosen architecture.

    :param name: a model name listed by 'available_models()' or the path to a model checkpoint containing the state_dict
    :param device: device to load the model onto
    :param jit: whether to load the optimized JIT model or the original state_dict
    :param download_root: path to download the model files; defaults to ~/.cache/clip
    :param load_from_clip: load a pre-trained model from the official OpenAI repository
    """

    def _download(url: str, root: str):
        os.makedirs(root, exist_ok=True)
        filename = os.path.basename(url)

        expected_sha256 = url.split("/")[-2]
        download_target = os.path.join(root, filename)

        if os.path.exists(download_target) and not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists but is not a regular file!")

        if os.path.isfile(download_target):
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
                return download_target
            else:
                warnings.warn(
                    f"{download_target} exists but the SHA256 checksum doesnt not match; Re-downloading the file...")

        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=90, unit='iB', unit_scale=True,
                      unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
            raise RuntimeError("Model has been downloaded but the SHA256 checksum does not match!")

        return download_target

    def _extend_positional_embeddings(positional_embedding_pre, keep_len=20):
        """
        Extend CLIP's positional embeddings to support longer text sequences.

        :param positional_embedding_pre: original positional embeddings
        :param keep_len: number of initial embeddings to keep unchanged. See: https://arxiv.org/abs/2403.15378
        :return: torch.Tensor: extended positional embeddings
        """
        length, dim = positional_embedding_pre.shape
        # new extended embedding shape (4x longer)
        positional_embedding_res = torch.zeros([4 * length - 3 * keep_len, dim], dtype=model.dtype)
        # preserve first `keep_len` positions
        positional_embedding_res[:keep_len] = positional_embedding_pre[:keep_len]
        # interpolate for extended positions
        for i in range(length - 1 - keep_len):
            start_idx = 4 * i + keep_len
            positional_embedding_res[start_idx] = positional_embedding_pre[i + keep_len]
            positional_embedding_res[start_idx + 1] = 3 * positional_embedding_pre[i + keep_len] / 4 + 1 * \
                                                      positional_embedding_pre[i + 1 + keep_len] / 4
            positional_embedding_res[start_idx + 2] = 2 * positional_embedding_pre[i + keep_len] / 4 + 2 * \
                                                      positional_embedding_pre[i + 1 + keep_len] / 4
            positional_embedding_res[start_idx + 3] = 1 * positional_embedding_pre[i + keep_len] / 4 + 3 * \
                                                      positional_embedding_pre[i + 1 + keep_len] / 4
        # extend final position smoothly
        last_idx = 4 * length - 3 * keep_len - 4
        for j in range(4):
            positional_embedding_res[last_idx + j] = positional_embedding_pre[length - 1] + j * (
                    positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2]) / 4

        return positional_embedding_res

    if load_from_clip:
        if name in _MODELS:
            model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(f"Model {name} not found; Available models = {available_models()}")
    else:
        model_path = name

    with open(model_path, "rb") as opened_file:
        try:
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict(), load_from_clip).to(device)

        if load_from_clip:
            positional_embedding_pre = model.positional_embedding.type(model.dtype)
            positional_embedding_resized = _extend_positional_embeddings(positional_embedding_pre, keep_len=20)

            model.positional_embedding = nn.Parameter(positional_embedding_resized, requires_grad=False)
            model.positional_embedding_res = nn.Parameter(positional_embedding_resized.clone(), requires_grad=True)

        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

        # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """
        Gets attributes of a node which is polymorphic over return type.
        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())
