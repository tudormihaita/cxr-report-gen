import logging
from pathlib import Path
from typing import Union, Optional, Dict, List, Any

import torch
from PIL import Image

from cxrclip import build_model
from configs import load_config_from_file
from data import load_tokenizer, load_transform, transform_image

log = logging.getLogger(__name__)


class CLIPXRGenPipeline:
    def __init__(
            self,
            model,
            tokenizer,
            transform,
            device: torch.device,
            config: Dict[str, Any],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.transform = transform
        self.device = device
        self.config = config

        self.model.eval().to(self.device)

    @classmethod
    def from_pretrained(
            cls,
            model_path: Union[str, Path],
            config_path: Union[str, Path],
            device: Union[str, torch.device] = "auto",
            **kwargs
    ) -> "CLIPXRGenPipeline":
        model_path = Path(model_path)
        config_path = Path(config_path)

        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        log.info(f"Loading model from {model_path}")
        log.info(f"Using device: {device}")

        if not config_path.is_file():
            log.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config = load_config_from_file(config_path)

        tokenizer = load_tokenizer(**config["tokenizer"])
        transform = load_transform(split="inference", transform_config=config["transform"])

        backbone = build_model(config["encoder"]["model"], config["encoder"]["loss"], tokenizer)
        model = build_model(config["decoder"]["model"], config["decoder"]["loss"], tokenizer,
                            pretrained_backbone=backbone)

        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint["model_state_dict"],
                    strict=False
                )

                log.info(f"Loaded decoder checkpoint")
                log.info(f"Missing keys: {len(missing_keys)}")
                log.info(f"Unexpected keys: {len(unexpected_keys)}")

                log.info(f"Loaded model from {model_path}")
            except Exception as e:
                log.error(f"Failed to load weights from {model_path}: {e}")
                raise RuntimeError(f"Failed to load model weights from {model_path}: {e}")
        else:
            log.warning(f"Checkpoint {model_path} not found Initializing model without pre-trained weights.")

        model = model.to(device)

        return cls(
            model=model,
            tokenizer=tokenizer,
            transform=transform,
            device=device,
            config=config
        )

    def __preprocess_image(self, image: Union[Image.Image, str, Path]) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type {type(image)}.")

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_tensor = transform_image(self.transform, image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor

    def __call__(
            self,
            image: Union[Image.Image, str, Path],
            findings: Optional[List[str]] = None,
            num_beams: int = 3,
            temperature: float = 1.0,
            top_p: float = 1.0,
            repetition_penalty: float = 1.1,
            sample: bool = False,
            **kwargs
    ) -> str:
        image_tensor = self.__preprocess_image(image)

        with torch.no_grad():
            try:
                reports = self.model.generate(
                    images=image_tensor,
                    findings=findings,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    sample=sample,
                    device=self.device,
                    **kwargs
                )

                if isinstance(reports, list):
                    report = reports[0] if reports else "Unable to annotate image."
                else:
                    report = reports

                return report
            except Exception as e:
                log.error(f"Error during report generation: {e}")
                raise RuntimeError(f"Failed to generate report: {e}") from e

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.config.get("model_name", "CLIPXRGen"),
            "version": self.config.get("version", "1.0"),
            "description": self.config.get("description", "CLIP-based X-ray report generation model"),
            "device": str(self.device),
            "vocab_size": len(self.tokenizer),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "config": self.config
        }

if __name__ == "__main__":
    from api import MODEL_PATH, CONFIG_PATH
    pipeline = CLIPXRGenPipeline.from_pretrained(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
    )

    model = pipeline.model
    print("Same image encoder?", model.image_encoder is model.prompt_constructor.image_encoder)

    img_enc_param1 = next(model.image_encoder.parameters())
    prompt_img_enc_param1 = next(model.prompt_constructor.image_encoder.parameters())
    print("Same weights?", torch.equal(img_enc_param1, prompt_img_enc_param1))

