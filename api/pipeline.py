import logging
from pathlib import Path
from typing import Union, Optional, Dict, List, Any

import torch
from PIL import Image

from clipxrgen import build_model
from configs import load_config_from_file
from data import load_tokenizer, load_transform, transform_image

log = logging.getLogger(__name__)


class CLIPXRGenPipeline:
    def __init__(
            self,
            model,
            tokenizer,
            transform,
            prompt_constructor,
            device: torch.device,
            config: Dict[str, Any],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.transform = transform
        self.prompt_constructor = prompt_constructor

        self.device = device
        self.config = config

        self.model.eval().to(self.device)

    @classmethod
    def from_pretrained(
            cls,
            config_path: Union[str, Path],
            device: Union[str, torch.device] = "auto",
            **kwargs
    ) -> "CLIPXRGenPipeline":
        config_path = Path(config_path)

        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        log.info(f"Using device: {device}")

        if not config_path.is_file():
            log.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config = load_config_from_file(config_path)

        tokenizer = load_tokenizer(**config["tokenizer"])
        transform = load_transform(split="inference", transform_config=config["transform"])

        model = build_model(config, tokenizer)
        model = model.to(device)
        prompt_constructor = build_model(config["prompt_constructor"], tokenizer)

        return cls(
            model=model,
            tokenizer=tokenizer,
            transform=transform,
            prompt_constructor=prompt_constructor,
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
                prompts = self.prompt_constructor(
                    images=image_tensor,
                    labels=findings,
                    tokenizer=self.tokenizer,
                    device=self.device,
                )

                reports = self.model.generate(
                    images=image_tensor,
                    prompts=prompts,
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


