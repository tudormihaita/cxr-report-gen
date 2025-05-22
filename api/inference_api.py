import os
import io
import yaml
import torch

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import blip
from utils.logger import LoggerManager
from utils.transforms import TransformBuilder

logger = LoggerManager.get_logger(__name__)

app = FastAPI(title="Medical Report Generation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReportResponse(BaseModel):
    report: str
    success: bool
    message: str = ""


class ModelService:
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()

    def load_model(self):
        """Load the BLIP decoder model"""
        try:
            config_path = '../configs/generation.yaml'
            model_config = yaml.load(open(config_path), Loader=yaml.Loader)

            pretrained_path = os.path.join('../notebooks/output/generation', 'blip-decoder-gen-checkpoint-6000.pt')

            self.model = blip.blip_decoder(
                pretrained=pretrained_path,
                use_custom=True,
                image_size=model_config['image_size'],
                vit=model_config['vit'],
                vit_grad_ckpt=model_config['vit_grad_ckpt'],
                vit_ckpt_layer=model_config['vit_ckpt_layer'],
                prompt=model_config['prompt'],
            )
            self.model.eval()
            self.model = self.model.to(self.device)

            transform_builder = TransformBuilder(
                image_size=model_config['image_size'],
            )
            self.transform = transform_builder.build('test')

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess the uploaded image"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            if self.transform:
                image = self.transform(image)
            else:
                from torchvision import transforms
                basic_transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                image = basic_transform(image)

            image = image.unsqueeze(0).to(self.device)
            return image

        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")

    def generate_report(self, image: torch.Tensor) -> str:
        """Generate medical report from image"""
        try:
            with torch.no_grad():
                reports = self.model.generate(
                    image,
                    sample=False,
                    num_beams=3,
                    repetition_penalty=1.1
                )

                report = reports[0] if reports else "Unable to generate report."
                return report

        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate report: {str(e)}")


model_service = ModelService()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model_service.model is not None}


@app.post("/generate-report", response_model=ReportResponse)
async def generate_report(file: UploadFile = File(...)):
    """
    Generate a medical report from an uploaded chest X-ray image
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty image file"
            )

        image_tensor = model_service.preprocess_image(image_bytes)

        report = model_service.generate_report(image_tensor)

        logger.info(f"Successfully generated report for image: {file.filename}")

        return ReportResponse(
            report=report,
            success=True,
            message="Report generated successfully"
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during report generation"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # update port if needed
        log_level="info"
    )