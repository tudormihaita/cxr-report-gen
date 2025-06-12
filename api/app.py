import io
import time
import logging
from typing import Optional

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import CLIPXRGenPipeline
from api import MODEL_PATH, CONFIG_PATH

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="VistaScan Report Generation integration API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReportResponse(BaseModel):
    report: str = Field(...,  description="Generated radiology report")
    success: bool = Field(..., description="Indicates if the report generation was successful")
    message: str = Field(..., description="Additional information or error message")
    processing_time: Optional[float] = Field(None, description="Time taken to process the request in seconds")


pipeline = CLIPXRGenPipeline.from_pretrained(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": pipeline.model is not None, "model_info": pipeline.model_info}


@app.post("/generate-report", response_model=ReportResponse)
async def generate_report(file: UploadFile = File(...)):
    start_time = time.time()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded or available"
        )

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        report = pipeline(image)

        processing_time = time.time() - start_time
        log.info(f"Generated report for {file.filename} in {processing_time:.2f} seconds")

        return ReportResponse(
            report=report,
            success=True,
            message="Report generated successfully",
            processing_time=processing_time
        )

    except Exception as e:
        log.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # update port if needed
        log_level="info"
    )