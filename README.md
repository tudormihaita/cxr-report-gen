# CLIP-XRGen

**Weakly Supervised Multimodal Vision-Language Model for Automated Radiology Report Generation using Contrastive Vision-Language Pretraining.**

Official implementation of the thesis "CLIP-XRGen: A Contrastive Learning approach for Automated Radiology Report Generation with Medical Concept Alignment" and **TBD** paper.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

## Overview

CLIP-XRGen is structured similarly to CXR-CLIP, using config files and factory functions to instantiate models for both training and inference. The implementation includes fine-tuned adaptations of baseline CLIP and BLIP models for image-text retrieval performance comparisons.

### Three-Stage Training Pipeline

#### 1. **Vision-Language Pre-training** (`clip.py` + `pretrain.py`)
- Contrastive learning between chest X-ray images and radiology reports
- Learns joint vision-language representations with medical concept alignment using custom MCSL loss objective
- Foundation for downstream tasks

#### 2. **Supervised Classification** (`classifier.py` + `finetune.py`) 
- Fine-tunes vision encoder for medical finding classification
- Learns to predict CheXpert labels from X-ray images
- Enables automated finding detection

#### 3. **Report Generation** (`decoder.py` + `downstream.py` + `prompt_constructor.py`)
- Combines vision encoder with transformer-based text decoder
- Uses prompt constructor for providing additional context to report generation
- Generates coherent clinical reports from visual findings

## Available Models

- [CLIP-XRad Pretrained Multimodal Dual-Encoder](https://huggingface.co/tudormihaita/clip-xrad-pretrained-mscl)
- [CLIP-XRad Supervised Classifier](https://huggingface.co/tudormihaita/clip-xrad-finetuned-classifier) 
- [CLIP-XRGen Report Generation Encoder-Decoder](https://huggingface.co/tudormihaita/clip-xrgen)

### Model Usage Example

```python
from api.pipeline import CLIPXRGenPipeline

# Load the model
pipeline = CLIPXRGenPipeline.from_pretrained(
    config_path="configs/model_config.yaml"
)

# Generate report
report = pipeline("path/to/chest_xray.jpg")
print(f"Generated Report: {report}")
```

### API Server

```bash
# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Test the API
curl -X POST "http://localhost:8000/generate-report" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@chest_xray.jpg"
```