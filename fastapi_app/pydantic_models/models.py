from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel


class DeviceType(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


class ImageGenerationRequest(BaseModel):
    prompt: Optional[list[str]] = None
    negative_prompt: Optional[list[str]] = None
    scale: float = 0.6
    num_samples: int = 1
    random_seed: int = 42
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    device: DeviceType = "cuda"


class LoadAdapterRequest(BaseModel):
    id: str
    description: Optional[str] = None


class LoadAdapterResponse(BaseModel):
    message: str


class ChangeAdapterRequest(BaseModel):
    id: str


class ChangeAdapterResponse(BaseModel):
    message: str


class ModelType(str, Enum):
    STANDARD = "standard"
    ANIME = "anime"


class ChangeModelRequest(BaseModel):
    model_type: ModelType


class ChangeModelResponse(BaseModel):
    message: str


class ModelListResponse(BaseModel):
    models: list[Dict]


class RemoveResponse(BaseModel):
    message: str
