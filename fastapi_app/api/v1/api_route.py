import io
import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import List

import aiofiles
import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from starlette.responses import StreamingResponse
from PIL import Image
from pydantic_models.models import (ChangeAdapterRequest,
                                    ChangeAdapterResponse, ChangeModelRequest,
                                    ChangeModelResponse,
                                    ImageGenerationRequest,
                                    LoadAdapterRequest, LoadAdapterResponse,
                                    ModelListResponse, RemoveResponse)
from ip_adapter import IPAdapter

DEFAULT_CHECKPOINT_PATH = "512_res_model_checkpoint_100"
PRELOADED_CHECKPOINTS_PATH = "/home/chaichuk/Team73-Annual-Project/checkpoints"
TEMPORARY_CHECKPOINTS_PATH = "/home/chaichuk/Team73-Annual-Project/tmp_checkpoints"
IMAGE_ENCODER_PATH = "/home/chaichuk/Team73-Annual-Project/models/image_encoder"
VAE_MODEL_PATH = "stabilityai/sd-vae-ft-mse"
SD_STANDARD_MODEL_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD_ANIME_MODEL_PATH = "dreamlike-art/dreamlike-anime-1.0"

config = {
    "scheduler": None,
    "vae": None,
    "pipeline": None,
    "ip_adapter": None,
    "sd_version": None,
    "adapters_list": {},
    "current_adapter": None,
    "current_device": None,
    "cuda_available": None,
}


@asynccontextmanager
async def lifespan(lifespan_router: APIRouter):  # pylint: disable=unused-argument
    """lifespan function for router"""

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    vae = AutoencoderKL.from_pretrained(VAE_MODEL_PATH).to(dtype=torch.float16)

    pipeline = StableDiffusionPipeline.from_pretrained(
        SD_ANIME_MODEL_PATH,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )

    pipeline.set_progress_bar_config(disable=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(TEMPORARY_CHECKPOINTS_PATH, exist_ok=True)

    available_checkpoints = os.listdir(PRELOADED_CHECKPOINTS_PATH)

    basic_ckpt_path = f"{PRELOADED_CHECKPOINTS_PATH}/{available_checkpoints[0]}/ip_adapter.bin"

    ip_model = IPAdapter(pipeline, IMAGE_ENCODER_PATH, basic_ckpt_path, device)

    config["scheduler"] = noise_scheduler
    config["vae"] = vae
    config["pipeline"] = pipeline
    config["sd_version"] = "anime"
    config["ip_adapter"] = ip_model
    config["adapters_list"] = {
        adapter_name: {
            "description": f"preloaded checkpoint {i}",
            "path": f"{PRELOADED_CHECKPOINTS_PATH}/{adapter_name}/ip_adapter.bin",
            "preloaded": True,
        }
        for i, adapter_name in enumerate(available_checkpoints)
    }
    config["current_adapter"] = available_checkpoints[0]
    config["current_device"] = device
    config["cuda_available"] = torch.cuda.is_available()

    yield

    del config["scheduler"]
    del config["vae"]
    del config["pipeline"]
    del config["ip_adapter"]

    for adapter_id in config["adapters_list"]:
        if not config["adapters_list"][adapter_id]["preloaded"]:
            os.remove(config["adapters_list"][adapter_id]["path"])

    config.clear()


router = APIRouter(lifespan=lifespan)


@router.post(
    "/generate_images",
    status_code=HTTPStatus.OK,
)
async def generate_images(request: ImageGenerationRequest = Depends(), files: List[UploadFile] = File(...)):
    """Generate images with IP Adapter"""

    if request.device == "cuda" and not config["cuda_available"]:
        raise HTTPException(status_code=422, detail="CUDA is not available")

    if request.device != config["current_device"]:
        config["current_device"] = request.device
        config["ip_adapter"].device = request.device
        config["ip_adapter"].image_encoder = config["ip_adapter"].image_encoder.to(request.device)
        config["ip_adapter"].image_proj_model = config["ip_adapter"].image_proj_model.to(request.device)
        config["ip_adapter"].pipe = config["ip_adapter"].pipe.to(request.device)

    image_prompts = []

    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB').resize((512, 512))
        image_prompts.append(image)

    try:
        generated_images = config["ip_adapter"].generate(
            pil_image=image_prompts,
            prompt=request.prompt if len(request.prompt) > 1 else request.prompt[0],
            negative_prompt=request.negative_prompt if len(request.negative_prompt) > 1 else request.negative_prompt[0],
            num_samples=request.num_samples,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            scale=request.scale,
            guidance_scale=request.guidance_scale,
            seed=request.random_seed,
        )

        async def image_stream():
            for image in generated_images:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)

                yield buffered.getvalue()
                yield b"--image--"

        return StreamingResponse(image_stream(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/change_model", response_model=ChangeModelResponse, status_code=HTTPStatus.OK)
async def change_model(request: ChangeModelRequest):
    """Change StableDiffusion model type from anime to standard or vice versa"""

    if request.model_type == config["sd_version"]:
        return ChangeModelResponse(message=f"Model type is already {request.model_type}")

    try:
        config["pipeline"] = StableDiffusionPipeline.from_pretrained(
            SD_ANIME_MODEL_PATH if request.model_type == "anime" else SD_STANDARD_MODEL_PATH,
            torch_dtype=torch.float16,
            scheduler=config["scheduler"],
            vae=config["vae"],
            feature_extractor=None,
            safety_checker=None,
        )

        config["pipeline"].set_progress_bar_config(disable=True)
        config["ip_adapter"] = IPAdapter(
            config["pipeline"],
            IMAGE_ENCODER_PATH,
            config["adapters_list"][config["current_adapter"]]["path"],
            config["current_device"],
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return ChangeModelResponse(message=f"Model type successfully changed to {request.model_type}")


@router.post("/change_adapter", response_model=ChangeAdapterResponse, status_code=HTTPStatus.OK)
async def change_adapter(request: ChangeAdapterRequest):
    """Change IP Adapter checkpoint used in model"""

    if request.id not in config["adapters_list"]:
        raise HTTPException(status_code=422, detail=f"IP Adapter {request.id} not found")

    if request.id == config["current_adapter"]:
        return ChangeAdapterResponse(message=f"IP Adapter {request.id} is already in use")

    try:
        config["ip_adapter"] = IPAdapter(
            config["pipeline"],
            IMAGE_ENCODER_PATH,
            config["adapters_list"][request.id]["path"],
            config["current_device"],
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return ChangeAdapterResponse(message=f"IP Adapter successfully changed to {request.id}")


@router.post("/load_new_adapter_checkpoint", response_model=LoadAdapterResponse, status_code=HTTPStatus.OK)
async def load_new_adapter_checkpoint(data: LoadAdapterRequest = Depends(), file: UploadFile = File(...)):
    """Change IP Adapter checkpoint used in model"""

    if data.id in config["adapters_list"]:
        raise HTTPException(status_code=422, detail=f"IP Adapter {data.id} is already exists")

    file_save_path = f"{TEMPORARY_CHECKPOINTS_PATH}/{data.id}.bin"

    try:
        async with aiofiles.open(file_save_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    config["adapters_list"][data.id] = {
        "description": data.description if data.description is not None else f"{data.id} checkpoint",
        "path": file_save_path,
        "preloaded": False,
    }

    return ChangeAdapterResponse(message=f"IP Adapter {data.id} loaded successfully")


@router.get("/get_adapters_list", response_model=ModelListResponse, status_code=HTTPStatus.OK)
async def get_adapters_list():
    """Get list of all available for inference IP Adapters"""

    adapters_list = [
        {"id": adapter_id, "description": config["adapters_list"][adapter_id]["description"]}
        for adapter_id in config["adapters_list"]
    ]
    return ModelListResponse(models=adapters_list)


@router.get("/get_models_list", response_model=ModelListResponse, status_code=HTTPStatus.OK)
async def get_models_list():
    """Get list of all available StableDiffusion types"""

    models_list = [
        {"id": "anime", "description": "StableDiffusion-v1-5 version finetuned for better anime style generation"},
        {"id": "standard", "description": "StableDiffusion-v1-5 basic version"},
    ]
    return ModelListResponse(models=models_list)


@router.delete("/remove_adapter_checkpoint/{model_id}", response_model=RemoveResponse, status_code=HTTPStatus.OK)
async def remove(model_id: str):
    """Remove IP Adapter checkpoint with id model_id"""

    if model_id not in config["adapters_list"]:
        raise HTTPException(status_code=422, detail=f"Model '{model_id}' not found")

    if model_id in config["adapters_list"] and config["adapters_list"][model_id]["preloaded"]:
        raise HTTPException(status_code=422, detail=f"Model '{model_id}' cannot be removed")

    del config["adapters_list"][model_id]
    os.remove(config["adapters_list"][model_id]["path"])

    if model_id == config["current_adapter"]:
        try:
            config["ip_adapter"] = IPAdapter(
                config["pipeline"],
                IMAGE_ENCODER_PATH,
                config["adapters_list"][DEFAULT_CHECKPOINT_PATH]["path"],
                config["current_device"],
            )
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        return RemoveResponse(message=f"Adapter {model_id} removed, ip adapter changed to default")

    return RemoveResponse(message=f"Adapter {model_id} removed")


@router.delete("/remove_all", response_model=list[RemoveResponse], status_code=HTTPStatus.OK)
async def remove_all():
    """Remove all loaded IP Adapter checkpoints"""
    responses = []
    loaded_adapter_deleted = False
    for model_id in config["adapters_list"]:
        if config["adapters_list"][model_id]["preloaded"]:
            continue
        if model_id == config["current_adapter"]:
            loaded_adapter_deleted = True
        os.remove(config["adapters_list"][model_id]["path"])
        del config["adapters_list"][model_id]
        responses.append(RemoveResponse(message=f"Adapter {model_id} removed"))

    if loaded_adapter_deleted:
        try:
            config["ip_adapter"] = IPAdapter(
                config["pipeline"],
                IMAGE_ENCODER_PATH,
                config["adapters_list"][DEFAULT_CHECKPOINT_PATH]["path"],
                config["current_device"],
            )
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return responses
