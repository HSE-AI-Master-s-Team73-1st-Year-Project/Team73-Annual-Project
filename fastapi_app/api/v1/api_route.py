import io
import os
import logging
from logging.handlers import RotatingFileHandler

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

LOG_DIR = "/home/chaichuk/Team73-Annual-Project/logs"


logger = logging.getLogger("fastapi_logger")
logger.setLevel(logging.DEBUG)

os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "fastapi_app.log")
log_handler = RotatingFileHandler(log_file, maxBytes=32 * 1024 * 1024, backupCount=5)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)

logger.addHandler(log_handler)


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

    logger.info("Application started.")

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
    logger.info("Application stopped.")


router = APIRouter(lifespan=lifespan)


@router.post(
    "/generate_images",
    status_code=HTTPStatus.OK,
)
async def generate_images(request: ImageGenerationRequest = Depends(), files: List[UploadFile] = File(...)):
    """Generate images with IP Adapter"""

    logger.info("GENERATE_IMAGES. Request.")
    logger.debug("GENERATE_IMAGES. Request with params: %s.", request)

    if request.device == "cuda" and not config["cuda_available"]:
        logger.error("GENERATE_IMAGES. ERROR: CUDA is not available.")
        raise HTTPException(status_code=422, detail="CUDA is not available")

    if request.device != config["current_device"]:
        logger.info("GENERATE_IMAGES. Changing device to %s", request.device)
        config["current_device"] = request.device
        config["ip_adapter"].device = request.device
        config["ip_adapter"].image_encoder = config["ip_adapter"].image_encoder.to(request.device)
        config["ip_adapter"].image_proj_model = config["ip_adapter"].image_proj_model.to(request.device)
        config["ip_adapter"].pipe = config["ip_adapter"].pipe.to(request.device)

    image_prompts = []

    logger.info("GENERATE_IMAGES. Reading Image files.")

    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB').resize((512, 512))
        image_prompts.append(image)

    logger.info("GENERATE_IMAGES. Image files processed successfully.")

    try:
        logger.info("GENERATE_IMAGES. Starting generation.")

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

        logger.info("GENERATE_IMAGES. Images generated successfully.")

        async def image_stream():
            for image in generated_images:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)

                yield buffered.getvalue()
                yield b"--image--"

        return StreamingResponse(image_stream(), media_type="image/png")

    except Exception as e:
        logger.error("GENERATE_IMAGES. ERROR while generating images: %s.", str(e))
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/change_model", response_model=ChangeModelResponse, status_code=HTTPStatus.OK)
async def change_model(request: ChangeModelRequest):
    """Change StableDiffusion model type from anime to standard or vice versa"""

    logger.info("CHANGE_MODEL. Request.")
    logger.debug("CHANGE_MODEL. Request with params: %s.", request)

    if request.model_type == config["sd_version"]:
        logger.info("CHANGE_MODEL. Model type is already %s.", request.model_type)
        return ChangeModelResponse(message=f"Model type is already {request.model_type}")

    try:
        logger.info("CHANGE_MODEL. Changing model type to %s.", request.model_type)
        logger.info("CHANGE_MODEL. Loading %s pipeline.", request.model_type)
        config["pipeline"] = StableDiffusionPipeline.from_pretrained(
            SD_ANIME_MODEL_PATH if request.model_type == "anime" else SD_STANDARD_MODEL_PATH,
            torch_dtype=torch.float16,
            scheduler=config["scheduler"],
            vae=config["vae"],
            feature_extractor=None,
            safety_checker=None,
        )
        logger.info("CHANGE_MODEL. %s pipeline successfully loaded.", request.model_type)

        config["pipeline"].set_progress_bar_config(disable=True)

        logger.info("CHANGE_MODEL. Loading IP-Adapter %s into new model.", config["current_adapter"])
        config["ip_adapter"] = IPAdapter(
            config["pipeline"],
            IMAGE_ENCODER_PATH,
            config["adapters_list"][config["current_adapter"]]["path"],
            config["current_device"],
        )

    except Exception as e:
        logger.error("CHANGE_MODEL. ERROR while loading model pipeline: %s.", str(e))
        raise HTTPException(status_code=422, detail=str(e)) from e

    logger.info("CHANGE_MODEL. IP-Adapter %s successfully loaded into new model.", config["current_adapter"])
    return ChangeModelResponse(message=f"Model type successfully changed to {request.model_type}")


@router.post("/change_adapter", response_model=ChangeAdapterResponse, status_code=HTTPStatus.OK)
async def change_adapter(request: ChangeAdapterRequest):
    """Change IP Adapter checkpoint used in model"""

    logger.info("CHANGE_ADAPTER. Request.")
    logger.debug("CHANGE_ADAPTER. Request with params: %s", request)
    if request.id not in config["adapters_list"]:
        logger.error("CHANGE_ADAPTER. ERROR. Adapter not found: %s.", request.id)
        raise HTTPException(status_code=422, detail=f"IP-Adapter {request.id} not found")

    if request.id == config["current_adapter"]:
        logger.info('CHANGE_ADAPTER. Adapter %s is already in use.')
        return ChangeAdapterResponse(message=f"IP-Adapter {request.id} is already in use")

    try:
        logger.info('CHANGE_ADAPTER. Loading %s adapter.', request.id)

        config["ip_adapter"] = IPAdapter(
            config["pipeline"],
            IMAGE_ENCODER_PATH,
            config["adapters_list"][request.id]["path"],
            config["current_device"],
        )

    except Exception as e:
        logger.error('CHANGE_ADAPTER. ERROR while loading IP-Adapter: %s.', str(e))
        raise HTTPException(status_code=422, detail=str(e)) from e

    logger.info('CHANGE_ADAPTER. Adapter %s successfully loaded.', request.id)
    return ChangeAdapterResponse(message=f"IP-Adapter successfully changed to {request.id}")


@router.post("/load_new_adapter_checkpoint", response_model=LoadAdapterResponse, status_code=HTTPStatus.OK)
async def load_new_adapter_checkpoint(data: LoadAdapterRequest = Depends(), file: UploadFile = File(...)):
    """Change IP Adapter checkpoint used in model"""

    logger.info("LOAD_NEW_ADAPTER_CHECKPOINT. Request.")
    logger.debug("LOAD_NEW_ADAPTER_CHECKPOINT. Request with params: %s.", data)

    if data.id in config["adapters_list"]:
        logger.error("LOAD_NEW_ADAPTER_CHECKPOINT. ERROR. IP-Adapter %s is already exists.")
        raise HTTPException(status_code=422, detail=f"IP Adapter {data.id} is already exists")

    file_save_path = f"{TEMPORARY_CHECKPOINTS_PATH}/{data.id}.bin"

    try:
        logger.info("LOAD_NEW_ADAPTER_CHECKPOINT. Loading checkpoint.")
        async with aiofiles.open(file_save_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
    except Exception as e:
        logger.info("LOAD_NEW_ADAPTER_CHECKPOINT. ERROR while loading checkpoint file: %s.", str(e))
        raise HTTPException(status_code=422, detail=str(e)) from e

    config["adapters_list"][data.id] = {
        "description": data.description if data.description is not None else f"{data.id} checkpoint",
        "path": file_save_path,
        "preloaded": False,
    }
    logger.info("LOAD_NEW_ADAPTER_CHECKPOINT. Checkpoint successfully saved into %s.", file_save_path)
    logger.debug("LOAD_NEW_ADAPTER_CHECKPOINT. Checkpoint saved with params: %s.", config["adapters_list"][data.id])
    return ChangeAdapterResponse(message=f"IP Adapter {data.id} loaded successfully")


@router.get("/get_adapters_list", response_model=ModelListResponse, status_code=HTTPStatus.OK)
async def get_adapters_list():
    """Get list of all available for inference IP Adapters"""

    logger.info("GET_ADAPTERS_LIST. Request.")
    adapters_list = [
        {"id": adapter_id, "description": config["adapters_list"][adapter_id]["description"]}
        for adapter_id in config["adapters_list"]
    ]
    return ModelListResponse(models=adapters_list)


@router.get("/get_models_list", response_model=ModelListResponse, status_code=HTTPStatus.OK)
async def get_models_list():
    """Get list of all available StableDiffusion types"""

    logger.info("GET_MODELS_LIST. Request.")
    models_list = [
        {"id": "anime", "description": "StableDiffusion-v1-5 version finetuned for better anime style generation"},
        {"id": "standard", "description": "StableDiffusion-v1-5 basic version"},
    ]
    return ModelListResponse(models=models_list)


@router.delete("/remove_adapter_checkpoint/{model_id}", response_model=RemoveResponse, status_code=HTTPStatus.OK)
async def remove(model_id: str):
    """Remove IP Adapter checkpoint with id model_id"""

    logger.info("REMOVE_ADAPTER_CHECKPOINT. Request.")
    logger.debug("GET_ADAPTERS_LIST. Requestfor id %s.", model_id)
    if model_id not in config["adapters_list"]:
        logger.error("REMOVE_ADAPTER_CHECKPOINT. ERROR. IP-Adapter %s not found.", model_id)
        raise HTTPException(status_code=422, detail=f"Model '{model_id}' not found")

    if model_id in config["adapters_list"] and config["adapters_list"][model_id]["preloaded"]:
        logger.error("REMOVE_ADAPTER_CHECKPOINT. ERROR. IP-Adapter %s cannot be removed.", model_id)
        raise HTTPException(status_code=422, detail=f"Model '{model_id}' cannot be removed")

    logger.info("REMOVE_ADAPTER_CHECKPOINT. Removing adapter %s.", model_id)
    
    os.remove(config["adapters_list"][model_id]["path"])
    del config["adapters_list"][model_id]

    if model_id == config["current_adapter"]:
        logger.info("REMOVE_ADAPTER_CHECKPOINT. Switching to default adapter.")
        try:
            config["ip_adapter"] = IPAdapter(
                config["pipeline"],
                IMAGE_ENCODER_PATH,
                config["adapters_list"][DEFAULT_CHECKPOINT_PATH]["path"],
                config["current_device"],
            )
        except Exception as e:
            logger.error("REMOVE_ADAPTER_CHECKPOINT. ERROR while loading default adapter %s.", str(e))
            raise HTTPException(status_code=422, detail=str(e)) from e
        logger.info("REMOVE_ADAPTER_CHECKPOINT. Adapter %s successfully removed.", model_id)
        return RemoveResponse(message=f"Adapter {model_id} removed, ip adapter changed to default")

    logger.info("REMOVE_ADAPTER_CHECKPOINT. Adapter %s successfully removed.", model_id)
    return RemoveResponse(message=f"Adapter {model_id} removed")


@router.delete("/remove_all", response_model=list[RemoveResponse], status_code=HTTPStatus.OK)
async def remove_all():
    """Remove all loaded IP Adapter checkpoints"""

    logger.info("REMOVE_ALL. Request.")
    logger.info("REMOVE_ALL. Removing all loaded checkpoints.")

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
            logger.info("REMOVE_ALL. Switching to default adapter.")

            config["ip_adapter"] = IPAdapter(
                config["pipeline"],
                IMAGE_ENCODER_PATH,
                config["adapters_list"][DEFAULT_CHECKPOINT_PATH]["path"],
                config["current_device"],
            )
        except Exception as e:
            logger.error("REMOVE_ALL. ERROR while loading default adapter %s.", str(e))
            raise HTTPException(status_code=422, detail=str(e)) from e

    logger.info("REMOVE_ALL. All loaded checkpoints successfully removed.")
    return responses
