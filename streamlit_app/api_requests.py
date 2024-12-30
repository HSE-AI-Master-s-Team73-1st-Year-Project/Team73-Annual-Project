import io
import aiohttp

from PIL import Image

API_URL = "http://0.0.0.0:8000/api/v1/ip_adapter"


async def generate_images(
    image_bytes: list[io.BytesIO], params: dict = None, prompts: list[str] = None, negative_prompts: list[str] = None
):
    """Generate images with IP-Adapter"""
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()

        for image in image_bytes:
            data.add_field('files', image, content_type='image/jpeg')

        if prompts is None:
            data.add_field('prompt', '')
        else:
            for prompt in prompts:
                data.add_field('prompt', prompt)

        if negative_prompts is None:
            data.add_field('negative_prompt', '')
        else:
            for prompt in negative_prompts:
                data.add_field('negative_prompt', prompt)

        if params is None:
            params = {}

        async with session.post(f"{API_URL}/generate_images/", params=params, data=data) as response:

            image_data = b''
            async for data in response.content.iter_any():
                image_data += data

            separator = b'--image--'
            images = image_data.split(separator)
            final_images = []

            for img_data in images:
                if img_data:
                    image = Image.open(io.BytesIO(img_data)).convert('RGB')
                    final_images.append(image)

            return final_images


async def load_new_adapter_checkpoint(file: io.BytesIO, adapter_id: str, description: str =None):
    """Change IP-Adapter checkpoint used in model"""

    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('file', file, content_type='application/octet-stream')

        params = {"id": adapter_id, "description": description}

        async with session.post(f"{API_URL}/generate_images/", data, params=params) as response:
            return await response.json()


async def change_adapter(adapter_id: str, description: str = None):
    """Change IP-Adapter checkpoint used in model"""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{API_URL}/change_adapter/", json={"id": adapter_id, "description": description}
        ) as response:
            return await response.json()


async def change_model(model_type: str):
    """Change StableDiffusion model type from anime to standard or vice versa"""

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_URL}/change_model/", json={"model_type": model_type}) as response:
            return await response.json()


async def get_adapters_list():
    """Get list of all available for inference IP Adapters"""

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/get_adapters_list/") as response:
            return await response.json()


async def get_models_list():
    """Get list of all available StableDiffusion types"""

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/get_models_list/") as response:
            return await response.json()


async def remove(model_id: str):
    """Remove IP Adapter checkpoint with id model_id"""

    async with aiohttp.ClientSession() as session:
        async with session.delete(f"{API_URL}/remove/{model_id}") as response:
            return await response.json()


async def remove_all():
    """Remove all loaded IP-Adapter checkpoints"""

    async with aiohttp.ClientSession() as session:
        async with session.delete(f"{API_URL}/remove_all/") as response:
            return await response.json()
