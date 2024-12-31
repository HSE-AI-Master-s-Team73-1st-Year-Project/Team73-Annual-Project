import io
import aiohttp

from PIL import Image

API_URL = "http://localhost:8000/api/v1/ip_adapter"


class RepeatableFormData(aiohttp.FormData):
    def __call__(self):
        if self._is_multipart:
            if self._is_processed:
                return self._writer

            return self._gen_form_data()
        return self._gen_form_urlencoded()


async def generate_images(
    images, params: dict = None, prompts: list[str] = None, negative_prompts: list[str] = None
):
    """Generate images with IP-Adapter"""
    async with aiohttp.ClientSession() as session:
        form_data = RepeatableFormData()

        for image in images:
            form_data.add_field('files', image.read(), filename=image.name, content_type=image.type)

        if prompts is not None:
            for prompt in prompts:
                form_data.add_field('prompt', prompt)

        if negative_prompts is not None:
            for prompt in negative_prompts:
                form_data.add_field('negative_prompt', prompt)

        if params is None:
            params = {}

        async with session.post(f"{API_URL}/generate_images/", params=params, data=form_data) as response:
            if response.status != 200:
                result = await response.json()
                return {'code': response.status, 'result': result}

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

            return {'code': 200, 'result': final_images}


async def load_new_adapter_checkpoint(file, adapter_id: str, description: str = None):
    """Change IP-Adapter checkpoint used in model"""
    async with aiohttp.ClientSession() as session:
        form_data = RepeatableFormData()
        form_data.add_field('file', file.read(), content_type=file.type)

        params = {"id": adapter_id, "description": description}
        async with session.post(f"{API_URL}/load_new_adapter_checkpoint/", params=params, data=form_data) as response:
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


async def get_available_adapter_checkpoints():
    """Get list of all available for inference IP Adapters"""

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/get_available_adapter_checkpoints/") as response:
            return await response.json()


async def get_available_model_types():
    """Get list of all available StableDiffusion types"""

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/get_available_model_types/") as response:
            return await response.json()


async def get_current_model_type():
    """Get type of a current StableDiffusion model"""

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/get_current_model_type/") as response:
            return await response.json()


async def remove(model_id: str):
    """Remove IP Adapter checkpoint with id model_id"""

    async with aiohttp.ClientSession() as session:
        async with session.delete(f"{API_URL}/remove_adapter_checkpoint/{model_id}") as response:
            return await response.json()


async def remove_all():
    """Remove all loaded IP-Adapter checkpoints"""

    async with aiohttp.ClientSession() as session:
        async with session.delete(f"{API_URL}/remove_all/") as response:
            return await response.json()
