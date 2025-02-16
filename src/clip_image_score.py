from typing import Any, List, Union, TYPE_CHECKING

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics import Metric
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_10

if TYPE_CHECKING and _TRANSFORMERS_GREATER_EQUAL_4_10:
    from transformers import CLIPModel, CLIPProcessor


def _clip_image_score_update(
    real_images: Union[Tensor, List[Tensor]],
    fake_images: Union[Tensor, List[Tensor]],
    model: CLIPModel,  # pylint: disable=E0606
    processor: CLIPProcessor,  # pylint: disable=E0606
) -> tuple[Tensor, int]:
    if not isinstance(real_images, list):
        if real_images.ndim == 3:
            real_images = [real_images]
    else:  # unwrap into list
        real_images = list(real_images)

    if not isinstance(fake_images, list):
        if fake_images.ndim == 3:
            fake_images = [fake_images]
    else:  # unwrap into list
        fake_images = list(fake_images)
    if not all(i.ndim == 3 for i in real_images) or not all(i.ndim == 3 for i in fake_images):
        raise ValueError("Expected all images to be 3d but found image that has either more or less")

    if len(fake_images) != len(real_images):
        raise ValueError(f"Expected the sizes to be the same but got {len(real_images)} and {len(fake_images)}")

    device = real_images[0].device
    processed_input_real = processor(images=[i.cpu() for i in real_images], return_tensors="pt", padding=True)
    processed_input_fake = processor(images=[i.cpu() for i in fake_images], return_tensors="pt", padding=True)

    img_features_real = model.get_image_features(processed_input_real["pixel_values"].to(device))
    img_features_real = img_features_real / img_features_real.norm(p=2, dim=-1, keepdim=True)

    img_features_fake = model.get_image_features(processed_input_fake["pixel_values"].to(device))
    img_features_fake = img_features_fake / img_features_fake.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity between feature vectors
    score = 100 * (img_features_real * img_features_fake).sum(axis=-1)
    return score, len(real_images)


class CLIPImageScore(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound = 100.0

    score: Tensor
    n_samples: Tensor

    def __init__(
        self,
        model_name_or_path: Literal[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14-336",
            "openai/clip-vit-large-patch14",
        ] = "openai/clip-vit-large-patch14",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.processor = _get_clip_model_and_processor(model_name_or_path)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.score = None
        self.n_samples = None

    def update(
        self, real_images: Union[Tensor, List[Tensor]], fake_images: Union[Tensor, List[Tensor]]
    ):  # pylint: disable=W0221
        score, n_samples = _clip_image_score_update(real_images, fake_images, self.model, self.processor)
        if self.score is None:
            self.score = score.sum(0)
            self.n_samples = n_samples
        else:
            self.score += score.sum(0)
            self.n_samples += n_samples

    def compute(self) -> Tensor:
        """Compute accumulated clip score."""
        return torch.max(self.score / self.n_samples, torch.zeros_like(self.score))
