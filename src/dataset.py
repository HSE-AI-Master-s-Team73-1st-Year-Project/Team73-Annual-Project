import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import torch
import os

from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from tqdm import tqdm


class MyDataset(Dataset):

    def __init__(self, csv_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        data = pd.read_csv(csv_file)

        self.tokenized_texts = {}
        self.image_files = data['ID'].tolist()

        for i in tqdm(range(10), desc='tokenizing texts'):
            texts = data[f'caption_{i}'].tolist()

            self.tokenized_texts[i] = self.tokenizer(
                texts,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids

        self.empty_text = self.tokenizer("",
                                         max_length=self.tokenizer.model_max_length,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt"
                                         ).input_ids.squeeze()

        self.transform = T.Compose([
            T.Resize((self.size, self.size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        text_column = np.random.choice(10)
        text_input_ids = self.tokenized_texts[text_column][idx]
        
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
        drop_image_embed = 0
        rand_num = np.random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text_input_ids = self.empty_text
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text_input_ids = self.empty_text
            drop_image_embed = 1
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed
        }

    def __len__(self):
        return len(self.image_files)


class MetricsDataset(Dataset):

    def __init__(self,
                 faces_csv_file='/home/chaichuk/datasets/CelebAMask-HQ/new_celeba_captions.csv', 
                 anime_faces_csv_file='/home/chaichuk/datasets/anime_faces/images.csv', 
                 anime_faces_path='/home/chaichuk/datasets/anime_faces/images',
                 faces_pictures_path='/home/chaichuk/datasets/CelebAMask-HQ/CelebA-HQ-img',
                 faces_caption_col='blip2_caption',
                 num_samples=2000,
                 size=512,
                 random_seed=2204):
        super().__init__()

        self.size = size
        self.num_samples = num_samples
        self.anime_faces_path = anime_faces_path
        self.faces_pictures_path = faces_pictures_path

        faces_data = pd.read_csv(faces_csv_file)
        anime_faces_data = pd.read_csv(anime_faces_csv_file)

        self.anime_images = anime_faces_data.sample(num_samples, random_state=random_seed)['ID'].tolist()
        faces_samples = faces_data.sample(num_samples, random_state=random_seed)
        self.faces_images = faces_samples['ID'].tolist()
        self.faces_texts = faces_samples[faces_caption_col].tolist()

        self.faces_pil_transform = T.Compose([
            T.Resize((self.size, self.size)),
        ])
        self.faces_tensor_transform = T.Compose([
            T.Resize((self.size, self.size)),
            T.ToTensor(),
        ])
        self.anime_faces_transform = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        face_image_file = self.faces_images[idx]
        anime_image_file = self.anime_images[idx]
        text = self.faces_texts[idx]
        
        face_image = Image.open(os.path.join(self.faces_pictures_path, face_image_file)).convert("RGB")
        anime_image = Image.open(os.path.join(self.anime_faces_path, anime_image_file)).convert("RGB")        
        
        return {
            "pil_face_image": self.faces_pil_transform(face_image),
            "pt_face_image": self.faces_tensor_transform(face_image),
            'anime_image': self.anime_faces_transform(anime_image),
            "text": 'one person face - ' + text + ', hand-drawn anime style'
        }

    def __len__(self):
        return self.num_samples


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.stack([example["text_input_ids"] for example in data])
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds
    }