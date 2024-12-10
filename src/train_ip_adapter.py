import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import math
import wandb
import gc

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from tqdm import tqdm

from ip_adapter.ip_adapter import ImageProjModel, IPAdapter
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
from dataset import MyDataset, collate_fn
from accelerate.utils import set_seed
from PIL import Image
    

class IPAdapterModule(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def free_memory():
    """Runs garbage collection. Then clears the cache of the available accelerator."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_csv_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=-1,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--save_epochs",
        type=int,
        default=10,
        help=(
            "Save a checkpoint of the training state every X epochs"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ', `"wandb"` (default) and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="IP-Adapter-HSE",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="test_run",
    )
    parser.add_argument(
        "--prompt_image_path",
        type=str,
        default="",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def save_ip_adapter(args, accelerator, checkpoint_name):
    # save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    # model = accelerator.unwrap_model(ip_adapter)

    # # Словарное включение для извлечения нужных параметров
    # image_proj_sd = {k.replace("image_proj_model.", ""): v for k, v in model.state_dict().items() if k.startswith("image_proj_model")}
    # ip_sd = {k.replace("adapter_modules.", ""): v for k, v in model.state_dict().items() if k.startswith("adapter_modules")}
    # os.makedirs(save_path, exist_ok=True)
    # # Сохранение отфильтрованных состояний
    # print('!!!!!!!!!!!!!!!!!', model.image_proj_model.device)
    # torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, os.path.join(save_path, "ip_adapter.bin"))
    
    save_path = os.path.join(args.output_dir, checkpoint_name)
    accelerator.save_state(save_path, safe_serialization=False)

    model_path = os.path.join(save_path, "pytorch_model.bin")
    sd = torch.load(model_path, map_location="cpu")
    image_proj_sd = {}
    ip_sd = {}
    for k in sd:
        if k.startswith("unet"):
            pass
        elif k.startswith("image_proj_model"):
            image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
        elif k.startswith("adapter_modules"):
            ip_sd[k.replace("adapter_modules.", "")] = sd[k]

    checkpoint_path = os.path.join(save_path, "ip_adapter.bin")
    torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, checkpoint_path)
    os.remove(model_path)
    del sd, image_proj_sd, ip_sd


def training_epoch(args,
                   accelerator,
                   vae,
                   ip_adapter,
                   image_encoder,
                   text_encoder,
                   noise_scheduler,
                   optimizer,
                   train_dataloader,
                   weight_dtype,
                   progress_bar,
                   global_step):
    
    train_epoch_loss = 0.0
    train_epoch_batch_sum = 0.0
        
    for step, batch in enumerate(train_dataloader):

        train_step_loss = 0.0

        with accelerator.accumulate(ip_adapter):
            # Convert images to latent space
            with torch.no_grad():
                latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
            with torch.no_grad():
                image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
            image_embeds_ = []
            for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                if drop_image_embed == 1:
                    image_embeds_.append(torch.zeros_like(image_embed))
                else:
                    image_embeds_.append(image_embed)
            image_embeds = torch.stack(image_embeds_)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
            
            noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds)
    
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_step_loss += avg_loss.item() / 1

            train_epoch_loss += train_step_loss * args.train_batch_size
            train_epoch_batch_sum += args.train_batch_size
            
            # Backpropagate
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            # accelerator.log({"train Loss": train_step_loss}, step=global_step)
            global_step += 1
            logs = {"step_loss": train_step_loss}
            progress_bar.set_postfix(**logs)
    
    return train_epoch_loss / train_epoch_batch_sum, global_step
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    set_seed(2204)
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    #ip-adapter
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapterModule(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_csv_file, tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_project_name, config=args, init_kwargs={"wandb": {"name": args.wandb_run_name}})
    
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / 1)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    global_step = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        train_loss, global_step = training_epoch(args, accelerator, vae, ip_adapter, image_encoder, text_encoder, noise_scheduler, 
                                                 optimizer, train_dataloader, weight_dtype, progress_bar, global_step)
        accelerator.log({"Train Loss": train_loss}, step=epoch + 1)
        
        if accelerator.is_main_process:
            if epoch == 0 or ((epoch + 1) % args.save_epochs == 0) or (epoch == args.num_train_epochs - 1) or (args.save_steps != -1 and global_step % args.save_steps == 0): 
                checkpoint_name = f"checkpoint-{epoch + 1}"
                save_ip_adapter(args, accelerator, checkpoint_name=checkpoint_name)

                checkpoint_path = os.path.join(args.output_dir, checkpoint_name, "ip_adapter.bin")
                    
                pipeline = StableDiffusionPipeline.from_pretrained(
                        "stable-diffusion-v1-5/stable-diffusion-v1-5",
                        torch_dtype=torch.float16,
                        feature_extractor=None,
                        safety_checker=None
                    )
                pipeline.set_progress_bar_config(disable=True)
                ip_model = IPAdapter(pipeline, args.image_encoder_path, checkpoint_path, accelerator.device)
                prompt_image = Image.open("/home/chaichuk/IP-Adapter-repo/assets/images/ai_face.png")
                images = ip_model.generate(pil_image=prompt_image, num_samples=6, num_inference_steps=50, scale=1.0, height=args.resolution, width=args.resolution)
                anime_images = ip_model.generate(pil_image=prompt_image, num_samples=6, prompt=['anime style'], num_inference_steps=50, scale=0.7, height=args.resolution, width=args.resolution)
                log_dict = {"Image Validation": wandb.Image(image_grid(images, 2, 3), caption="Ai face"),
                            "Anime Image Validation": wandb.Image(image_grid(anime_images, 2, 3), caption="Anime Ai face")}

                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        tracker.log(log_dict)

                del pipeline, ip_model
                free_memory()

                
if __name__ == "__main__":
    main()    