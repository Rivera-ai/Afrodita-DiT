import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from ModelDiT2 import DiT_models
from Diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from Text import get_bert, get_tokenizer, tokenize, bert_embed

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)

#################################################################################
#                                  Training Loop                                #
#################################################################################

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, max_length=77):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = get_tokenizer()
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.transform(item['image'].convert('RGB'))
        caption = item['caption']
        
        tokens = self.tokenizer.encode(caption, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        tokens = torch.tensor(tokens)
        
        return {
            'image': image,
            'caption': caption,
            'tokens': tokens
        }

def main(args):
    # Setup
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        text_embed_dim=768
    ).to(device)
    ema = deepcopy(model)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    bert_model = get_bert().to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = load_dataset("tungdop2/nsfw_w_caption", split="train")
    dataset = CustomDataset(dataset, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch}")
        for step, batch in enumerate(loader):
            x = batch['image'].to(device)
            tokens = batch['tokens'].to(device)
            
            with torch.no_grad():
                # Encode images to latent space
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                # Get text embeddings
                text_embeds = bert_embed(tokens)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            
            # Forward pass
            model_kwargs = {'y': torch.zeros(x.shape[0], dtype=torch.long, device=device), 'text_embed': text_embeds}
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs=model_kwargs)
            loss = loss_dict["loss"].mean()
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Logging
            if step % args.log_every == 0:
                logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")

        # Save checkpoint
        if (epoch + 1) % args.ckpt_every == 0:
            checkpoint = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args
            }
            checkpoint_path = f"{checkpoint_dir}/epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/8")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10)
    args = parser.parse_args()
    main(args)