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

from AfroditaDiT import DiT_models
from Diffusion import create_diffusion
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
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

class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, max_length=77):
        # Convertir el dataset si es necesario
        if isinstance(dataset, dict):
            self.dataset = dataset['test']
        else:
            self.dataset = dataset
            
        print(f"Inicializando dataset con {len(self.dataset)} muestras")  # Debug print
        self.transform = transform
        self.tokenizer = get_tokenizer()
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            # Debug print para el primer item
            if idx == 0:
                print(f"Ejemplo de item: {item.keys()}")
                
            image = self.transform(item['image'].convert('RGB'))
            # Si hay múltiples captions, toma el primero
            caption = item['caption'][0] if isinstance(item['caption'], list) else str(item['caption'])

            tokens = self.tokenizer.encode(caption, 
                                        add_special_tokens=True, 
                                        max_length=self.max_length, 
                                        padding='max_length', 
                                        truncation=True)
            tokens = torch.tensor(tokens)

            return {
                'image': image,
                'caption': caption,
                'tokens': tokens
            }
        except Exception as e:
            print(f"Error procesando item {idx}: {str(e)}")
            raise e

@torch.no_grad()
def generate_and_save_images(model, vae, diffusion, device, epoch, step, save_dir, batch):
    """
    Generate images using the model and save them along with real images
    """
    model.eval()

    # Create directory for this checkpoint
    save_path = os.path.join(save_dir, f"samples/epoch_{epoch}_step_{step}")
    os.makedirs(save_path, exist_ok=True)

    # Get a real image from the batch to save as reference
    real_images = batch['image'][:4]  # Save first 4 images

    # Save real images
    for idx, img in enumerate(real_images):
        # Denormalize the images
        img = img * 0.5 + 0.5  # from [-1, 1] to [0, 1]
        img = (img * 255).clamp(0, 255).to(torch.uint8)
        img = transforms.ToPILImage()(img)
        img.save(f"{save_path}/real_{idx}.png")

    # Generate latent noise
    noise = torch.randn(4, 4, 32, 32).to(device)  # Adjust size based on your VAE

    # Get text embeddings for generation
    tokens = batch['tokens'][:4].to(device)
    with torch.no_grad():
        text_embeds = bert_embed(tokens)

    # Sample images
    model_kwargs = {
        'y': torch.zeros(4, dtype=torch.long, device=device),
        'text_embed': text_embeds
    }

    # Generate images using ddim_sample_loop
    samples = diffusion.ddim_sample_loop(
        model=model,
        shape=noise.shape,
        noise=noise,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
        eta=0.0  # Deterministic DDIM sampling
    )

    # Decode the latents to images
    with torch.no_grad():
        samples = 1 / 0.18215 * samples
        samples = vae.decode(samples).sample

    # Save generated images
    for idx, img in enumerate(samples):
        # Denormalize and convert to PIL
        img = (img * 0.5 + 0.5).clamp(0, 1)
        img = transforms.ToPILImage()(img.cpu())
        img.save(f"{save_path}/generated_{idx}.png")


    return save_path

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
    print("Cargando dataset...")
    dataset = load_dataset("nlphuji/flickr30k")
    print(f"Dataset cargado. Estructura: {dataset}")
    print(f"Llaves disponibles: {dataset.keys()}")

    print("Creando CustomDataset...")
    dataset = Flickr30kDataset(dataset, transform)
    print(f"Tamaño del dataset: {len(dataset)}")
    print("Creando DataLoader...")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print("DataLoader creado")
    logger.info(f"Dataset contains {len(dataset):,} images")

    # Training loop
    model.train()
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch}")
        print(f"\nBeginning epoch {epoch + 1}/{args.epochs}...")
        for step, batch in enumerate(loader):
            print(f"Epoch {epoch}, Step {step}", flush=True)
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
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")

                save_path = generate_and_save_images(
                    model=model,
                    vae=vae,
                    diffusion=diffusion,
                    device=device,
                    epoch=epoch,
                    step=step,
                    save_dir=experiment_dir,
                    batch=batch
                )


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
    parser.add_argument("--ckpt-every", type=int, default=1)
    args = parser.parse_args([
        "--batch-size", "10",
        "--epochs", "14",
    ])
    main(args)
