import os
import time

import torch
from loguru import logger
from torchvision.io import write_jpeg

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5


def save_image_tensor(image_tensor, file_path, quality=75):
    image_tensor = (image_tensor.cpu().clamp(-1, 1) + 1) * 127.5
    image_tensor = image_tensor.to(torch.uint8)
    write_jpeg(image_tensor.permute(2, 0, 1), file_path, quality=quality)


class FluxGenerator:
    def __init__(self, model_name, device="cuda", offload=False):
        logger.info(f"Initializing FluxGenerator with model: {model_name}")
        self.device = torch.device(device)
        self.offload = offload
        self.model_name = model_name
        self.is_schnell = model_name == "flux-schnell"

        logger.info("Loading T5 model")
        self.t5 = load_t5(self.device, max_length=256 if self.is_schnell else 512)
        logger.info("Loading CLIP model")
        self.clip = load_clip(self.device)
        logger.info("Loading flow model")
        self.model = load_flow_model(model_name, device="cpu" if offload else self.device)
        logger.info("Loading autoencoder")
        self.ae = load_ae(model_name, device="cpu" if offload else self.device)
        logger.info("Loading NSFW classifier")

    @torch.inference_mode()
    def generate(self, prompt, width=1360, height=768, num_steps=50, guidance=3.5, seed=None):
        logger.info(f"Starting image generation for prompt: '{prompt}'")

        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        logger.info(f"Using seed: {seed}")

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        logger.info("Generating initial noise")
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )

        logger.info("Getting schedule")
        timesteps = get_schedule(
            opts.num_steps,
            (x.shape[-1] * x.shape[-2]) // 4,
            shift=(not self.is_schnell),
        )

        if self.offload:
            logger.info("Moving T5 and CLIP models to device")
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        logger.info("Preparing input")
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)

        if self.offload:
            logger.info("Moving T5 and CLIP models back to CPU")
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            logger.info("Moving flow model to device")
            self.model = self.model.to(self.device)

        logger.info("Running denoising process")
        x = denoise(self.model, **inp, timesteps=timesteps, guidance=opts.guidance)

        if self.offload:
            logger.info("Moving flow model back to CPU")
            self.model.cpu()
            torch.cuda.empty_cache()
            logger.info("Moving autoencoder decoder to device")
            self.ae.decoder.to(x.device)

        logger.info("Unpacking and decoding image")
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            logger.info("Moving autoencoder decoder back to CPU")
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        x = x.clamp(-1, 1)
        x = x.float()
        x = x[0].permute(1, 2, 0)

        return x


if __name__ == "__main__":
    logger.add("flux_generator.log", rotation="1 day")

    # input
    generator = FluxGenerator("flux-dev", device="cuda", offload=False)
    prompt = "a photo of a forest with mist swirling around the tree trunks"

    # generation
    start_time = time.time()
    img = generator.generate(prompt)
    end_time = time.time()
    logger.info(f"Total time for image generation: {end_time - start_time:.2f} seconds")

    # Save the generated image
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_image.jpg")
    save_image_tensor(img, output_path)

    logger.info(f"Image saved in {output_dir} ({end_time - start_time:.4f} seconds)")
