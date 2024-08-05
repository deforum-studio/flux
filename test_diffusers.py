import os
import torch
from flux_pipeline import FluxImg2ImgPipeline

pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

max_frames = 100
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
image = None

for i in range(max_frames):

    prompt = "cat sushi"
    out = pipe(
        image=image,
        prompt=prompt,
        guidance_scale=3,
        strength=0.1,
        height=768,
        width=1344,
        num_inference_steps=50,
        max_sequence_length=256,
    ).images[0]

    output_path = os.path.join(output_dir, f"generated_image_{i:05d}.jpg")
    out.save(output_path)
    image = out