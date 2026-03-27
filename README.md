# Unit 6: assignments development environment

## Hugging Face authentication

Some models (e.g. Stable Diffusion 3.5) require a Hugging Face token. Create a token at
<https://huggingface.co/settings/tokens>, then save it to the file `models/token`.

This file is gitignored.

## Image generation model notes:

- `CompVis/stable-diffusion-v1-4`: works OK, uses ~5 GB VRAM at native 512 x 512 resolution, not gated.
- `stabilityai/stable-diffusion-3.5-medium`: works great, uses peak of ~10 GB VRAM with `enable_model_cpu_offload()`, generation takes ~1 minute on my elderly Pascal GPU. Is gated.