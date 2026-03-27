# Unit 6: assignments development environment

## Hugging Face authentication

Some models (e.g. Stable Diffusion 3.5) require a Hugging Face token. Create a token at
<https://huggingface.co/settings/tokens>, then save it to the file `models/token`.

This file is gitignored.

## Image generation model notes:

- [**`CompVis/stable-diffusion-v1-4`**](https://huggingface.co/CompVis/stable-diffusion-v1-4): Older, less powerful model, but still works great and is not gated so no need to deal with logging in. Uses ~5 GB VRAM at native 512 x 512 resolution. Was originally a stability AI model but is no longer avalible on the stability AI Hugging Face Hub page due to it's age.

- [**`stability/stable-diffusion-3.5-medium`**](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium): More modern/powerful model from Stability AI's current generation, must accept license agreement and authenticate your Hugging Face client to use. Peak of ~10 GB VRAM with [`enable_model_cpu_offload()`](https://huggingface.co/docs/diffusers/v0.37.1/en/api/pipelines/overview#diffusers.DiffusionPipeline.enable_model_cpu_offload), generation takes ~1 minute on my elderly Pascal GPU. Can speed generation up by setting 512 x 512 output resolution (default is 1024 x x1024).