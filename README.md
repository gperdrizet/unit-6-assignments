# Unit 6: assignments development environment

## Hugging Face authentication

Some models (e.g. Stable Diffusion 3.5) require a Hugging Face token. Create a token at
<https://huggingface.co/settings/tokens>, then save it to `models/token`:

```bash
echo "hf_YOUR_TOKEN_HERE" > models/token
```

This file is gitignored.

## Image generation model notes:

CompVis/stable-diffusion-v1-4: not gated, works OK, uses ~5 GB VRAM at native 512 x 512 resolution, not gated.