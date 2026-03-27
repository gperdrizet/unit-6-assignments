# Unit 6: assignments development environment

[![pages-build-deployment](https://github.com/gperdrizet/unit-6-assignments/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/gperdrizet/unit-6-assignments/actions/workflows/pages/pages-build-deployment)

This repo was originally intended as a template for the unit 6 assignments development environment. It still works for that purpose, but as usual, I had a little too much fun along the way. It now also contains (and was mostly taken over by) a benchmark comparing several image generation models across two tiers of hardware.

For full results, see [here](https://gperdrizet.github.io/unit-6-assignments).

---

## Requirements

Two requirements files are provided depending on your setup:

| File | Use when |
|---|---|
| `requirements.txt` | **Recommended**: running inside the devcontainer. The base image already includes torch, transformers, LangChain, gradio, and other heavy dependencies. This file only installs the small set of packages not present in the base image. |
| `requirements-complete.txt` | Running without the devcontainer - e.g. a plain `venv` on your own machine. Contains the full set of packages needed to run. Note: install torch separately first following the instructions at <https://pytorch.org/get-started/locally> to get the right CUDA version for your hardware. |

---

## Hugging Face authentication

Some models (e.g. Stable Diffusion 3.5) require a Hugging Face token. Create a token at
<https://huggingface.co/settings/tokens>, then save it as plaintext in the file `models/token`.

This file is gitignored.

---

## Model benchmark

The purpose of the benchmark is to illustrate the difference between models and what's possible with two common tiers of GPU hardware: 

- Used consumer card with <8 GB VRAM (GTX 1070, 8 GB ~$75 on Ebay March 2026)
- Cheap second-hand server GPU with >12 GB VRAM (Tesla P100, 16 GB, ~$100 on Ebay March 2026). 

Nine models spanning SD 1.x through modern flow-matching architectures are tested across four execution modes (full GPU, model offload, sequential offload, and CPU-only), with generation time, peak GPU VRAM, and peak system RAM recorded for each combination.

**Models tested:**

| Model | Steps |
|---|---|
| `CompVis/stable-diffusion-v1-4` | 30 |
| `sd2-community/stable-diffusion-2-1-base` | 30 |
| `stabilityai/stable-diffusion-xl-base-1.0` | 30 |
| `stabilityai/sdxl-turbo` | 4 |
| `stabilityai/stable-diffusion-3.5-medium` | 28 |
| `stabilityai/stable-diffusion-3.5-large-turbo` | 4 |
| `black-forest-labs/FLUX.1-schnell` | 4 |
| `kandinsky-community/kandinsky-2-2-decoder` | 30 |
| `PixArt-alpha/PixArt-XL-2-512x512` | 20 |

Each (model, mode) pair is run for 3 timed replicates at 512×512 output resolution on the prompt `"a turtle and a bird together in a forest"`. See the full results, per-hardware breakdowns, and generated images on the [GitHub Pages site](https://gperdrizet.github.io/unit-6-assignments).

To run the benchmark yourself:

```bash
python image_gen_benchmark/benchmark.py --hardware gtx1070 --gpu 0
```