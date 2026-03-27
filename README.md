# Unit 6: assignments development environment

## Requirements

Two requirements files are provided depending on your setup:

| File | Use when |
|---|---|
| `requirements.txt` | Running inside the **devcontainer** (`gperdrizet/llms-gpu` or `gperdrizet/llms-cpu`). The base image already includes torch, transformers, LangChain, gradio, and other heavy dependencies. This file only installs the small set of packages not present in the base image. |
| `requirements-complete.txt` | Running **outside the devcontainer** — e.g. a plain `venv` on your own machine. Contains the full set of packages needed to run the solutions from scratch. Note: install torch separately first following the instructions at <https://pytorch.org/get-started/locally> to get the right CUDA version for your hardware. |

## Hugging Face authentication

Some models (e.g. Stable Diffusion 3.5) require a Hugging Face token. Create a token at
<https://huggingface.co/settings/tokens>, then save it as plaintext to the file `models/token`.

This file is gitignored.

## Image generation models

| Model | Gated | Architecture | Description |
|---|---|---|---|
| [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4) | No | UNet + CLIP | SD 1.x series final checkpoint. Trained on LAION-aesthetics at 512×512. Hosted by the original research group (LMU Munich) rather than Stability AI. |
| [`sd2-community/stable-diffusion-2-1-base`](https://huggingface.co/sd2-community/stable-diffusion-2-1-base) | No | UNet + OpenCLIP | SD 2.x series, 512×512 base variant. Upgraded from CLIP to OpenCLIP text encoder. Community mirror — original stabilityai repo was deleted. |
| [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | No | UNet (2×) + dual encoders | SDXL base. Significantly larger UNet than SD 1/2, uses both CLIP and OpenCLIP in parallel. Native 1024×1024. |
| [`stabilityai/sdxl-turbo`](https://huggingface.co/stabilityai/sdxl-turbo) | No | UNet (2×) + dual encoders | Adversarially distilled SDXL. Generates in 1–4 steps with CFG disabled (`guidance_scale=0.0`). |
| [`stabilityai/stable-diffusion-3.5-medium`](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | Yes | DiT (MMDiT) + T5 | SD 3.x generation. Switches from UNet to a multimodal diffusion transformer. T5 text encoder enables much stronger prompt following. |
| [`stabilityai/stable-diffusion-3.5-large-turbo`](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo) | Yes | DiT (MMDiT) + T5 | 8B parameter SD 3.5, distilled for 4-step generation. Largest model in the benchmark. |
| [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Yes | Flow matching + T5 | State-of-the-art flow matching architecture from the original SD authors. Schnell ("fast") variant distilled for 1–4 steps. |
| [`kandinsky-community/kandinsky-2-2-decoder`](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) | No | Two-stage prior + UNet | DALL-E 2 style two-stage pipeline: a CLIP image prior generates an image embedding, then a UNet decoder renders it. Architecturally distinct from all other models in the benchmark. |
| [`PixArt-alpha/PixArt-XL-2-512x512`](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512) | No | DiT + T5 | Lightweight DiT model natively trained at 512×512. T5 text encoder with efficient transformer blocks. Fast and practical for limited VRAM. |

### Running larger models on limited VRAM

If you want to try a model that exceeds your GPU's VRAM, replace `enable_model_cpu_offload()` with `enable_sequential_cpu_offload()`. This moves individual layers to the GPU one at a time, reducing peak VRAM to ~6-8GB at the cost of much slower generation (5-15 minutes per image). Worth trying if you have the patience and want the best possible quality.

---

## Model benchmark

`src/benchmark.py` measures generation latency, peak system RAM, and peak GPU VRAM for each model across four execution modes on a consumer GPU (NVIDIA GeForce GTX 1070, 8 GB VRAM, sm_61).

**Test hardware:** GTX 1070 (8 GB) — `CUDA_VISIBLE_DEVICES=1`

**Prompt:** "a turtle and a bird together in a forest"

**Output resolution:** 512 × 512

**Methodology:**
- 1 untimed warmup run, then 3 timed replicates per (model, mode) combination.
- Peak system RAM = max RSS of the Python process during timed replicates (polled every 0.5 s via `psutil`).
- Peak GPU VRAM = `torch.cuda.max_memory_allocated()` reset between replicates.
- GPU modes use `torch.float16`; CPU mode uses `torch.float32`.
- OOM = caught `torch.cuda.OutOfMemoryError` or `RuntimeError` during load or inference.

**Execution modes:**
| Mode | Description |
|---|---|
| `gpu_only` | Full model loaded to GPU VRAM in fp16 |
| `model_offload` | `enable_model_cpu_offload()` — submodels moved to GPU one at a time |
| `sequential_offload` | `enable_sequential_cpu_offload()` — individual layers moved layer-by-layer |
| `cpu_only` | Full model on CPU in fp32, no GPU used |

**Inference steps per model:**
| Model | Steps | Rationale |
|---|---|---|
| `CompVis/stable-diffusion-v1-4` | 30 | Standard DDIM/PNDM schedule for SD 1.x |
| `sd2-community/stable-diffusion-2-1-base` | 30 | Same schedule as SD 1.x; community mirror of deleted stabilityai repo |
| `stabilityai/stable-diffusion-xl-base-1.0` | 30 | Standard schedule for SDXL base |
| `stabilityai/sdxl-turbo` | 4 | Adversarially distilled; CFG disabled (`guidance_scale=0.0`) |
| `stabilityai/stable-diffusion-3.5-medium` | 28 | Recommended in model card |
| `stabilityai/stable-diffusion-3.5-large-turbo` | 4 | Turbo distilled model — few steps sufficient |
| `black-forest-labs/FLUX.1-schnell` | 4 | Schnell ("fast") variant designed for 1–4 steps |
| `kandinsky-community/kandinsky-2-2-decoder` | 30 | Two-stage prior+decoder architecture via `KandinskyV22CombinedPipeline` |
| `PixArt-alpha/PixArt-XL-2-512x512` | 20 | Good quality/speed trade-off for DiT architecture |

### Results

Run it from the repository root:

```bash
python image_gen_benchmark/benchmark.py --hardware gtx1070
```

The `--hardware` label organises results under `docs/data/{hardware}/` and `docs/images/{hardware}/`. If omitted, a slug is derived automatically from the detected GPU name.