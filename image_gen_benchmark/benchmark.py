'''
Image-generation benchmark - compares model latency and memory usage across
diffusion models and execution modes on a single GPU.

Results are written to:
    docs/data/{hardware}/benchmark_results.json
    docs/images/{hardware}/{model_short}_{mode}_rep{n}.png

Run from the repository root:
    python image_gen_benchmark/benchmark.py --hardware gtx1070

The --hardware label is used to organize output paths and is embedded in every
result entry so the JSON is self-describing.  If omitted, a slug is derived
automatically from the detected GPU name (or "cpu" on CPU-only machines).
'''

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse CLI args before setting environment variables
# ---------------------------------------------------------------------------
# This must happen before torch is imported so CUDA_VISIBLE_DEVICES is set
# in time to affect device enumeration.
_parser = argparse.ArgumentParser(description='Image generation benchmark', add_help=False)

_parser.add_argument('--gpu', default='0',
    help='CUDA device index to use, passed to CUDA_VISIBLE_DEVICES (default: 0)')

_parser.add_argument('--hardware', default=None)
_pre_args, _ = _parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = _pre_args.gpu

# Point HuggingFace cache to the project-local models/ directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ['HF_HOME'] = str(_REPO_ROOT / 'models')

# Suppress the torchvision-not-installed fallback warnings from transformers.
# These are expected: we intentionally do not install torchvision to avoid
# conflicting with the container's custom-compiled torch build.
# transformers emits these via its own logging abstraction (not Python warnings
# or the standard logging module), so the only reliable suppression is to set
# the transformers verbosity level before any diffusers/transformers imports.
import transformers as _transformers  # noqa: E402
_transformers.utils.logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Now safe to import torch and core helpers
# ---------------------------------------------------------------------------
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_monitor import MemoryMonitor  # noqa: E402


def _import_pipeline_class(class_name: str):
    '''Lazily import a single diffusers pipeline class by name.

    Importing pipeline classes at the top level fails when torchvision is not
    present or is incompatible with the container's torch build (e.g.
    FluxPipeline pulls in CLIPImageProcessor which needs torchvision). Lazy
    per-class imports isolate the failure to the specific model that needs the
    broken dependency, leaving the rest of the benchmark unaffected.
    '''

    import importlib
    diffusers = importlib.import_module('diffusers')

    return getattr(diffusers, class_name)

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

PROMPT = 'a turtle and a bird together in a forest'

HEIGHT = 512
WIDTH = 512
REPLICATES = 3          # Timed runs per (model, mode) combination
WARMUP_RUNS = 1         # Untimed runs before timing starts

# Model registry: model_id -> (pipeline_class_name, num_inference_steps, short_name)
# Pipeline classes are imported lazily at runtime so an import error for one
# model (e.g. FluxPipeline needing torchvision) does not abort the whole run.
MODELS: dict[str, tuple] = {
    # model_id: (pipeline_class_name, steps, short_name, extra_inference_kwargs)
    # extra_inference_kwargs are passed directly to the pipeline call and allow
    # per-model overrides (e.g. guidance_scale=0.0 for SDXL Turbo).
    'CompVis/stable-diffusion-v1-4':                ('StableDiffusionPipeline',      30, 'sd1_4',             {}),
    'sd2-community/stable-diffusion-2-1-base':      ('StableDiffusionPipeline',      30, 'sd2_1_base',        {}),
    'stabilityai/stable-diffusion-xl-base-1.0':     ('StableDiffusionXLPipeline',    30, 'sdxl',              {}),
    'stabilityai/sdxl-turbo':                       ('StableDiffusionXLPipeline',    4,  'sdxl_turbo',        {'guidance_scale': 0.0}),
    'stabilityai/stable-diffusion-3.5-medium':      ('StableDiffusion3Pipeline',     28, 'sd3_5_medium',      {}),
    'stabilityai/stable-diffusion-3.5-large-turbo': ('StableDiffusion3Pipeline',     4,  'sd3_5_large_turbo', {}),
    'black-forest-labs/FLUX.1-schnell':             ('FluxPipeline',                 4,  'flux_schnell',      {}),
    'kandinsky-community/kandinsky-2-2-decoder':    ('KandinskyV22CombinedPipeline', 30, 'kandinsky_2_2',     {}),
    'PixArt-alpha/PixArt-XL-2-512x512':             ('PixArtAlphaPipeline',          20, 'pixart_512',        {}),
}

# Execution modes to test per model
MODES = ['gpu_only', 'model_offload', 'sequential_offload']

# Paths — computed at runtime once the --hardware label is known
# (see run_benchmark())
_DOCS_ROOT = _REPO_ROOT / 'docs'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hf_cache_exists(model_id: str) -> bool:
    '''Return True if the model snapshot already exists in HF_HOME/hub/.'''

    slug = 'models--' + model_id.replace('/', '--')
    hub_path = Path(os.environ['HF_HOME']) / 'hub' / slug

    return hub_path.exists()


def _load_pipeline(
    model_id: str,
    pipeline_class_name: str,
    mode: str,
) -> object:
    '''
    Load a diffusion pipeline with the dtype and offload strategy dictated by
    *mode*.  Returns the pipeline object (already moved to the target device
    where applicable).

    Modes
    -----
    gpu_only           - fp16, .to('cuda'), no offload
    model_offload      - fp16, enable_model_cpu_offload()
    sequential_offload - fp16, enable_sequential_cpu_offload()
    '''

    dtype = torch.float16

    pipeline_cls = _import_pipeline_class(pipeline_class_name)

    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    if mode == 'gpu_only':
        pipe = pipe.to('cuda')

    elif mode == 'model_offload':
        pipe.enable_model_cpu_offload()

    elif mode == 'sequential_offload':
        pipe.enable_sequential_cpu_offload()

    return pipe


def _run_inference(pipe, steps: int, extra_kwargs: dict) -> tuple[object, float]:
    '''Run one inference pass. Returns (image, elapsed_seconds).'''

    t0 = time.perf_counter()

    result = pipe(
        PROMPT,
        num_inference_steps=steps,
        height=HEIGHT,
        width=WIDTH,
        **extra_kwargs,
    )

    elapsed = time.perf_counter() - t0

    return result.images[0], elapsed


def _cleanup(pipe) -> None:
    '''Release GPU and CPU memory held by *pipe*.'''

    del pipe
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(hardware: str) -> list[dict]:
    results_dir = _DOCS_ROOT / 'data' / hardware
    images_dir = _DOCS_ROOT / 'images' / hardware
    results_file = results_dir / 'benchmark_results.json'

    results_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load any results already saved to disk so we don't re-run
    # completed (model, mode) pairs if the benchmark is restarted.
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as fh:
            all_results: list[dict] = json.load(fh)
        completed = {(r['model_id'], r['mode']) for r in all_results}
        print(f'Resuming — {len(completed)} (model, mode) pairs already completed.')
    else:
        all_results = []
        completed = set()

    for model_id, (pipeline_class_name, steps, short_name, extra_kwargs) in MODELS.items():
        print(f'\n{"=" * 70}')
        print(f'Model: {model_id}')
        print(f'{"=" * 70}')

        cached_before_run = _hf_cache_exists(model_id)

        for mode in MODES:
            if (model_id, mode) in completed:
                print(f'\n  Mode: {mode}  [skipped — already in results]')
                continue

            print(f'\n  Mode: {mode}')

            result_entry: dict = {
                'hardware': hardware,
                'model_id': model_id,
                'model_short': short_name,
                'mode': mode,
                'prompt': PROMPT,
                'height': HEIGHT,
                'width': WIDTH,
                'steps': steps,
                'cached_at_load': None,   # filled after load attempt
                'load_time_s': None,
                'load_error': None,
                'warmup_error': None,
                'replicates': [],
                'peak_system_ram_mb': None,
                'peak_gpu_vram_mb': None,
            }

            # --- Load model ---------------------------------------------------
            monitor = MemoryMonitor(cuda_device_index=0)  # device 0 after remapping

            try:
                result_entry['cached_at_load'] = _hf_cache_exists(model_id)
                t_load_start = time.perf_counter()

                monitor.start()
                pipe = _load_pipeline(model_id, pipeline_class_name, mode)
                load_time = time.perf_counter() - t_load_start
                monitor.stop()

                result_entry['load_time_s'] = round(load_time, 3)

                print(f'    Loaded in {load_time:.1f}s  '
                      f'(cached={result_entry["cached_at_load"]})')

            except (torch.cuda.OutOfMemoryError, RuntimeError, ImportError, AttributeError) as exc:

                monitor.stop()
                msg = str(exc)
                result_entry['load_error'] = msg
                all_results.append(result_entry)
                print(f'    LOAD ERROR: {msg[:120]}')

                continue
    
            except Exception as exc:

                monitor.stop()
                result_entry['load_error'] = str(exc)
                all_results.append(result_entry)
                print(f'    UNEXPECTED LOAD ERROR: {exc}')
                _cleanup(None)

                continue

            # --- Warmup run(s) ------------------------------------------------
            try:
                for _ in range(WARMUP_RUNS):
                    _run_inference(pipe, steps, extra_kwargs)

                print(f'    Warmup done.')

            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
    
                msg = str(exc)
                result_entry['warmup_error'] = msg
                all_results.append(result_entry)
                _cleanup(pipe)
                print(f'    WARMUP OOM: {msg[:120]}')
    
                continue

            # --- Timed replicates ---------------------------------------------
            monitor = MemoryMonitor(cuda_device_index=0)
            monitor.start()

            rep_errors = False
            for rep in range(1, REPLICATES + 1):

                rep_entry: dict = {
                    'rep': rep,
                    'elapsed_s': None,
                    'error': None,
                    'image_path': None,
                }

                try:
                    image, elapsed = _run_inference(pipe, steps, extra_kwargs)
                    rep_entry['elapsed_s'] = round(elapsed, 3)

                    img_filename = f'{short_name}_{mode}_rep{rep}.png'
                    img_path = images_dir / img_filename
                    image.save(img_path)
                    rep_entry['image_path'] = str(img_path.relative_to(_REPO_ROOT))

                    print(f'    Rep {rep}: {elapsed:.1f}s  -> {img_filename}')

                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:

                    msg = str(exc)
                    rep_entry['error'] = msg
                    rep_errors = True
                    print(f'    Rep {rep} OOM: {msg[:120]}')

                    break

                result_entry['replicates'].append(rep_entry)

            monitor.stop()
            result_entry['peak_system_ram_mb'] = round(monitor.peak_system_mb, 1)
            result_entry['peak_gpu_vram_mb'] = round(monitor.peak_gpu_mb, 1)

            print(f'    Peak RAM: {result_entry["peak_system_ram_mb"]} MB  '
                  f'Peak VRAM: {result_entry["peak_gpu_vram_mb"]} MB')

            _cleanup(pipe)
            all_results.append(result_entry)

            # Persist results after every (model, mode) pair so partial data
            # is not lost if the run crashes later
            _save_results(all_results, results_file)

    return all_results


def _save_results(results: list[dict], results_file: Path) -> None:
    with open(results_file, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)

    print(f'\n  [saved -> {results_file}]')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image generation benchmark')
    parser.add_argument(
        '--gpu', default='0',
        help='CUDA device index to use, passed to CUDA_VISIBLE_DEVICES (default: 0)',
    )
    parser.add_argument(
        '--hardware',
        default=None,
        help=(
            'Label for this hardware run (e.g. gtx1070, p100). '
            'Used to organise output paths under docs/. '
            'Defaults to a slug derived from the detected GPU name, '
            'or "cpu" on CPU-only machines.'
        ),
    )
    args = parser.parse_args()
    # Note: --gpu was already consumed by _pre_args above to set
    # CUDA_VISIBLE_DEVICES before torch was imported. args.gpu is the same value.

    # Auto-detect hardware label if not provided
    hardware = args.hardware
    if hardware is None:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            hardware = re.sub(r'[^a-z0-9]+', '_', gpu_name.lower()).strip('_')
        else:
            hardware = 'cpu'
        print(f'Hardware label auto-detected: {hardware!r}')
        print(f'  (override with --hardware <label> if desired)')

    print(f'PyTorch version : {torch.__version__}')
    print(f'CUDA available  : {torch.cuda.is_available()}')

    if torch.cuda.is_available():
        print(f'GPU             : {torch.cuda.get_device_name(0)}')
        props = torch.cuda.get_device_properties(0)
        total_vram_mb = props.total_memory / (1024 ** 2)
        print(f'Total VRAM      : {total_vram_mb:.0f} MB')

    print(f'Hardware label  : {hardware}')
    print(f'HF_HOME         : {os.environ["HF_HOME"]}')
    print(f'Results dir     : {_DOCS_ROOT / "data" / hardware}')

    results = run_benchmark(hardware)

    print(f'\nBenchmark complete. {len(results)} entries written to:')
    print(f'  {_DOCS_ROOT / "data" / hardware / "benchmark_results.json"}')
