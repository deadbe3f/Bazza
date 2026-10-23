# CUDA image migration plan

Build completed alongside the active v0.4 training on 13 October 2026. GPU acceptance remains deferred.

## Current environment

- Active image: `kiln-transformers-cuda:5.2`, derived from local `runpod-ml:latest`.
- Image ID: `sha256:ac19b35709b218010080711439c3791d0bf3041232535546bb6e161e6de21f21`.
- Ubuntu 22.04, Python 3.11.11, PyTorch `2.8.0.dev20250319+cu128`, CUDA toolkit 12.8.1.
- RTX 3090 24 GiB; host driver 570.133.07.
- Bazza additionally mounts `gemma4_python` as `/packages`, ahead of the image
  packages in `PYTHONPATH`. Runtime versions must include those overrides.

## Migration

1. Build a separate reproducible development image from a maintained CUDA or
   RunPod base. Pin its base digest, Python and dependencies; use a descriptive
   tag such as `kiln-pytorch:2.14.1-cu126`. Include the development toolkit for
   future extensions. Retain the old image for previous experiment reproduction.
2. Start with stable PyTorch 2.14.1 and a matching CUDA 12.6 toolkit/wheel on the
   existing driver. CUDA 13 is a later option following a host driver upgrade;
   the documented CUDA 13.0 Linux requirement is at least 580.65.06. Verify the
   requirements of the exact chosen toolkit before upgrading the host.
3. Install pinned bitsandbytes, tokenizers, safetensors, tqdm, experiment logging
   dependencies and the optional Transformers oracle inside the image. Replace
   the `/packages` override with image-managed dependencies for new runs. Check
   actual import paths/versions and dependency consistency.
4. Run the existing small Gemma numerical audit on exactly the same weights and
   inputs: unpadded singles, padded singles and mixed batches. Compare current
   defaults and explicit BF16 reduction settings, valid-token hidden states and
   decision predictions. An upgrade is not evidence that the drift is fixed.
5. Measure a representative forward/backward, latency, throughput and peak VRAM
   on the RTX 3090. Accept the image once numerical and runtime checks pass;
   select it through `KNEEJERK_CUDA_IMAGE` and record tag/digest/package versions.
6. Keep old feature banks immutable. A changed backbone runtime/precision policy
   needs a distinct bank identity and output directory. Head-only comparisons
   may reuse the current bank, recording its original export identity separately
   from the new head runtime. Do not mix these results silently.

A separate DEADBE3F runtime has now been built. No active trainer, host driver or old image was changed.

Sources: [PyTorch 2.14.1 release](https://dev-discuss.pytorch.org/t/pytorch-2-14-1-general-availability/3455),
[official CUDA 12.6 wheels](https://download.pytorch.org/whl/cu126/torch/),
[CUDA/driver changes](https://pytorch.org/blog/pytorch-2-12-release-blog/),
[RunPod image sources](https://github.com/runpod/containers).

## Built candidate and next comparison

- Image: `deadbe3f/pytorch:2.14.1-cu126`.
- Image ID: `sha256:e217781427c2d7a3cf90be8b595b2cbba3e8f50d4f906a8a70a26f3d837f7b3e`.
- Official PyTorch devel base, digest `sha256:976e46afb65d637e3aa5214a326370bbd2fd6348545eee1b51fb85dc52c684cc`; equivalent to the RunPod development setup. RunPod's inspected build matrix lists Torch through 2.13.0, whereas the official image includes stable 2.14.1.
- Ubuntu 24.04 / Python 3.12.3 / Torch 2.14.1+cu126 / CUDA toolkit 12.6.3.
- Transformers 5.19.0, tokenizers 0.23.3, bitsandbytes 0.50.2, HF Hub 1.33.0 and TensorBoard 2.21.0; full transitive constraints pinned.
- Image-managed venv inherits official Torch/CUDA; no `/packages` overlay. Optional upstream Spin helper excluded for Click/HF compatibility.
- Dependency checks, imports, CPU autograd, CUDA SM86 compilation, five trainer contracts and DEADBE3F CPU codec tests passed. No GPU execution or model-parity claim yet. Optional CPU NF4 kernels are not installed/tested.
- Build recipe and report: `../0xDEADBE3F/docker/pytorch/README.md`; raw provenance under its `build/docker/pytorch-2.14.1-cu126/`.

After training finishes, manually run:

```bash
bash /media/adrian/Fastly/Projects/ProbablyConverges/KneeJerk/scripts/audit_bazza_batching_deadbe3f.sh
```

The launcher refuses while Bazza v0.4 runs, creates a separate output root, and writes audit results under its `results/` directory. It repeats the same original twelve training requests, exact backbone and head, with the new image and no legacy package mount. It does not update a feature bank or switch training automatically. Changes in several runtime packages mean an improved result would not by itself identify which component fixed the drift.
