<div align=center>

# DocRes-CPU

CPU-only fork of [starinspace/DocRes-Fork](https://github.com/starinspace/DocRes-Fork), which is itself a fork of [ZZZHANG-jx/DocRes](https://github.com/ZZZHANG-jx/DocRes)

[CVPR 2024] A Generalist Model Toward Unifying Document Image Restoration Tasks

</div>

## Changes from starinspace/DocRes-Fork

- **CPU-only support**: removed all CUDA hard-dependencies, runs without a GPU
- `data/MBD/infer.py`: replaced `DataParallel`, `.cuda()`, and hardcoded CUDA `torch.load` with CPU equivalents; stripped `module.` prefix from weight keys when loading without `DataParallel`
- `inference.py`: replaced all `.half()` (float16) calls with `.float()` (float32), as CPU PyTorch does not support float16 convolutions
- `requirements.txt`: switched torch and torchvision to CPU builds, updated numpy and scikit-image versions to fix binary incompatibility

## My recommended task order

I could not find any recommended order for chaining tasks. The `end2end` mode only chains dewarping → deshadowing → appearance, skipping deblurring and binarization entirely.

Based on what each task's prompt actually computes, my recommended order is:

```
Dewarping → Deshadowing → Appearance → Deblurring → Binarization
```

- **Dewarping** does not always give expected results. Consider skipping it or preprocessing the input image to be more suitable for the model
- Skip **binarization** if you want color output
- **Binarization must always be last** if used, as it converts to black-and-white irreversibly

## Setup

Clone the repository and enter the directory, then:

```bash
uv venv --python 3.10
.venv\Scripts\activate
uv pip install -r requirements.txt
mkdir output
```

1. Put MBD model weights [mbd.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `./models/`
2. Put DocRes model weights [docres.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `./models/`
3. Put your images in `./input/` and run:

```bash
python inference.py --im_path ./input/photo.png --task deshadowing --memory_fix 2
```

Results are saved in `./output/`.

### Arguments

- `--im_path`: path to the input image or folder (default: `./distorted/`)
- `--task`: task to execute, one of `dewarping`, `deshadowing`, `appearance`, `deblurring`, `binarization`, `end2end`
- `--memory_fix`: limits the long edge of the image to reduce memory usage. `0` = default (no limit, except appearance and deshadowing which cap at 1600px), `1` = 1500px, `2` = 2000px, `3` = 3000px
- `--out_folder`: path to the output folder (default: `./output/`)
- `--model_path`: path to the model checkpoint (default: `models/docres.pkl`), also accepts `.safetensors` models
- `--save_dtsprompt`: set to `1` to save the DTSPrompt alongside the output (default: `0`)

---

## Credits

- [starinspace/DocRes-Fork](https://github.com/starinspace/DocRes-Fork)
- [ZZZHANG-jx/DocRes](https://github.com/ZZZHANG-jx/DocRes)