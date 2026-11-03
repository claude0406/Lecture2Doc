---
license: apache-2.0
base_model: Qwen/Qwen3-VL-2B-Instruct
tags:
  - uni-mumer
  - hmer
  - math-ocr
  - handwritten-math
  - latex
  - qwen3-vl
  - vision-language
pipeline_tag: image-to-text
library_name: transformers
datasets:
  - phxember/Uni-MuMER-Data
language:
  - en
---

# Uni-MuMER-Qwen3-VL-2B

This model is a **Uni-MuMER** fine-tuned version of [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) for **Handwritten Mathematical Expression Recognition (HMER)** -- converting images of handwritten math expressions into LaTeX.

Uni-MuMER uses unified multi-task fine-tuning with three auxiliary tasks (Tree-CoT, Error-Driven Learning, Symbol Counting) to inject domain-specific knowledge into a general-purpose vision-language model.

> **Paper**: [Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2505.23566) (NeurIPS 2025 Spotlight)

## Model Details

| Property | Value |
|----------|-------|
| Base Model | Qwen/Qwen3-VL-2B-Instruct |
| Architecture | Qwen3-VL (dedicated vision-language model with ViT encoder) |
| Parameters | 2.1B |
| Training Data | [Uni-MuMER-Data](https://huggingface.co/datasets/phxember/Uni-MuMER-Data) (~1.6M samples, 26 subsets) |
| Fine-tuning | Full (all parameters, including vision encoder) |
| Precision | bf16 |
| Template | `qwen3_vl_nothink` (via LLaMA-Factory) |

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 1 |
| Learning Rate | 1e-5 (cosine schedule, 10% warmup) |
| Effective Batch Size | 512 (8 GPUs x 4 per-device x 16 accumulation) |
| DeepSpeed | ZeRO-3 |
| Image Max Pixels | 262,144 (512x512) |
| Cutoff Length | 2048 tokens |
| Hardware | 8x NVIDIA A100 80GB |
| Training Time | ~18 hours |
| Final Train Loss | 0.0433 |

## Benchmark Results

### ExpRate (Exact Match Rate)

| Dataset | Samples | Uni-MuMER-3B | Qwen3.5-2B | Qwen3.5-4B | **This Model** | Qwen3-VL-4B |
|---------|---------|:------------:|:----------:|:----------:|:--------------:|:-----------:|
| CROHME 2014 | 986 | 82.25% | 83.98% | 82.56% | 83.27% | 82.35% |
| CROHME 2016 | 1,147 | 78.29% | 81.17% | 78.20% | 78.55% | 79.34% |
| CROHME 2019 | 1,199 | 79.82% | 80.15% | 75.98% | 79.40% | 78.98% |
| CROHME 2023 Test | 2,300 | 69.52% | 69.43% | 66.74% | **70.96%** | 69.17% |
| CROHME 2023 Val | 555 | 68.11% | 69.91% | 67.75% | **70.63%** | 66.67% |
| HME100K Test | 24,607 | 69.50% | 70.43% | 70.02% | 69.31% | 69.79% |
| Im2LaTeXv2 Test | 10,118 | 76.99% | 77.82% | 77.20% | 77.11% | 77.46% |
| MathWriting Test | 7,643 | 53.03% | 51.84% | 54.32% | 50.66% | 53.15% |
| **Average** | | 72.19% | **73.09%** | 71.60% | 72.49% | 72.11% |

### CDM Metrics (Visual-Equivalence-Aware)

| Dataset | CDM F1 | CDM ExpRate |
|---------|--------|-------------|
| CROHME 2014 | 0.9700 | 86.50% |
| CROHME 2016 | 0.9660 | 81.50% |
| CROHME 2019 | 0.9700 | 82.60% |
| HME100K Test | 0.9710 | 72.40% |
| Im2LaTeXv2 Test | 0.9940 | 92.60% |
| MathWriting Test | 0.9480 | 69.00% |
| **Average** | **0.9692** | **80.35%** |

## Usage

### Inference with vLLM

```python
from vllm import LLM, SamplingParams

model = LLM(
    model="phxember/Uni-MuMER-Qwen3-VL-2B",
    trust_remote_code=True,
    max_model_len=2048,
)

sampling_params = SamplingParams(temperature=0.2, top_p=0.8, max_tokens=2048)

# See https://github.com/BFlameSwift/Uni-MuMER for full inference scripts
```

### Batch Evaluation

```bash
git clone https://github.com/BFlameSwift/Uni-MuMER.git
cd Uni-MuMER
bash eval/eval_all.sh -m phxember/Uni-MuMER-Qwen3-VL-2B
```

## Related Models

| Model | Base | Avg ExpRate | HuggingFace |
|-------|------|-------------|-------------|
| Uni-MuMER-Qwen2.5-VL-3B | Qwen2.5-VL-3B | 72.19% | [Link](https://huggingface.co/phxember/Uni-MuMER-Qwen2.5-VL-3B) |
| Uni-MuMER-Qwen2.5-VL-7B | Qwen2.5-VL-7B | - | [Link](https://huggingface.co/phxember/Uni-MuMER-Qwen2.5-VL-7B) |
| Uni-MuMER-Qwen3.5-2B | Qwen3.5-2B | **73.09%** | [Link](https://huggingface.co/phxember/Uni-MuMER-Qwen3.5-2B) |
| Uni-MuMER-Qwen3.5-4B | Qwen3.5-4B | 71.60% | [Link](https://huggingface.co/phxember/Uni-MuMER-Qwen3.5-4B) |
| **Uni-MuMER-Qwen3-VL-2B** | Qwen3-VL-2B | 72.49% | This model |
| Uni-MuMER-Qwen3-VL-4B | Qwen3-VL-4B | 72.11% | [Link](https://huggingface.co/phxember/Uni-MuMER-Qwen3-VL-4B) |

## Citation

```bibtex
@article{li2025unimumer,
  title = {Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition},
  author = {Li, Yu and Jiang, Jin and Zhu, Jianhua and Peng, Shuai and Wei, Baole and Zhou, Yuxuan and Gao, Liangcai},
  year = {2025},
  journal = {arXiv preprint arXiv:2505.23566},
}
```
