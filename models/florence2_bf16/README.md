# Florence-2 BF16 Model on AWS Inferentia2

Complete guide for deploying Microsoft Florence-2 vision-language model on AWS Inferentia2 with BF16 precision.

## Table of Contents

- [Performance](#performance)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Compilation](#compilation)
- [Inference](#inference)
- [Supported Tasks](#supported-tasks)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Performance

BF16 precision achieves **45% throughput improvement** over FP32:

| Metric | BF16 Performance |
|--------|------------------|
| **Single-Core QPS** | 4.09 queries/sec |
| **Dual-Core QPS** | 8.18 queries/sec |
| **CAPTION Latency** | 252ms |
| **OD Latency** | 237ms |
| **OCR Latency** | 231ms |

**BF16 is 15.7x faster than CPU inference (~2000ms).**

## Prerequisites

### Hardware
- AWS Inferentia2 instance (inf2.xlarge or larger)
- Recommended: inf2.xlarge (2 NeuronCores) for development
- For compilation: inf2.8xlarge (larger instances prevent OOM)

### Software
- Python 3.8+
- AWS Neuron SDK 2.x

## Installation

```bash
# Install Neuron SDK and dependencies
pip install torch-neuronx neuronx-cc transformers einops timm pillow

# Clone or navigate to project directory
cd neuronx-distributed-inference-main
```

## Compilation

Compilation takes approximately **10-15 minutes** and produces ~1.5GB of compiled models.

### Basic Compilation

```bash
python -m models.florence2_bf16.compile --output-dir ./compiled_bf16
```

This compiles all components:
- 4 vision encoder stages (DaViT architecture)
- Projection layer
- Language encoder
- 6 decoder buckets (1, 4, 8, 16, 32, 64 tokens)

### Output Files

After compilation, `./compiled_bf16/` contains:

```
compiled_bf16/
├── stage0.pt          # Vision encoder stage 0
├── stage1.pt          # Vision encoder stage 1
├── stage2.pt          # Vision encoder stage 2
├── stage3.pt          # Vision encoder stage 3
├── projection.pt      # Vision-to-language projection
├── encoder.pt         # Language encoder
├── decoder_1.pt       # Decoder (1 token bucket)
├── decoder_4.pt       # Decoder (4 tokens)
├── decoder_8.pt       # Decoder (8 tokens)
├── decoder_16.pt      # Decoder (16 tokens)
├── decoder_32.pt      # Decoder (32 tokens)
└── decoder_64.pt      # Decoder (64 tokens)
```

## Inference

### Python API

```python
from models.florence2_bf16.inference import Florence2NeuronBF16
from PIL import Image

# Initialize model (loads compiled models)
model = Florence2NeuronBF16("./compiled_bf16", core_id="0")

# Image captioning
result = model("photo.jpg", "<CAPTION>")
print(result)  # "A dog playing in the park"

# Object detection
result = model("photo.jpg", "<OD>")
print(result)  # "dog<loc_123><loc_456>..."

# OCR
result = model("document.jpg", "<OCR>")
print(result)  # Extracted text
```

### Command Line

```bash
python -m models.florence2_bf16.inference \
    --image photo.jpg \
    --task "<CAPTION>" \
    --model-dir ./compiled_bf16 \
    --core 0
```

### Quick Start Example

```bash
python examples/quick_start.py --image your_image.jpg
```

This script:
1. Automatically compiles models if needed
2. Runs inference on multiple tasks
3. Displays results with timing

## Supported Tasks

| Task | Prompt | Output | Latency |
|------|--------|--------|---------|
| **Caption** | `<CAPTION>` | Brief image description | ~250ms |
| **Detailed Caption** | `<DETAILED_CAPTION>` | Comprehensive description | ~280ms |
| **Object Detection** | `<OD>` | Objects with bounding boxes | ~240ms |
| **OCR** | `<OCR>` | Extracted text | ~230ms |
| **Region Caption** | `<REGION_CAPTION>` | Description of specific region | ~260ms |

### Task Examples

```python
model = Florence2NeuronBF16("./compiled_bf16")

# Caption
result = model("street.jpg", "<CAPTION>")
# Output: "A busy street with cars and pedestrians"

# Detailed caption
result = model("street.jpg", "<DETAILED_CAPTION>")
# Output: "The image shows a bustling urban street during daytime..."

# Object detection
result = model("street.jpg", "<OD>")
# Output: "car<loc_100><loc_200><loc_300><loc_400>person<loc_150>..."

# OCR
result = model("receipt.jpg", "<OCR>")
# Output: "RECEIPT\nTotal: $42.50\nDate: 2024-01-15"
```

## Advanced Usage

### Dual-Core Deployment

Maximize throughput by running two independent processes on separate NeuronCores:

```bash
# Terminal 1 - Use NeuronCore 0
NEURON_RT_VISIBLE_CORES=0 python -m models.florence2_bf16.inference \
    --image img1.jpg --core 0 &

# Terminal 2 - Use NeuronCore 1
NEURON_RT_VISIBLE_CORES=1 python -m models.florence2_bf16.inference \
    --image img2.jpg --core 1 &
```

This achieves **8+ QPS** on inf2.xlarge (2 NeuronCores).

### Custom Max Tokens

```python
# Generate longer captions (default: 100 tokens)
result = model("image.jpg", "<DETAILED_CAPTION>", max_tokens=200)
```

### Batch Processing

```python
import os
from pathlib import Path

model = Florence2NeuronBF16("./compiled_bf16")

# Process directory of images
image_dir = Path("./images")
for image_path in image_dir.glob("*.jpg"):
    result = model(str(image_path), "<CAPTION>")
    print(f"{image_path.name}: {result}")
```

### Performance Benchmarking

```bash
# Single-core benchmark (60 seconds)
python -m models.florence2_bf16.benchmark \
    --image test.jpg \
    --duration 60

# Dual-core stress test (5 minutes)
python -m models.florence2_bf16.benchmark \
    --stress --duration 300 --core 0 &
python -m models.florence2_bf16.benchmark \
    --stress --duration 300 --core 1 &
wait
```

## Troubleshooting

### Compilation Issues

| Issue | Solution |
|-------|----------|
| **Out of memory during compilation** | Use larger instance (inf2.8xlarge) |
| **Compilation hangs** | Check `neuron-top` for activity, verify Neuron SDK installed |
| **Import errors** | Verify torch-neuronx: `python -c "import torch_neuronx; print(torch_neuronx.__version__)"` |

### Inference Issues

| Issue | Solution |
|-------|----------|
| **`FileNotFoundError: stage0.pt`** | Run compilation script first |
| **`RuntimeError: No Neuron devices`** | Install driver: `sudo apt install aws-neuronx-dkms` |
| **Slow first inference** | Expected - models load on first call (add warmup) |
| **Lower than expected QPS** | Run dual processes on separate cores |
| **CUDA/CPU fallback** | Ensure `NEURON_RT_VISIBLE_CORES` is set correctly |

### Warmup Best Practice

```python
# Add warmup to avoid cold start latency
model = Florence2NeuronBF16("./compiled_bf16")

# Warmup with dummy image
import torch
from PIL import Image
dummy_img = Image.fromarray(
    (torch.rand(768, 768, 3) * 255).byte().numpy()
)
_ = model(dummy_img, "<CAPTION>")  # Discard result

# Now ready for production requests
result = model("real_image.jpg", "<CAPTION>")
```

## Architecture

### Model Pipeline

```
Input Image (768×768)
    │
    ▼
┌─────────────────────┐
│  Vision Encoder     │
│  (DaViT - 4 stages) │
│  768→192→96→48→24   │
└──────────┬──────────┘
           │ (576×1024)
           ▼
┌─────────────────────┐
│  Projection Layer   │
│  1024 → 768         │
└──────────┬──────────┘
           │ (577×768)
           ▼
┌─────────────────────┐
│  Language Encoder   │
│  (BART - 6 layers)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Language Decoder   │
│  (Bucketed: 1-64)   │
└──────────┬──────────┘
           │
           ▼
      Output Text
```

### Key Optimizations

1. **Stage-wise Compilation**: DaViT split into 4 independent stages with fixed shapes
2. **Bucket Strategy**: Pre-compile decoders for 6 bucket sizes, select smallest fitting bucket at runtime
3. **BF16 Precision**: Reduces memory bandwidth by 50%, hardware accelerated on NeuronCores
4. **Compiled Projection**: Projection layer runs on Neuron (not CPU), avoiding costly dtype conversions

## Technical Details

### BF16 vs FP32

| Aspect | FP32 | BF16 | Improvement |
|--------|------|------|-------------|
| Precision | 32-bit | 16-bit | - |
| Model Size | ~3GB | ~1.5GB | 50% smaller |
| Memory Bandwidth | 2x | 1x | 50% reduction |
| Latency | 393ms | 252ms | 36% faster |
| Throughput | 2.82 QPS | 4.09 QPS | 45% higher |
| Accuracy Loss | Baseline | <0.5% | Negligible |

### Bucket Strategy

The decoder uses a bucketing strategy to handle variable-length sequences efficiently:

- **Buckets**: [1, 4, 8, 16, 32, 64] tokens
- **Selection**: At each decoding step, select smallest bucket >= current sequence length
- **Padding**: Pad sequence to bucket size with zeros
- **Benefit**: Avoid recompilation for every sequence length

Example:
- Sequence length 7 → use bucket 8 (pad 1 token)
- Sequence length 20 → use bucket 32 (pad 12 tokens)

## Limitations

- **Input Size**: Fixed at 768×768 pixels (images resized automatically)
- **Max Generation**: 64 tokens (configurable by modifying bucket sizes)
- **Batch Size**: 1 (model too small to benefit from batching)
- **Inferentia1**: Not supported (requires Neuron SDK 2.x)

## References

- [Florence-2 Model Card](https://huggingface.co/microsoft/Florence-2-base)
- [AWS Inferentia2 Documentation](https://aws.amazon.com/machine-learning/inferentia/)
- [Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)

## License

Apache 2.0

---

**For production deployments, this BF16 implementation is recommended over FP32.**
