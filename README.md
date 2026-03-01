# Florence-2 BF16 Inference for AWS Inferentia2

Optimized Florence-2 vision-language model inference using BF16 precision on AWS Inferentia2 (inf2 instances). This project provides two implementation approaches for different use cases.

## Overview

Florence-2 is a vision-language model capable of:
- Image captioning (brief and detailed)
- Object detection
- OCR (Optical Character Recognition)
- Region-specific captioning
- Visual grounding

This repository implements Florence-2 inference with BF16 precision, optimized for AWS Inferentia2 NeuronCores.

## Two Implementation Methods

### Standard BF16 (`models/florence2_bf16/`)

**Direct torch-neuronx implementation** - Simple and lightweight.

- ✅ Minimal dependencies and setup
- ✅ ~200 lines of core code
- ✅ Single NeuronCore inference
- ✅ Pre-compiled models included
- ❌ No tensor parallelism
- ❌ No vLLM server support

**Best for:** Quick experiments, learning, simple applications

### NxD BF16 (`models/florence2_nxd/`)

**neuronx-distributed-inference framework** - Production-ready with advanced features.

- ✅ Tensor parallelism support (TP 1/2/4/8)
- ✅ vLLM server with OpenAI-compatible REST API
- ✅ Request scheduling and queuing
- ✅ Production-grade serving
- ⚠️ More complex setup
- ⚠️ Requires compilation

**Best for:** Production deployments, high throughput, vLLM integration

## Quick Start

### Standard BF16 (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference (pre-compiled models included)
python -m models.florence2_bf16.inference --image image.jpg --task "<CAPTION>"

# Or use Python API
python examples/quick_start.py --image image.jpg
```

### NxD BF16 (15 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Compile models
python -m models.florence2_nxd.compile --output-dir ./compiled_nxd

# Option 1: Offline inference
python -c "
from models.florence2_nxd import Florence2NxDModel
model = Florence2NxDModel('./compiled_nxd', tp_degree=1)
result = model.generate('image.jpg', '<CAPTION>')
print(result)
"

# Option 2: Start vLLM server
python -m models.florence2_nxd.vllm_server \
    --model-dir ./compiled_nxd \
    --port 8000
```

## Requirements

### Hardware
- AWS Inferentia2 instance (inf2.xlarge or larger)
- At least 16GB system memory
- NeuronCore access

### Software
- Python 3.8+
- AWS Neuron SDK 2.1+
- PyTorch with Neuron support
- See `requirements.txt` for complete dependencies

## Project Structure

```
neuronx-distributed-inference-main/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── models/
│   ├── florence2_bf16/           # Standard BF16 implementation
│   │   ├── compile.py            # Compilation script
│   │   ├── inference.py          # Inference engine
│   │   ├── benchmark.py          # Benchmarking tools
│   │   └── README.md             # Technical documentation
│   │
│   └── florence2_nxd/            # NxD BF16 implementation
│       ├── compile.py            # NxD compilation
│       ├── model.py              # NxD inference model
│       ├── vllm_server.py        # vLLM server
│       ├── vllm_plugin.py        # vLLM plugin
│       ├── README.md             # Technical documentation
│       └── ...                   # Additional modules
│
├── compiled_bf16/                # Pre-compiled Standard BF16 models
│   ├── stage0-3.pt              # Vision encoder stages
│   ├── projection.pt            # Projection layer
│   ├── encoder.pt               # Language encoder
│   └── decoder_*.pt             # Decoder buckets
│
└── examples/
    └── quick_start.py            # End-to-end example
```

## Supported Tasks

| Task | Prompt | Description |
|------|--------|-------------|
| Caption | `<CAPTION>` | Brief image description |
| Detailed Caption | `<DETAILED_CAPTION>` | Comprehensive image description |
| Object Detection | `<OD>` | Detect and locate objects |
| OCR | `<OCR>` | Extract text from image |
| Region Caption | `<REGION_CAPTION>` | Describe specific region |

## Python API Examples

### Standard BF16

```python
from models.florence2_bf16.inference import Florence2NeuronBF16

# Initialize model
model = Florence2NeuronBF16("./compiled_bf16")

# Run inference
result = model("image.jpg", "<CAPTION>")
print(result)  # "A photo of a cat sitting on a windowsill"
```

### NxD BF16

```python
from models.florence2_nxd import Florence2NxDModel

# Initialize with tensor parallelism
model = Florence2NxDModel(
    model_dir="./compiled_nxd",
    tp_degree=1  # 1, 2, 4, or 8 NeuronCores
)

# Run inference
result = model.generate(
    image="image.jpg",
    task="<CAPTION>",
    max_new_tokens=100
)
print(result)
```

### vLLM Server

```bash
# Start server
python -m models.florence2_nxd.vllm_server \
    --model-dir ./compiled_nxd \
    --tp-degree 1 \
    --port 8000

# Send request (OpenAI-compatible)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "florence-2",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "<CAPTION>"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }]
  }'
```

## Documentation

- **models/florence2_bf16/README.md** - Standard BF16 technical details
- **models/florence2_nxd/README.md** - NxD BF16 comprehensive documentation

## Choosing Between Methods

**Use Standard BF16 if you want:**
- Quickest path to working inference
- Simple experimentation and learning
- Minimal code complexity
- Single-image processing

**Use NxD BF16 if you need:**
- Production-grade serving
- High throughput with request queuing
- vLLM server with REST API
- Tensor parallelism support
- Advanced deployment features

## Migration

Both methods produce identical outputs using the same Florence-2-base model with BF16 precision. You can start with Standard BF16 for development and migrate to NxD BF16 for production deployment.

## Resources

- [Florence-2 Model Card](https://huggingface.co/microsoft/Florence-2-base)
- [AWS Inferentia2](https://aws.amazon.com/machine-learning/inferentia/)
- [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)

## License

This project uses the Florence-2 model from Microsoft, which is available under the MIT license.
