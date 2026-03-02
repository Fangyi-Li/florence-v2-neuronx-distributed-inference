# Florence-2 NxD Inference

Production-ready Florence-2 vision-language model integration with neuronx-distributed-inference (NxD) for AWS Inferentia2.

## Features

- **NxD Inference Integration**: Standardized model compilation and inference APIs
- **Stage-wise Compilation**: Preserves DaViT's 4-stage architecture for optimal performance
- **Bucket Strategy**: Variable-length decoding with 6 bucket sizes (1, 4, 8, 16, 32, 64 tokens)
- **Tensor Parallelism**: Supports TP degrees 1, 2, 4, 8 for distributed inference
- **OpenAI-Compatible API**: Production-ready REST API server
- **Concurrent Requests**: Request scheduling and queuing for high throughput
- **BF16 Precision**: Maintains BF16 throughout pipeline for performance

## Files

```
florence2_nxd/
├── compile.py              # Model compilation script
├── model.py                # Florence2NxDModel inference engine
├── vllm_server.py          # OpenAI-compatible API server
├── vllm_plugin.py          # Server plugin interface
├── openai_protocol.py      # OpenAI protocol definitions
├── request_scheduler.py    # Request scheduling
├── config.py               # Configuration classes
├── errors.py               # Custom error types
├── metadata.py             # Model metadata
├── nxd_wrappers.py         # NxD wrapper modules
└── README.md               # This file
```

## Usage

### 1. Compilation

Compile Florence-2 models for Inferentia2 (takes ~15-20 minutes):

```bash
python -m models.florence2_nxd.compile --output-dir ./compiled_nxd
```

**Output**: 12 compiled model files in `./compiled_nxd/`:
- `stage0.pt`, `stage1.pt`, `stage2.pt`, `stage3.pt` - Vision encoder stages
- `projection.pt` - Vision-to-language projection
- `encoder.pt` - BART encoder
- `decoder_1.pt`, `decoder_4.pt`, ..., `decoder_64.pt` - BART decoder buckets
- `metadata.json` - Model metadata

#### Tensor Parallelism

Enable multi-core deployment:

```bash
# 2 NeuronCores (recommended for production)
python -m models.florence2_nxd.compile --tp-degree 2 --output-dir ./compiled_tp2

# 4 NeuronCores (high throughput)
python -m models.florence2_nxd.compile --tp-degree 4 --output-dir ./compiled_tp4
```

### 2. Offline Inference

Use compiled models directly in Python:

```python
from models.florence2_nxd import Florence2NxDModel

# Initialize model
model = Florence2NxDModel(model_dir="./compiled_nxd", tp_degree=1)

# Image captioning
caption = model.generate(image="photo.jpg", task="<CAPTION>")
print(caption)

# Object detection
objects = model.generate(image="photo.jpg", task="<OD>")

# OCR
text = model.generate(image="document.jpg", task="<OCR>")
```

### 3. Production Server

Start an OpenAI-compatible REST API server:

```bash
python -m models.florence2_nxd.vllm_server \
    --model-dir ./compiled_nxd \
    --tp-degree 1 \
    --port 8000
```

#### Send Requests

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "florence-2",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "<CAPTION>"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ]
  }'
```

#### Python Client

```python
import requests
import base64

# Encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "florence-2",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "<CAPTION>"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"
                }}
            ]
        }],
        "max_tokens": 100
    }
)

result = response.json()
caption = result["choices"][0]["message"]["content"]
print(caption)
```

## Supported Tasks

| Task | Description | Prompt |
|------|-------------|--------|
| Caption | Generate image caption | `<CAPTION>` |
| Detailed Caption | Generate detailed description | `<DETAILED_CAPTION>` |
| Object Detection | Detect objects with bounding boxes | `<OD>` |
| OCR | Extract text from image | `<OCR>` |
| Region Caption | Generate region-specific description | `<REGION_CAPTION>` |

## API Reference

### Florence2NxDModel

```python
class Florence2NxDModel:
    def __init__(self, model_dir: str, tp_degree: int = 1):
        """
        Initialize Florence-2 NxD inference model.

        Args:
            model_dir: Path to compiled model directory
            tp_degree: Tensor parallelism degree (1, 2, 4, or 8)
        """

    def generate(
        self,
        image: Union[str, PIL.Image],
        task: str,
        max_new_tokens: int = 100
    ) -> str:
        """
        Run inference on an image.

        Args:
            image: File path or PIL Image object
            task: Task prompt (e.g., "<CAPTION>")
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
```

### Server Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{"status": "healthy", "model_loaded": true}
```

#### `GET /v1/models`
List available models (OpenAI-compatible).

#### `POST /v1/chat/completions`
Generate completions (OpenAI-compatible).

#### `GET /stats`
Server statistics (requests, latency, etc.).

## Configuration

### Compilation Options

```bash
python -m models.florence2_nxd.compile \
    --model-name microsoft/Florence-2-base \  # Model from HuggingFace
    --output-dir ./compiled_nxd \             # Output directory
    --tp-degree 1 \                           # Tensor parallelism degree
    --batch-size 1                            # Batch size (fixed at 1)
```

### Server Options

```bash
python -m models.florence2_nxd.vllm_server \
    --model-dir ./compiled_nxd \              # Compiled models path
    --tp-degree 1 \                           # Must match compilation
    --port 8000 \                             # Server port
    --host 0.0.0.0 \                          # Bind address
    --max-concurrent-requests 10 \            # Concurrent request limit
    --max-new-tokens 150 \                    # Default max tokens
    --log-level INFO                          # Logging level
```

## Architecture

```
┌─────────────────────────────────────────────┐
│      OpenAI-Compatible API Server           │
│  (FastAPI, request scheduling, queuing)     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         Florence2 Inference Plugin          │
│  (Multimodal processing, image pipeline)    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Florence2NxDModel (BF16)             │
│  Vision: DaViT (4 stages)                   │
│  Projection: Linear + LayerNorm + PositionEmb│
│  Encoder: BART encoder                      │
│  Decoder: BART decoder (6 buckets)          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        NxD Inference Runtime                │
│  (Model loading, TP coordination)           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        AWS Inferentia2 Hardware             │
└─────────────────────────────────────────────┘
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: stage0.pt` | Models not compiled | Run `compile.py` first |
| `NeuronCore not found` | No Neuron devices | Use Inferentia2 instance |
| `TP degree mismatch` | Runtime TP ≠ compilation TP | Use same `--tp-degree` for both |
| `CUDA/CPU fallback` | Wrong environment | Set `NEURON_RT_VISIBLE_CORES` |
| `Memory error during compile` | OOM | Use larger instance (inf2.8xlarge+) |
| `Server won't start` | Port in use | Change port with `--port` |
| High latency | Cold start | Add warmup inference |
| Low throughput | Underutilized cores | Check with `neuron-top` |

### Debug Mode

Enable detailed logging:
```bash
python -m models.florence2_nxd.vllm_server --log-level DEBUG
```

### Check NeuronCore Utilization

```bash
neuron-top
```

Look for:
- NeuronCore utilization > 80%
- Memory usage stable
- No errors in logs

## Best Practices

1. **Pre-compile models** before deployment (don't compile in production)
2. **Use TP=2** for production workloads (balances throughput and resources)
3. **Warm up** with 1-2 dummy inferences after server start
4. **Monitor** with `/stats` endpoint and `neuron-top`
5. **Set resource limits** on `max-concurrent-requests` to prevent overload
6. **Use reverse proxy** (nginx) for SSL/TLS in production

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torch-neuronx
- neuronx-distributed-inference
- transformers
- Pillow
- FastAPI and uvicorn (for server)
- AWS Inferentia2 instance

## License

Apache 2.0
