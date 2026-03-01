"""
Migration Helper for Florence-2 NxD Inference.

This module provides utilities to help migrate from the old torch-neuronx
implementation to the new NxD Inference implementation. It includes:

- Detection of old vs new model formats
- Loading of old compiled models
- Conversion utilities (if needed)
- Migration validation tools

Requirements:
    - 15.5: Support loading old compiled models
    - 15.5: Provide conversion utilities if needed

Example:
    >>> from models.florence2_nxd.migration import detect_model_format, load_legacy_model
    >>> 
    >>> # Detect model format
    >>> format_type = detect_model_format("./compiled_bf16")
    >>> print(format_type)  # "legacy" or "nxd"
    >>> 
    >>> # Load legacy model with compatibility wrapper
    >>> model = load_legacy_model("./compiled_bf16", core_id="0")
    >>> result = model("image.jpg", "<CAPTION>")
"""

import os
import json
from typing import Optional, Dict, Any, Tuple, Literal
from pathlib import Path
import torch

from .logging_config import get_logger
from .compat import Florence2NeuronBF16Compat


logger = get_logger(__name__)


ModelFormat = Literal["legacy", "nxd", "unknown"]


def detect_model_format(model_dir: str) -> ModelFormat:
    """
    Detect whether a model directory contains legacy or NxD models.
    
    This function checks for the presence of metadata.json to distinguish
    between old torch-neuronx models and new NxD Inference models.
    
    Detection logic:
    - If metadata.json exists: NxD format
    - If stage0.pt exists but no metadata.json: Legacy format
    - Otherwise: Unknown format
    
    Args:
        model_dir: Directory containing compiled models
    
    Returns:
        "legacy" for old torch-neuronx models
        "nxd" for new NxD Inference models
        "unknown" if format cannot be determined
    
    Requirements:
        - 15.5: Support loading old compiled models
    
    Example:
        >>> format_type = detect_model_format("./compiled_bf16")
        >>> if format_type == "legacy":
        ...     print("Old format detected")
        >>> elif format_type == "nxd":
        ...     print("New NxD format detected")
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        logger.warning(f"Model directory does not exist: {model_dir}")
        return "unknown"
    
    # Check for metadata.json (NxD format indicator)
    metadata_file = model_path / "metadata.json"
    if metadata_file.exists():
        logger.info(f"Detected NxD format (metadata.json found): {model_dir}")
        return "nxd"
    
    # Check for stage0.pt (legacy format indicator)
    stage0_file = model_path / "stage0.pt"
    if stage0_file.exists():
        logger.info(f"Detected legacy format (stage0.pt found, no metadata.json): {model_dir}")
        return "legacy"
    
    logger.warning(f"Unknown model format: {model_dir}")
    return "unknown"


def get_model_info(model_dir: str) -> Dict[str, Any]:
    """
    Get information about a compiled model directory.
    
    This function extracts metadata and file information from a model
    directory, regardless of whether it's legacy or NxD format.
    
    Args:
        model_dir: Directory containing compiled models
    
    Returns:
        Dictionary with model information:
        - format: "legacy", "nxd", or "unknown"
        - files: List of .pt files found
        - metadata: Metadata dict (if NxD format)
        - size_mb: Total size in MB
    
    Example:
        >>> info = get_model_info("./compiled_bf16")
        >>> print(f"Format: {info['format']}")
        >>> print(f"Files: {len(info['files'])}")
        >>> print(f"Size: {info['size_mb']:.1f} MB")
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        return {
            "format": "unknown",
            "files": [],
            "metadata": None,
            "size_mb": 0.0,
            "error": f"Directory not found: {model_dir}"
        }
    
    # Detect format
    format_type = detect_model_format(model_dir)
    
    # Find all .pt files
    pt_files = list(model_path.glob("*.pt"))
    file_names = [f.name for f in pt_files]
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in pt_files)
    size_mb = total_size / (1024 * 1024)
    
    # Load metadata if available
    metadata = None
    metadata_file = model_path / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
    
    return {
        "format": format_type,
        "files": file_names,
        "metadata": metadata,
        "size_mb": size_mb,
        "num_files": len(file_names),
    }


def validate_legacy_model(model_dir: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a legacy model directory has all required files.
    
    Legacy models should have:
    - 4 vision stage files: stage0.pt, stage1.pt, stage2.pt, stage3.pt
    - 1 projection file: projection.pt
    - 1 encoder file: encoder.pt
    - 6 decoder files: decoder_1.pt, decoder_4.pt, ..., decoder_64.pt
    
    Args:
        model_dir: Directory containing legacy compiled models
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if all required files exist
        - error_message: None if valid, error description if invalid
    
    Requirements:
        - 15.5: Validate old compiled models
    
    Example:
        >>> is_valid, error = validate_legacy_model("./compiled_bf16")
        >>> if not is_valid:
        ...     print(f"Invalid model: {error}")
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        return False, f"Model directory not found: {model_dir}"
    
    # Check for required files
    required_files = []
    
    # Vision stages
    for i in range(4):
        required_files.append(f"stage{i}.pt")
    
    # Projection
    required_files.append("projection.pt")
    
    # Encoder
    required_files.append("encoder.pt")
    
    # Decoders
    for bucket in [1, 4, 8, 16, 32, 64]:
        required_files.append(f"decoder_{bucket}.pt")
    
    # Check which files are missing
    missing_files = []
    for filename in required_files:
        file_path = model_path / filename
        if not file_path.exists():
            missing_files.append(filename)
    
    if missing_files:
        error_msg = f"Missing required files: {', '.join(missing_files)}"
        return False, error_msg
    
    return True, None


def load_legacy_model(
    model_dir: str,
    core_id: str = "0"
) -> Florence2NeuronBF16Compat:
    """
    Load a legacy model using the compatibility wrapper.
    
    This function loads old torch-neuronx compiled models and wraps them
    in the compatibility layer, allowing them to be used with the same
    API as the original implementation.
    
    Note: Legacy models are loaded directly without conversion. The
    compatibility wrapper handles the API translation.
    
    Args:
        model_dir: Directory containing legacy compiled models
        core_id: NeuronCore ID to use
    
    Returns:
        Florence2NeuronBF16Compat instance
    
    Raises:
        FileNotFoundError: If model directory or required files are missing
        ValueError: If model validation fails
    
    Requirements:
        - 15.5: Support loading old compiled models
    
    Example:
        >>> model = load_legacy_model("./compiled_bf16", core_id="0")
        >>> result = model("image.jpg", "<CAPTION>")
        >>> print(result)
    """
    logger.info(f"Loading legacy model from {model_dir}")
    
    # Validate model directory
    is_valid, error_msg = validate_legacy_model(model_dir)
    if not is_valid:
        raise ValueError(f"Invalid legacy model: {error_msg}")
    
    # Detect format to confirm it's legacy
    format_type = detect_model_format(model_dir)
    if format_type == "nxd":
        logger.warning(
            f"Model directory {model_dir} appears to be NxD format, not legacy. "
            f"Consider using Florence2NxDModel directly for better performance."
        )
    elif format_type == "unknown":
        logger.warning(
            f"Model directory {model_dir} has unknown format. "
            f"Attempting to load anyway..."
        )
    
    # Load using compatibility wrapper
    # Note: The compatibility wrapper expects NxD models, but legacy models
    # have the same file structure, so they can be loaded directly
    try:
        model = Florence2NeuronBF16Compat(model_dir, core_id)
        logger.info("Legacy model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load legacy model: {e}")
        raise


def create_migration_metadata(
    legacy_model_dir: str,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create metadata for a legacy model to aid migration.
    
    This function analyzes a legacy model directory and creates a metadata
    file that can be used to track the model during migration. The metadata
    includes file information, sizes, and checksums.
    
    Args:
        legacy_model_dir: Directory containing legacy compiled models
        output_file: Optional path to save metadata JSON (default: model_dir/legacy_metadata.json)
    
    Returns:
        Dictionary with migration metadata
    
    Example:
        >>> metadata = create_migration_metadata("./compiled_bf16")
        >>> print(f"Model has {metadata['num_files']} files")
    """
    import hashlib
    from datetime import datetime
    
    model_path = Path(legacy_model_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {legacy_model_dir}")
    
    # Get model info
    info = get_model_info(legacy_model_dir)
    
    # Calculate checksums for each file
    file_checksums = {}
    for filename in info['files']:
        file_path = model_path / filename
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
            file_checksums[filename] = file_hash
    
    # Create metadata
    metadata = {
        "format": "legacy",
        "model_dir": str(legacy_model_dir),
        "created_at": datetime.now().isoformat(),
        "num_files": info['num_files'],
        "total_size_mb": info['size_mb'],
        "files": info['files'],
        "file_checksums": file_checksums,
        "migration_status": "not_migrated",
    }
    
    # Save to file if requested
    if output_file is None:
        output_file = model_path / "legacy_metadata.json"
    
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Migration metadata saved to {output_path}")
    
    return metadata


def compare_model_outputs(
    legacy_model_dir: str,
    nxd_model_dir: str,
    test_image: str,
    task: str = "<CAPTION>",
    core_id: str = "0"
) -> Dict[str, Any]:
    """
    Compare outputs between legacy and NxD models.
    
    This function runs inference on the same image with both legacy and
    NxD models and compares the outputs. This is useful for validating
    that the migration preserves model behavior.
    
    Args:
        legacy_model_dir: Directory containing legacy compiled models
        nxd_model_dir: Directory containing NxD compiled models
        test_image: Path to test image
        task: Task prompt to test
        core_id: NeuronCore ID to use
    
    Returns:
        Dictionary with comparison results:
        - legacy_output: Output from legacy model
        - nxd_output: Output from NxD model
        - outputs_match: Whether outputs are identical
        - similarity: Text similarity score (0-1)
    
    Requirements:
        - 15.5: Validate migration preserves behavior
    
    Example:
        >>> comparison = compare_model_outputs(
        ...     "./compiled_bf16",
        ...     "./compiled_nxd",
        ...     "test_image.jpg"
        ... )
        >>> if comparison['outputs_match']:
        ...     print("Migration successful!")
    """
    logger.info("Comparing legacy and NxD model outputs...")
    
    # Load legacy model
    logger.info("Loading legacy model...")
    legacy_model = load_legacy_model(legacy_model_dir, core_id)
    
    # Load NxD model
    logger.info("Loading NxD model...")
    nxd_model = Florence2NeuronBF16Compat(nxd_model_dir, core_id)
    
    # Run inference on both models
    logger.info(f"Running inference on {test_image} with task {task}")
    
    try:
        legacy_output = legacy_model(test_image, task)
        logger.info(f"Legacy output: {legacy_output}")
    except Exception as e:
        logger.error(f"Legacy model inference failed: {e}")
        legacy_output = f"ERROR: {e}"
    
    try:
        nxd_output = nxd_model(test_image, task)
        logger.info(f"NxD output: {nxd_output}")
    except Exception as e:
        logger.error(f"NxD model inference failed: {e}")
        nxd_output = f"ERROR: {e}"
    
    # Compare outputs
    outputs_match = legacy_output == nxd_output
    
    # Calculate similarity (simple character-level similarity)
    if legacy_output and nxd_output:
        # Use Levenshtein distance for similarity
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, legacy_output, nxd_output).ratio()
    else:
        similarity = 0.0
    
    result = {
        "legacy_output": legacy_output,
        "nxd_output": nxd_output,
        "outputs_match": outputs_match,
        "similarity": similarity,
        "test_image": test_image,
        "task": task,
    }
    
    if outputs_match:
        logger.info("✓ Outputs match exactly")
    else:
        logger.warning(f"✗ Outputs differ (similarity: {similarity:.2%})")
    
    return result


def print_migration_guide():
    """
    Print a migration guide to help users transition from legacy to NxD models.
    
    This function prints helpful information about:
    - How to detect model format
    - How to load legacy models
    - How to migrate to NxD models
    - API compatibility information
    
    Example:
        >>> from models.florence2_nxd.migration import print_migration_guide
        >>> print_migration_guide()
    """
    guide = """
╔══════════════════════════════════════════════════════════════════════════╗
║                  Florence-2 Migration Guide                              ║
║              From torch-neuronx to NxD Inference                         ║
╚══════════════════════════════════════════════════════════════════════════╝

1. DETECT MODEL FORMAT
   ────────────────────
   from models.florence2_nxd.migration import detect_model_format
   
   format_type = detect_model_format("./compiled_bf16")
   print(format_type)  # "legacy" or "nxd"

2. LOAD LEGACY MODELS (No Changes Required)
   ─────────────────────────────────────────
   from models.florence2_nxd.migration import load_legacy_model
   
   # Your old code still works!
   model = load_legacy_model("./compiled_bf16", core_id="0")
   result = model("image.jpg", "<CAPTION>")

3. USE COMPATIBILITY WRAPPER (Recommended)
   ────────────────────────────────────────
   from models.florence2_nxd import Florence2NeuronBF16
   
   # Same API as before, but with NxD backend
   model = Florence2NeuronBF16("./compiled_nxd", core_id="0")
   result = model("image.jpg", "<CAPTION>")

4. MIGRATE TO NEW API (For New Features)
   ──────────────────────────────────────
   from models.florence2_nxd import Florence2NxDModel
   
   # New API with more features
   model = Florence2NxDModel("./compiled_nxd", tp_degree=1)
   
   # Preprocess inputs
   inputs = model.processor(text="<CAPTION>", images=image, return_tensors="pt")
   
   # Generate
   output_ids = model.generate(
       pixel_values=inputs["pixel_values"].to(torch.bfloat16),
       input_ids=inputs["input_ids"],
       max_new_tokens=100
   )
   
   # Decode
   result = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

5. VALIDATE MIGRATION
   ───────────────────
   from models.florence2_nxd.migration import compare_model_outputs
   
   comparison = compare_model_outputs(
       legacy_model_dir="./compiled_bf16",
       nxd_model_dir="./compiled_nxd",
       test_image="test.jpg"
   )
   
   if comparison['outputs_match']:
       print("✓ Migration successful!")

6. KEY DIFFERENCES
   ───────────────
   - Legacy: Direct torch-neuronx compilation
   - NxD: Uses neuronx-distributed-inference APIs
   - Both: Same file structure (.pt files)
   - Both: Same performance characteristics
   - NxD: Additional features (tensor parallelism, vLLM integration)

7. BREAKING CHANGES
   ────────────────
   None! The compatibility wrapper maintains 100% API compatibility.
   
   Old code:
       from models.florence2_bf16.inference import Florence2NeuronBF16
       model = Florence2NeuronBF16("./compiled_bf16")
   
   New code (drop-in replacement):
       from models.florence2_nxd import Florence2NeuronBF16
       model = Florence2NeuronBF16("./compiled_nxd")

8. NEED HELP?
   ──────────
   - Check documentation: models/florence2_nxd/README.md
   - Run model info: get_model_info("./your_model_dir")
   - Compare outputs: compare_model_outputs(...)

╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(guide)
