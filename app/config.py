"""
Configuration for AI Video Generation App
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR = BASE_DIR / "uploads"

# Ensure directories exist
for d in [MODELS_DIR, OUTPUTS_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Server config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Model config
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "cerspense/zeroscope_v2_576w")
ALLOW_NSFW = os.getenv("ALLOW_NSFW", "false").lower() == "true"
DEVICE = os.getenv("DEVICE", "auto")  # auto, cuda, cpu, mps
DTYPE = os.getenv("DTYPE", "auto")  # auto, fp16, bf16, fp32

# Detect if we're on a CPU-only environment (like GitHub Actions runner)
def _is_cpu_only():
    try:
        import torch
        return not torch.cuda.is_available()
    except ImportError:
        return True

try:
    _CPU_ONLY = _is_cpu_only()
except Exception:
    # If torch not installed yet, check env var
    _CPU_ONLY = os.getenv("DEVICE", "auto") == "cpu"

# Generation defaults - CPU-optimized when no GPU
if _CPU_ONLY:
    DEFAULT_NUM_FRAMES = int(os.getenv("DEFAULT_NUM_FRAMES", "8"))
    DEFAULT_NUM_INFERENCE_STEPS = int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", "15"))
    DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "256"))
    DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "256"))
else:
    DEFAULT_NUM_FRAMES = int(os.getenv("DEFAULT_NUM_FRAMES", "14"))
    DEFAULT_NUM_INFERENCE_STEPS = int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", "25"))
    DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
    DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "576"))

DEFAULT_FPS = int(os.getenv("DEFAULT_FPS", "6"))
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "-1"))  # -1 = random
DEFAULT_MOTION_BUCKET_ID = int(os.getenv("DEFAULT_MOTION_BUCKET_ID", "127"))
DEFAULT_NOISE_AUG_STRENGTH = float(os.getenv("DEFAULT_NOISE_AUG_STRENGTH", "0.02"))
DEFAULT_CFG_SCALE = float(os.getenv("DEFAULT_CFG_SCALE", "3.0"))

# Model registry - known model types and their configs
MODEL_REGISTRY = {
    "stabilityai/stable-video-diffusion-img2vid-xt": {
        "type": "svd",
        "pipeline": "StableVideoDiffusionPipeline",
        "description": "Stable Video Diffusion - Image to Video (XT)",
        "default_frames": 14,
        "max_frames": 25,
        "supports_text2video": False,
        "cpu_note": "Very slow on CPU - use low resolution (256x256, 8 frames)",
    },
    "stabilityai/stable-video-diffusion-img2vid": {
        "type": "svd",
        "pipeline": "StableVideoDiffusionPipeline",
        "description": "Stable Video Diffusion - Image to Video",
        "default_frames": 14,
        "max_frames": 14,
        "supports_text2video": False,
        "cpu_note": "Slow on CPU - use low resolution",
    },
    "cerspense/zeroscope_v2_576w": {
        "type": "text2video",
        "pipeline": "TextToVideoSDPipeline",
        "description": "Zeroscope V2 - Text to Video (CPU-friendly)",
        "default_frames": 8,
        "max_frames": 24,
        "supports_text2video": True,
        "recommended": True,
    },
    "cerspense/zeroscope_v2_XL": {
        "type": "text2video",
        "pipeline": "TextToVideoSDPipeline",
        "description": "Zeroscope V2 XL - Text to Video (Higher Quality)",
        "default_frames": 8,
        "max_frames": 24,
        "supports_text2video": True,
    },
    "damo-vilab/text-to-video-ms-1.7b": {
        "type": "text2video",
        "pipeline": "TextToVideoSDPipeline",
        "description": "ModelScope Text-to-Video (1.7B - smaller, faster on CPU)",
        "default_frames": 8,
        "max_frames": 16,
        "supports_text2video": True,
        "recommended": True,
    },
}

# CORS
CORS_ORIGINS = ["*"]

# Max upload size (50MB)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024
