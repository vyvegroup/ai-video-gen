"""
Model Manager - Download, load, and manage AI video generation models.
Supports HuggingFace repos and direct safetensors URLs.
"""
import os
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import requests
from huggingface_hub import (
    snapshot_download,
    hf_hub_download,
    login,
    HfFileSystem,
)

from app.config import MODELS_DIR, MODEL_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a downloaded model."""
    name: str
    path: str
    type: str  # "svd", "text2video", "custom"
    source: str  # "huggingface", "url", "local"
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    loaded: bool = False
    size_mb: float = 0.0


class ModelManager:
    """Manage video generation models - download, load, switch."""

    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._loaded_pipeline = None
        self._current_model_name: Optional[str] = None
        self._scan_existing_models()

    def _scan_existing_models(self) -> None:
        """Scan models directory for existing models."""
        if not MODELS_DIR.exists():
            return

        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                config_file = model_dir / "config.json"
                index_file = model_dir / "model_index.json"
                safetensors_file = model_dir / "model.safetensors"

                if config_file.exists() or index_file.exists() or any(model_dir.glob("*.safetensors")):
                    model_name = model_dir.name
                    registry_info = MODEL_REGISTRY.get(model_name, {})
                    self.models[model_name] = ModelInfo(
                        name=model_name,
                        path=str(model_dir),
                        type=registry_info.get("type", "custom"),
                        source="huggingface",
                        description=registry_info.get("description", "Custom model"),
                        config=registry_info,
                    )
                    # Calculate size
                    total_size = sum(
                        f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                    )
                    self.models[model_name].size_mb = round(total_size / (1024 * 1024), 2)

        logger.info(f"Found {len(self.models)} existing models: {list(self.models.keys())}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        result = []
        for name, info in self.models.items():
            result.append({
                "name": name,
                "type": info.type,
                "source": info.source,
                "description": info.description,
                "path": info.path,
                "loaded": info.loaded,
                "size_mb": info.size_mb,
                "supports_text2video": info.config.get("supports_text2video", False),
            })
        return result

    def download_model(
        self,
        model_id: str,
        source: str = "huggingface",
        hf_token: Optional[str] = None,
    ) -> ModelInfo:
        """
        Download a model from HuggingFace or direct URL.

        Args:
            model_id: HuggingFace repo ID or direct URL
            source: "huggingface" or "url"
            hf_token: Optional HuggingFace token for gated models
        """
        # Create a safe directory name
        safe_name = model_id.replace("/", "_").replace(":", "_").replace("https://", "").replace("http://", "")
        model_path = MODELS_DIR / safe_name

        if source == "url":
            return self._download_from_url(model_id, safe_name, model_path)
        else:
            return self._download_from_huggingface(model_id, safe_name, model_path, hf_token)

    def _download_from_huggingface(
        self,
        model_id: str,
        safe_name: str,
        model_path: Path,
        hf_token: Optional[str] = None,
    ) -> ModelInfo:
        """Download model from HuggingFace Hub."""
        logger.info(f"Downloading model from HuggingFace: {model_id}")

        if hf_token:
            login(token=hf_token)

        # Determine if it's a diffusers model or single file
        try:
            model_path.mkdir(parents=True, exist_ok=True)

            # Set HF cache to /tmp on CI runners to avoid disk space issues
            cache_dir = os.environ.get("HF_HOME", None)

            # Try to download as a diffusers pipeline
            snapshot_download(
                repo_id=model_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                cache_dir=cache_dir,
            )

            registry_info = MODEL_REGISTRY.get(model_id, {})
            model_type = registry_info.get("type", "custom")

            # Try to detect model type from config
            config_path = model_path / "config.json"
            model_index_path = model_path / "model_index.json"

            if model_index_path.exists():
                import json
                with open(model_index_path) as f:
                    idx = json.load(f)
                    class_name = idx.get("_class_name", "")
                    if "StableVideoDiffusion" in class_name:
                        model_type = "svd"
                    elif "TextToVideo" in class_name:
                        model_type = "text2video"
                    else:
                        model_type = "text2video"  # Default for unknown diffusers models
                    logger.info(f"Detected model type: {model_type}")

            info = ModelInfo(
                name=safe_name,
                path=str(model_path),
                type=model_type,
                source="huggingface",
                description=registry_info.get("description", f"HuggingFace: {model_id}"),
                config=registry_info,
            )

            # Calculate size
            total_size = sum(
                f.stat().st_size for f in model_path.rglob("*") if f.is_file()
            )
            info.size_mb = round(total_size / (1024 * 1024), 2)

            self.models[safe_name] = info
            logger.info(f"Model downloaded: {model_id} -> {model_path} ({info.size_mb} MB)")
            return info

        except Exception as e:
            logger.error(f"Failed to download from HuggingFace: {e}")
            # Clean up partial download
            if model_path.exists():
                shutil.rmtree(model_path)
            raise RuntimeError(f"Failed to download model: {e}")

    def _download_from_url(
        self,
        url: str,
        safe_name: str,
        model_path: Path,
    ) -> ModelInfo:
        """Download model from a direct URL (safetensors, zip, etc.)."""
        logger.info(f"Downloading model from URL: {url}")

        try:
            model_path.mkdir(parents=True, exist_ok=True)

            # Determine file type from URL or headers
            filename = url.split("/")[-1].split("?")[0]

            if filename.endswith(".zip"):
                zip_path = model_path / filename
                self._download_file(url, zip_path)
                logger.info("Extracting zip archive...")
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(model_path)
                zip_path.unlink()
            elif filename.endswith(".safetensors"):
                safetensors_path = model_path / filename
                self._download_file(url, safetensors_path)
            else:
                # Try to download and detect
                filepath = model_path / "model.safetensors"
                self._download_file(url, filepath)

            # Calculate size
            total_size = sum(
                f.stat().st_size for f in model_path.rglob("*") if f.is_file()
            )
            size_mb = round(total_size / (1024 * 1024), 2)

            # Detect model type
            model_type = "custom"

            # Check if there's a model_index.json (diffusers format)
            if (model_path / "model_index.json").exists():
                import json
                with open(model_path / "model_index.json") as f:
                    idx = json.load(f)
                    if "StableVideoDiffusion" in idx.get("_class_name", ""):
                        model_type = "svd"
                    elif "TextToVideo" in idx.get("_class_name", ""):
                        model_type = "text2video"

            info = ModelInfo(
                name=safe_name,
                path=str(model_path),
                type=model_type,
                source="url",
                description=f"Downloaded from {url[:80]}...",
                config={"supports_text2video": model_type == "text2video"},
            )
            info.size_mb = size_mb

            self.models[safe_name] = info
            logger.info(f"Model downloaded from URL: {model_path} ({size_mb} MB)")
            return info

        except Exception as e:
            logger.error(f"Failed to download from URL: {e}")
            if model_path.exists():
                shutil.rmtree(model_path)
            raise RuntimeError(f"Failed to download model from URL: {e}")

    def _download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> None:
        """Download a file with progress."""
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"  Download progress: {progress:.1f}%")

    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model."""
        if model_name not in self.models:
            return False

        info = self.models[model_name]

        # Unload if currently loaded
        if self._current_model_name == model_name:
            self.unload_model()

        # Delete files
        model_path = Path(info.path)
        if model_path.exists() and MODELS_DIR in model_path.parents:
            shutil.rmtree(model_path)

        del self.models[model_name]
        logger.info(f"Model deleted: {model_name}")
        return True

    def load_model(self, model_name: str, allow_nsfw: bool = False):
        """Load a model into memory for inference."""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        # Unload current model if different
        if self._current_model_name and self._current_model_name != model_name:
            self.unload_model()

        if self._loaded_pipeline is not None:
            return self._loaded_pipeline

        info = self.models[model_name]
        logger.info(f"Loading model: {model_name} (type: {info.type})")

        import torch
        from diffusers import (
            StableVideoDiffusionPipeline,
            TextToVideoSDPipeline,
            DiffusionPipeline,
        )

        # Determine device and dtype
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        # Free memory before loading
        torch.cuda.empty_cache()

        pipeline = None

        try:
            if info.type == "svd":
                pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    info.path,
                    torch_dtype=dtype,
                    variant="fp16" if device == "cuda" else None,
                )
            elif info.type == "text2video":
                pipeline = TextToVideoSDPipeline.from_pretrained(
                    info.path,
                    torch_dtype=dtype,
                    variant="fp16" if device == "cuda" else None,
                )
            else:
                # Try auto-detection via DiffusionPipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    info.path,
                    torch_dtype=dtype,
                )

            # Disable safety checker if NSFW allowed
            if allow_nsfw and hasattr(pipeline, "safety_checker"):
                pipeline.safety_checker = None
                logger.warning("NSFW safety checker disabled")

            # Enable memory optimizations
            if device == "cuda":
                pipeline.enable_model_cpu_offload()
                pipeline.enable_vae_slicing()
                if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                    try:
                        pipeline.enable_xformers_memory_efficient_attention()
                    except ImportError:
                        pass
            else:
                # CPU-specific optimizations
                # Set attention slicing to reduce memory usage on CPU
                pipeline.enable_attention_slicing("max")
                if hasattr(pipeline, 'enable_vae_slicing'):
                    pipeline.enable_vae_slicing()
                logger.info("CPU optimizations enabled: attention_slicing, vae_slicing")

            self._loaded_pipeline = pipeline
            self._current_model_name = model_name
            info.loaded = True

            logger.info(f"Model loaded successfully: {model_name}")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    def unload_model(self) -> None:
        """Unload current model from memory."""
        if self._loaded_pipeline is not None:
            import torch
            del self._loaded_pipeline
            self._loaded_pipeline = None
            torch.cuda.empty_cache()

            if self._current_model_name and self._current_model_name in self.models:
                self.models[self._current_model_name].loaded = False

            self._current_model_name = None
            logger.info("Model unloaded")

    def get_current_model(self) -> Optional[str]:
        """Get the name of the currently loaded model."""
        return self._current_model_name

    def get_pipeline(self):
        """Get the currently loaded pipeline."""
        return self._loaded_pipeline


# Global model manager instance
model_manager = ModelManager()
