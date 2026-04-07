"""
AI Video Generation App - FastAPI Backend
Serves the web UI and handles model management + video generation.
"""
import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from app.config import (
    BASE_DIR,
    OUTPUTS_DIR,
    UPLOADS_DIR,
    MODELS_DIR,
    CORS_ORIGINS,
    MAX_UPLOAD_SIZE,
    ALLOW_NSFW,
)
from app.model_manager import model_manager
from app.generator import video_generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Video Generator",
    description="Self-hosted AI Video Generation with local model inference",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount outputs directory
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Mount uploads directory
if UPLOADS_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


# ============================================================
# PAGES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="index.html not found")


# ============================================================
# MODEL MANAGEMENT
# ============================================================

@app.get("/api/models")
async def list_models():
    """List all available models."""
    return {"models": model_manager.list_models(), "current": model_manager.get_current_model()}


@app.post("/api/models/download")
async def download_model(
    model_id: str = Form(...),
    source: str = Form("huggingface"),
    hf_token: Optional[str] = Form(None),
):
    """Download a new model."""
    try:
        info = model_manager.download_model(
            model_id=model_id,
            source=source,
            hf_token=hf_token,
        )
        return {
            "success": True,
            "message": f"Model downloaded: {info.name}",
            "model": {
                "name": info.name,
                "type": info.type,
                "source": info.source,
                "description": info.description,
                "size_mb": info.size_mb,
            },
        }
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a model."""
    if model_manager.delete_model(model_name):
        return {"success": True, "message": f"Model deleted: {model_name}"}
    raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")


@app.post("/api/models/load")
async def load_model(model_name: str = Form(...), allow_nsfw: bool = Form(False)):
    """Load a model into memory."""
    try:
        pipeline = model_manager.load_model(model_name, allow_nsfw=allow_nsfw)
        return {"success": True, "message": f"Model loaded: {model_name}"}
    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/unload")
async def unload_model():
    """Unload current model from memory."""
    model_manager.unload_model()
    return {"success": True, "message": "Model unloaded"}


@app.get("/api/models/registry")
async def get_model_registry():
    """Get the list of known model configurations."""
    return {"registry": model_manager.models.__dict__ if hasattr(model_manager.models, '__dict__') else {},
            "suggested": [
                {"id": "stabilityai/stable-video-diffusion-img2vid-xt", "type": "Image to Video", "description": "SVD XT - High quality image to video"},
                {"id": "stabilityai/stable-video-diffusion-img2vid", "type": "Image to Video", "description": "SVD - Standard image to video"},
                {"id": "cerspense/zeroscope_v2_576w", "type": "Text to Video", "description": "Zeroscope V2 - Text to video generation"},
                {"id": "cerspense/zeroscope_v2_XL", "type": "Text to Video", "description": "Zeroscope V2 XL - Higher quality text to video"},
            ]}


# ============================================================
# IMAGE UPLOAD
# ============================================================

@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for image-to-video generation."""
    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_SIZE // (1024*1024)}MB")

    # Validate file type
    allowed_types = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

    # Save file
    filename = f"{uuid_str()}_{file.filename}"
    file_path = UPLOADS_DIR / filename
    with open(file_path, "wb") as f:
        f.write(contents)

    return {
        "success": True,
        "filename": filename,
        "path": str(file_path),
        "url": f"/uploads/{filename}",
        "size": len(contents),
    }


def uuid_str():
    import uuid
    return str(uuid.uuid4())[:8]


@app.get("/api/uploads")
async def list_uploads():
    """List uploaded images."""
    uploads = []
    if UPLOADS_DIR.exists():
        for f in sorted(UPLOADS_DIR.iterdir(), reverse=True):
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                uploads.append({
                    "filename": f.name,
                    "url": f"/uploads/{f.name}",
                    "size": f.stat().st_size,
                })
    return {"uploads": uploads}


# ============================================================
# VIDEO GENERATION
# ============================================================

@app.post("/api/generate")
async def generate_video(
    model_name: str = Form(...),
    prompt: Optional[str] = Form(None),
    negative_prompt: str = Form(""),
    image_path: Optional[str] = Form(None),
    num_frames: int = Form(14),
    num_inference_steps: int = Form(25),
    fps: int = Form(6),
    seed: int = Form(-1),
    motion_bucket_id: int = Form(127),
    noise_aug_strength: float = Form(0.02),
    cfg_scale: float = Form(3.0),
    width: int = Form(1024),
    height: int = Form(576),
    allow_nsfw: bool = Form(False),
):
    """Start video generation (runs in background thread). Returns video_id immediately."""
    import threading

    # Pre-create video_id so we can return it to the client
    import uuid
    video_id = str(uuid.uuid4())[:8]

    # Run generation in background thread
    def run_generation():
        try:
            result = video_generator.generate_video_with_id(
                video_id=video_id,
                model_name=model_name,
                prompt=prompt,
                image_path=image_path,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                fps=fps,
                seed=seed,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                allow_nsfw=allow_nsfw,
            )
        except Exception as e:
            logger.error(f"Background generation error: {e}")

    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    return {
        "success": True,
        "message": "Generation started",
        "video_id": video_id,
    }


@app.get("/api/generate/{video_id}")
async def get_generation_status(video_id: str):
    """Get the status of a generation task."""
    status = video_generator.get_task_status(video_id)
    if status:
        return status
    raise HTTPException(status_code=404, detail="Generation task not found")


@app.get("/api/outputs")
async def list_outputs():
    """List all generated videos."""
    return {"outputs": video_generator.list_outputs()}


@app.delete("/api/outputs/{video_id}")
async def delete_output(video_id: str):
    """Delete a generated video."""
    if video_generator.delete_output(video_id):
        return {"success": True, "message": f"Output deleted: {video_id}"}
    raise HTTPException(status_code=404, detail="Output not found")


# ============================================================
# WEBSOCKET - Real-time progress
# ============================================================

@app.websocket("/ws/generate/{video_id}")
async def ws_generation_progress(websocket: WebSocket, video_id: str):
    """WebSocket endpoint for real-time generation progress."""
    await websocket.accept()

    try:
        last_progress = -1
        while True:
            status = video_generator.get_task_status(video_id)
            if status:
                current_progress = status.get("progress", 0)
                if current_progress != last_progress:
                    await websocket.send_json(status)
                    last_progress = current_progress

                    # Stop if completed or error
                    if status.get("status") in ("completed", "error"):
                        await websocket.send_json(status)
                        break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {video_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ============================================================
# SYSTEM INFO
# ============================================================

@app.get("/api/system/info")
async def system_info():
    """Get system information."""
    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2)
        info["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        info["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)

    info["current_model"] = model_manager.get_current_model()
    info["allow_nsfw"] = ALLOW_NSFW
    info["models_dir"] = str(MODELS_DIR)
    info["outputs_dir"] = str(OUTPUTS_DIR)

    return info


@app.get("/api/system/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "AI Video Generator"}


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Run on startup."""
    logger.info("=" * 60)
    logger.info("  AI Video Generator Starting...")
    logger.info("=" * 60)
    logger.info(f"  Models dir: {MODELS_DIR}")
    logger.info(f"  Outputs dir: {OUTPUTS_DIR}")
    logger.info(f"  Static dir: {static_dir}")
    logger.info(f"  Allow NSFW: {ALLOW_NSFW}")

    import torch
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2)} GB")
    else:
        logger.warning("  No GPU detected - generation will be slow on CPU")

    # Scan for existing models
    models = model_manager.list_models()
    if models:
        logger.info(f"  Found {len(models)} models:")
        for m in models:
            logger.info(f"    - {m['name']} ({m['type']}) - {m['size_mb']} MB")
    else:
        logger.info("  No models found. Download a model to get started!")

    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
