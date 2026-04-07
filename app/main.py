"""
AI Video Generation App - FastAPI Backend
Serves the web UI and handles model management + video generation + AI chat.
"""
import asyncio
import json
import logging
import os
import shutil
import time
import threading
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
from app.state_manager import state_manager
from app.video_uploader import video_uploader
from app.chat_manager import chat_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Video Generator",
    description="Self-hosted AI Video Generation with local model inference + AI Chat",
    version="2.0.0",
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
# PERSISTENT STATE / GENERATIONS
# ============================================================

@app.get("/api/generations")
async def list_all_generations():
    """List ALL generations (persistent across refresh)."""
    return {
        "generations": state_manager.list_generations(),
        "active": state_manager.get_active_generations(),
    }


@app.get("/api/generations/{video_id}")
async def get_generation_persistent(video_id: str):
    """Get generation state (persists across refresh)."""
    gen = state_manager.get_generation(video_id)
    if gen:
        # Also check live status from generator
        live = video_generator.get_task_status(video_id)
        if live and live.get("status") == "generating":
            gen["progress"] = live.get("progress", gen.get("progress", 0))
            gen["message"] = live.get("message", gen.get("message", ""))
            gen["status"] = live.get("status", gen.get("status"))
        return gen
    raise HTTPException(status_code=404, detail="Generation not found")


# ============================================================
# MODEL MANAGEMENT
# ============================================================

@app.get("/api/models")
async def list_models():
    return {"models": model_manager.list_models(), "current": model_manager.get_current_model()}


@app.post("/api/models/download")
async def download_model(
    model_id: str = Form(...),
    source: str = Form("huggingface"),
    hf_token: Optional[str] = Form(None),
):
    try:
        info = model_manager.download_model(model_id=model_id, source=source, hf_token=hf_token)
        return {"success": True, "message": f"Model downloaded: {info.name}",
                "model": {"name": info.name, "type": info.type, "source": info.source, "size_mb": info.size_mb}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/models/{model_name}")
async def delete_model(model_name: str):
    if model_manager.delete_model(model_name):
        return {"success": True, "message": f"Model deleted: {model_name}"}
    raise HTTPException(status_code=404, detail="Model not found")


@app.post("/api/models/load")
async def load_model(model_name: str = Form(...), allow_nsfw: bool = Form(False)):
    try:
        model_manager.load_model(model_name, allow_nsfw=allow_nsfw)
        return {"success": True, "message": f"Model loaded: {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/unload")
async def unload_model():
    model_manager.unload_model()
    return {"success": True, "message": "Model unloaded"}


@app.get("/api/models/registry")
async def get_model_registry():
    return {
        "suggested": [
            {"id": "cerspense/zeroscope_v2_576w", "type": "Text to Video", "description": "Zeroscope V2 - CPU friendly"},
            {"id": "stabilityai/stable-video-diffusion-img2vid-xt", "type": "Image to Video", "description": "SVD XT"},
            {"id": "stabilityai/stable-video-diffusion-img2vid", "type": "Image to Video", "description": "SVD"},
            {"id": "cerspense/zeroscope_v2_XL", "type": "Text to Video", "description": "Zeroscope XL"},
        ]
    }


# ============================================================
# IMAGE UPLOAD
# ============================================================

@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_SIZE // (1024*1024)}MB")

    allowed_types = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

    import uuid
    filename = f"{str(uuid.uuid4())[:8]}_{file.filename}"
    file_path = UPLOADS_DIR / filename
    with open(file_path, "wb") as f:
        f.write(contents)

    return {"success": True, "filename": filename, "path": str(file_path), "url": f"/uploads/{filename}", "size": len(contents)}


@app.get("/api/uploads")
async def list_uploads():
    uploads = []
    if UPLOADS_DIR.exists():
        for f in sorted(UPLOADS_DIR.iterdir(), reverse=True):
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                uploads.append({"filename": f.name, "url": f"/uploads/{f.name}", "size": f.stat().st_size})
    return {"uploads": uploads}


# ============================================================
# VIDEO GENERATION (with persistent state + auto-upload)
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
    """Start video generation. Returns video_id immediately. Persists across refresh."""
    import uuid
    video_id = str(uuid.uuid4())[:8]

    # Save persistent state immediately
    state_manager.create_generation(
        video_id=video_id,
        model_name=model_name,
        prompt=prompt or "",
        image_path=image_path or "",
        num_frames=num_frames,
        fps=fps,
        width=width,
        height=height,
        seed=seed,
        allow_nsfw=allow_nsfw,
    )

    # Progress callback that updates persistent state
    def progress_callback(vid, progress, message, status):
        state_manager.update_generation(vid, progress=progress, message=message, status=status)

    def run_generation():
        try:
            state_manager.update_generation(video_id, status="generating", message="Loading model...")

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
                progress_callback=progress_callback,
            )

            # Mark completed in persistent state
            if result.get("status") == "completed":
                state_manager.mark_completed(
                    video_id=video_id,
                    video_url=result.get("video_url", ""),
                    gif_url=result.get("gif_url", ""),
                    output_path=result.get("output_path", ""),
                    elapsed=result.get("elapsed", 0),
                    num_frames=result.get("num_frames", num_frames),
                )

                # Auto-upload to GitHub repo (async)
                output_path = result.get("output_path")
                gif_path = result.get("gif_path")
                if output_path and os.path.exists(output_path):
                    video_uploader.upload_video_async(video_id, output_path, gif_path)
            else:
                state_manager.mark_error(video_id, result.get("message", "Unknown error"))

        except Exception as e:
            logger.error(f"Generation error for {video_id}: {e}")
            state_manager.mark_error(video_id, str(e))

    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    return {"success": True, "video_id": video_id, "message": "Generation started"}


@app.get("/api/generate/{video_id}")
async def get_generation_status(video_id: str):
    """Get generation status (checks both persistent state and live generator)."""
    # Check persistent state first
    gen = state_manager.get_generation(video_id)
    live = video_generator.get_task_status(video_id)

    if live:
        # Merge live data into persistent state
        state_manager.update_generation(
            video_id,
            progress=live.get("progress", 0),
            message=live.get("message", ""),
            status=live.get("status", "generating"),
        )
        return live
    elif gen:
        return gen
    raise HTTPException(status_code=404, detail="Generation task not found")


@app.get("/api/outputs")
async def list_outputs():
    return {"outputs": state_manager.get_completed_generations()}


@app.delete("/api/outputs/{video_id}")
async def delete_output(video_id: str):
    if video_generator.delete_output(video_id):
        return {"success": True, "message": f"Output deleted: {video_id}"}
    raise HTTPException(status_code=404, detail="Output not found")


# ============================================================
# AI CHAT
# ============================================================

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class CharacterSettings(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    personality: Optional[str] = None
    appearance: Optional[str] = None
    greeting: Optional[str] = None
    scenario: Optional[str] = None
    nsfw_enabled: Optional[bool] = None


@app.post("/api/chat/send")
async def chat_send(msg: ChatMessage):
    """Send a message to the AI chat."""
    try:
        result = await chat_manager.send_message(
            session_id=msg.session_id,
            user_message=msg.message,
        )
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/session")
async def chat_create_session(settings: Optional[CharacterSettings] = None):
    """Create or resume a chat session."""
    char = None
    if settings:
        char = settings.model_dump(exclude_none=True)
    session = await chat_manager.create_session(character=char)
    return session


@app.get("/api/chat/session/{session_id}")
async def chat_get_session(session_id: str):
    """Get chat session with messages."""
    session = chat_manager.get_session(session_id)
    if session:
        return session
    # Try loading from disk
    session = await chat_manager.create_session(session_id=session_id)
    return session


@app.put("/api/chat/session/{session_id}/character")
async def chat_update_character(session_id: str, settings: CharacterSettings):
    """Update character settings."""
    ok = chat_manager.update_character(session_id, settings.model_dump(exclude_none=True))
    if ok:
        return {"success": True, "message": "Character updated"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/chat/sessions")
async def chat_list_sessions():
    """List all chat sessions."""
    return {"sessions": chat_manager.list_sessions()}


@app.delete("/api/chat/session/{session_id}")
async def chat_delete_session(session_id: str):
    if chat_manager.delete_session(session_id):
        return {"success": True, "message": "Chat deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


# ============================================================
# WEBSOCKET - Real-time progress
# ============================================================

@app.websocket("/ws/generate/{video_id}")
async def ws_generation_progress(websocket: WebSocket, video_id: str):
    await websocket.accept()
    try:
        # First, send current persistent state
        gen = state_manager.get_generation(video_id)
        if gen:
            await websocket.send_json(gen)

        last_progress = -1
        while True:
            # Check live status
            live = video_generator.get_task_status(video_id)
            if live:
                current_progress = live.get("progress", -1)
                if current_progress != last_progress:
                    await websocket.send_json(live)
                    last_progress = current_progress
                    if live.get("status") in ("completed", "error"):
                        break
            elif gen and gen.get("status") in ("completed", "error"):
                await websocket.send_json(gen)
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
    import torch
    info = {"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu"}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2)
    info["current_model"] = model_manager.get_current_model()
    info["allow_nsfw"] = ALLOW_NSFW
    info["active_generations"] = len(state_manager.get_active_generations())
    info["total_generations"] = len(state_manager.list_generations())
    return info


@app.get("/api/system/health")
async def health_check():
    return {"status": "ok", "service": "AI Video Generator v2.0"}


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("  🎬 AI Video Generator v2.0 Starting...")
    logger.info("=" * 60)

    import torch
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("  CPU-only mode (no GPU)")

    # Restore active generations
    active = state_manager.get_active_generations()
    if active:
        logger.info(f"  Restoring {len(active)} active generations from state")
        for gen in active:
            logger.info(f"    - {gen['video_id']} ({gen['status']})")

    completed = state_manager.get_completed_generations()
    logger.info(f"  Found {len(completed)} completed generations in history")

    # Configure video uploader if GitHub token is available
    gh_token = os.getenv("GITHUB_TOKEN", "")
    if gh_token:
        video_uploader.configure(github_token)
        logger.info("  Video auto-upload to GitHub: configured")
    else:
        logger.info("  Video auto-upload: not configured (set GITHUB_TOKEN env)")

    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
