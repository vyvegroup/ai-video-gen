"""
Persistent State Manager - Save/restore generation state to disk.
Survives page refresh, server restart, and connection drops.
"""
import json
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

from app.config import BASE_DIR

logger = logging.getLogger(__name__)

STATE_DIR = BASE_DIR / "state"
STATE_FILE = STATE_DIR / "generations.json"
CHAT_DIR = BASE_DIR / "state" / "chats"

for d in [STATE_DIR, CHAT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()


class StateManager:
    """Persistent state for video generations and chat sessions."""

    def __init__(self):
        self._generations: Dict[str, Dict[str, Any]] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load generations state from disk."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    self._generations = json.load(f)
                logger.info(f"Loaded {len(self._generations)} generation states from disk")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                self._generations = {}

    def _save_state(self) -> None:
        """Save generations state to disk (thread-safe)."""
        with _lock:
            try:
                with open(STATE_FILE, "w") as f:
                    json.dump(self._generations, f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")

    def create_generation(self, video_id: str, model_name: str, **kwargs) -> Dict[str, Any]:
        """Create a new generation entry."""
        entry = {
            "video_id": video_id,
            "model_name": model_name,
            "status": "queued",
            "progress": 0,
            "message": "Queued...",
            "created_at": time.time(),
            "updated_at": time.time(),
            "prompt": kwargs.get("prompt", ""),
            "image_path": kwargs.get("image_path", ""),
            "num_frames": kwargs.get("num_frames", 0),
            "fps": kwargs.get("fps", 0),
            "width": kwargs.get("width", 0),
            "height": kwargs.get("height", 0),
            "seed": kwargs.get("seed", -1),
            "allow_nsfw": kwargs.get("allow_nsfw", False),
            # Output
            "video_url": None,
            "gif_url": None,
            "output_path": None,
            "elapsed": 0,
            "error": None,
            "uploaded_to_github": False,
            "github_url": None,
        }
        self._generations[video_id] = entry
        self._save_state()
        return entry

    def update_generation(self, video_id: str, **updates) -> None:
        """Update a generation entry."""
        if video_id not in self._generations:
            return
        self._generations[video_id].update(updates)
        self._generations[video_id]["updated_at"] = time.time()
        self._save_state()

    def get_generation(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get a single generation entry."""
        return self._generations.get(video_id)

    def list_generations(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all generations, optionally filtered by status."""
        gens = list(self._generations.values())
        if status:
            gens = [g for g in gens if g.get("status") == status]
        # Sort by creation time (newest first)
        gens.sort(key=lambda g: g.get("created_at", 0), reverse=True)
        return gens

    def get_active_generations(self) -> List[Dict[str, Any]]:
        """Get all in-progress generations."""
        return self.list_generations(status="generating")

    def get_completed_generations(self) -> List[Dict[str, Any]]:
        """Get all completed generations."""
        return [g for g in self.list_generations()
                if g.get("status") in ("completed", "error")]

    def mark_completed(self, video_id: str, video_url: str, gif_url: str,
                       output_path: str, elapsed: float, **kwargs) -> None:
        """Mark a generation as completed."""
        self.update_generation(
            video_id,
            status="completed",
            progress=100,
            message="Video generated successfully!",
            video_url=video_url,
            gif_url=gif_url,
            output_path=output_path,
            elapsed=elapsed,
            **kwargs,
        )

    def mark_error(self, video_id: str, error: str) -> None:
        """Mark a generation as failed."""
        self.update_generation(
            video_id,
            status="error",
            message=f"Error: {error}",
            error=error,
        )

    def mark_uploading(self, video_id: str) -> None:
        """Mark video as being uploaded to GitHub."""
        self.update_generation(video_id, status="uploading", message="Uploading to GitHub...")

    def mark_uploaded(self, video_id: str, github_url: str) -> None:
        """Mark video as uploaded to GitHub."""
        self.update_generation(
            video_id,
            uploaded_to_github=True,
            github_url=github_url,
            message="Uploaded to GitHub!",
        )

    # ---- Chat persistence ----

    def save_chat(self, session_id: str, messages: List[Dict]) -> None:
        """Save chat messages to disk."""
        chat_file = CHAT_DIR / f"{session_id}.json"
        try:
            with open(chat_file, "w") as f:
                json.dump({
                    "session_id": session_id,
                    "messages": messages,
                    "updated_at": time.time(),
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save chat: {e}")

    def load_chat(self, session_id: str) -> List[Dict]:
        """Load chat messages from disk."""
        chat_file = CHAT_DIR / f"{session_id}.json"
        if chat_file.exists():
            try:
                with open(chat_file, "r") as f:
                    data = json.load(f)
                return data.get("messages", [])
            except Exception:
                return []
        return []

    def list_chats(self) -> List[Dict]:
        """List all chat sessions."""
        chats = []
        for f in CHAT_DIR.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                chats.append({
                    "session_id": data.get("session_id"),
                    "message_count": len(data.get("messages", [])),
                    "updated_at": data.get("updated_at", 0),
                })
            except Exception:
                pass
        chats.sort(key=lambda c: c.get("updated_at", 0), reverse=True)
        return chats

    def delete_chat(self, session_id: str) -> bool:
        """Delete a chat session."""
        chat_file = CHAT_DIR / f"{session_id}.json"
        if chat_file.exists():
            chat_file.unlink()
            return True
        return False


# Global instance
state_manager = StateManager()
