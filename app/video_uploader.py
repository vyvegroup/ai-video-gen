"""
Video Uploader - Upload completed videos to a separate GitHub repository.
Uses GitHub Contents API to push MP4/GIF files.
"""
import os
import base64
import logging
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime

import requests

from app.state_manager import state_manager

logger = logging.getLogger(__name__)


class VideoUploader:
    """Upload completed videos to a GitHub repository."""

    def __init__(self):
        self._github_token: Optional[str] = None
        self._repo_owner: Optional[str] = None
        self._repo_name: Optional[str] = "ai-video-outputs"

    def configure(self, github_token: str, repo_name: str = "ai-video-outputs"):
        """Configure the uploader with GitHub credentials."""
        self._github_token = github_token
        # Extract owner from token by getting authenticated user
        self._repo_owner = self._get_authenticated_user()
        self._repo_name = repo_name
        if self._repo_owner and self._repo_name:
            self._ensure_repo_exists()

    def _get_authenticated_user(self) -> Optional[str]:
        """Get the authenticated GitHub username."""
        if not self._github_token:
            return None
        try:
            resp = requests.get(
                "https://api.github.com/user",
                headers={"Authorization": f"token {self._github_token}"},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("login")
        except Exception as e:
            logger.error(f"Failed to get GitHub user: {e}")
        return None

    def _ensure_repo_exists(self) -> bool:
        """Ensure the target repository exists (create if not)."""
        if not self._repo_owner or not self._github_token:
            return False

        repo_full = f"{self._repo_owner}/{self._repo_name}"
        headers = {"Authorization": f"token {self._github_token}"}

        # Check if repo exists
        resp = requests.get(
            f"https://api.github.com/repos/{repo_full}",
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info(f"Target repo exists: {repo_full}")
            return True

        # Create repo
        logger.info(f"Creating repo: {repo_full}")
        resp = requests.post(
            "https://api.github.com/user/repos",
            headers=headers,
            json={
                "name": self._repo_name,
                "description": "🎬 AI Generated Videos - Auto-uploaded",
                "private": True,
                "auto_init": False,
            },
            timeout=15,
        )
        if resp.status_code in (200, 201):
            logger.info(f"Repo created: {repo_full}")
            return True
        else:
            logger.error(f"Failed to create repo: {resp.status_code} {resp.text}")
            return False

    def upload_video(self, video_id: str, output_path: str) -> Optional[str]:
        """
        Upload a video to GitHub repository.

        Returns:
            GitHub URL of the uploaded file, or None on failure.
        """
        if not self._github_token or not self._repo_owner:
            logger.warning("Video uploader not configured (missing token/owner)")
            return None

        if not output_path or not Path(output_path).exists():
            logger.error(f"Output file not found: {output_path}")
            return None

        output_file = Path(output_path)
        repo_full = f"{self._repo_owner}/{self._repo_name}"
        headers = {"Authorization": f"token {self._github_token}"}

        # Determine target path in repo
        file_ext = output_file.suffix  # .mp4 or .gif
        date_prefix = datetime.utcnow().strftime("%Y-%m-%d")
        target_path = f"videos/{date_prefix}/{video_id}{file_ext}"

        try:
            # Read file and encode to base64
            with open(output_file, "rb") as f:
                file_content = f.read()
            b64_content = base64.b64encode(file_content).decode("utf-8")

            logger.info(f"Uploading {output_file.name} ({len(file_content) / (1024*1024):.1f} MB) to {repo_full}...")

            # Check if file already exists (to get SHA for update)
            sha = None
            resp = requests.get(
                f"https://api.github.com/repos/{repo_full}/contents/{target_path}",
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                sha = resp.json().get("sha")

            # Upload via Contents API
            body = {
                "message": f"🎬 Add video {video_id} ({file_ext})",
                "content": b64_content,
            }
            if sha:
                body["sha"] = sha

            resp = requests.put(
                f"https://api.github.com/repos/{repo_full}/contents/{target_path}",
                headers=headers,
                json=body,
                timeout=120,  # Large files need more time
            )

            if resp.status_code in (200, 201):
                data = resp.json()
                download_url = data.get("content", {}).get("download_url", "")
                html_url = data.get("content", {}).get("html_url", "")
                logger.info(f"Uploaded: {html_url}")
                return html_url or download_url
            else:
                logger.error(f"Upload failed: {resp.status_code} {resp.text}")
                return None

        except Exception as e:
            logger.error(f"Upload error: {e}")
            return None

    def upload_video_async(self, video_id: str, output_path: str, gif_path: str = None):
        """Upload video in background thread."""
        def _upload():
            try:
                state_manager.mark_uploading(video_id)

                # Upload MP4
                mp4_url = self.upload_video(video_id, output_path)
                uploaded_url = mp4_url

                # Upload GIF if available
                if gif_path and Path(gif_path).exists():
                    gif_url = self.upload_video(video_id, gif_path)
                    if not uploaded_url:
                        uploaded_url = gif_url

                if uploaded_url:
                    state_manager.mark_uploaded(video_id, uploaded_url)
                    logger.info(f"Video {video_id} uploaded to GitHub: {uploaded_url}")
                else:
                    state_manager.update_generation(video_id, message="Upload failed (check logs)")

            except Exception as e:
                logger.error(f"Async upload failed for {video_id}: {e}")
                state_manager.update_generation(video_id, message=f"Upload failed: {e}")

        thread = threading.Thread(target=_upload, daemon=True)
        thread.start()


# Global instance
video_uploader = VideoUploader()
