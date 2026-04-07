"""
Video Generator - Handles video generation using loaded models.
Supports Image-to-Video and Text-to-Video generation.
"""
import os
import time
import uuid
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import torch
import numpy as np
from PIL import Image

from app.config import (
    OUTPUTS_DIR,
    UPLOADS_DIR,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_FPS,
    DEFAULT_SEED,
    DEFAULT_MOTION_BUCKET_ID,
    DEFAULT_NOISE_AUG_STRENGTH,
    DEFAULT_CFG_SCALE,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
)
from app.model_manager import model_manager

logger = logging.getLogger(__name__)


class VideoGenerator:
    """Generate videos using loaded AI models."""

    def __init__(self):
        self._generation_tasks: Dict[str, Dict[str, Any]] = {}

    def generate_video(
        self,
        model_name: str,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        negative_prompt: str = "",
        num_frames: int = DEFAULT_NUM_FRAMES,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        fps: int = DEFAULT_FPS,
        seed: int = DEFAULT_SEED,
        motion_bucket_id: int = DEFAULT_MOTION_BUCKET_ID,
        noise_aug_strength: float = DEFAULT_NOISE_AUG_STRENGTH,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        allow_nsfw: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Generate a video (auto-creates video_id)."""
        video_id = str(uuid.uuid4())[:8]
        return self.generate_video_with_id(
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

    def generate_video_with_id(
        self,
        video_id: str,
        model_name: str,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        negative_prompt: str = "",
        num_frames: int = DEFAULT_NUM_FRAMES,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        fps: int = DEFAULT_FPS,
        seed: int = DEFAULT_SEED,
        motion_bucket_id: int = DEFAULT_MOTION_BUCKET_ID,
        noise_aug_strength: float = DEFAULT_NOISE_AUG_STRENGTH,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        allow_nsfw: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate a video with a pre-assigned video_id.

        Returns:
            Dict with video_id, output_path, and generation info.
        """
        task_info = {
            "video_id": video_id,
            "status": "loading_model",
            "progress": 0,
            "message": "Loading model...",
            "start_time": time.time(),
        }
        self._generation_tasks[video_id] = task_info

        def update_progress(progress: float, message: str, status: str = "generating"):
            task_info["progress"] = progress
            task_info["message"] = message
            task_info["status"] = status
            task_info["elapsed"] = time.time() - task_info["start_time"]
            if progress_callback:
                progress_callback(video_id, progress, message, status)

        try:
            # Step 1: Load model
            update_progress(5, "Loading model...")
            pipeline = model_manager.load_model(model_name, allow_nsfw=allow_nsfw)
            model_info = model_manager.models.get(model_name)

            # Step 2: Validate inputs
            if model_info and model_info.type == "svd" and not image_path:
                raise ValueError("Image-to-Video models require an input image. Switch to a Text-to-Video model like Zeroscope for text prompt generation.")

            if model_info and model_info.type == "text2video" and not prompt:
                raise ValueError("Text-to-Video models require a text prompt. Switch to an Image-to-Video model like SVD for image input.")

            # Step 3: Prepare inputs
            update_progress(10, "Preparing inputs...")
            input_image = None

            if image_path:
                img_path = Path(image_path)
                if not img_path.exists():
                    raise ValueError(f"Image not found: {image_path}")

                input_image = Image.open(img_path).convert("RGB")

                # Resize to target dimensions if needed
                if input_image.size != (width, height):
                    input_image = input_image.resize((width, height), Image.LANCZOS)

            # Step 4: Set seed
            generator = None
            if seed >= 0:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device).manual_seed(seed)

            # Step 5: Generate
            update_progress(15, "Generating video...")

            # Create output directory
            video_dir = OUTPUTS_DIR / video_id
            video_dir.mkdir(parents=True, exist_ok=True)

            result = None

            if model_info and model_info.type == "svd":
                # Image-to-Video generation
                result = self._generate_img2vid(
                    pipeline=pipeline,
                    image=input_image,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    motion_bucket_id=motion_bucket_id,
                    noise_aug_strength=noise_aug_strength,
                    generator=generator,
                    progress_callback=lambda p, m: update_progress(15 + p * 70, m),
                )
            elif model_info and model_info.type == "text2video":
                # Text-to-Video generation
                result = self._generate_text2vid(
                    pipeline=pipeline,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    generator=generator,
                    cfg_scale=cfg_scale,
                    progress_callback=lambda p, m: update_progress(15 + p * 70, m),
                )
            else:
                # Generic pipeline
                result = self._generate_generic(
                    pipeline=pipeline,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    generator=generator,
                    cfg_scale=cfg_scale,
                    progress_callback=lambda p, m: update_progress(15 + p * 70, m),
                )

            # Step 6: Save output
            update_progress(85, "Saving video...")

            # result is a list of PIL Images (frames)
            if result is not None and len(result) > 0:
                output_frames_path = video_dir / "frames"
                output_frames_path.mkdir(exist_ok=True)

                # Save frames
                frame_paths = []
                for i, frame in enumerate(result):
                    frame_path = output_frames_path / f"frame_{i:05d}.png"
                    frame.save(frame_path)
                    frame_paths.append(str(frame_path))

                # Convert frames to video using ffmpeg
                output_video_path = video_dir / "output.mp4"
                self._frames_to_video(
                    frame_paths,
                    output_video_path,
                    fps=fps,
                    width=width,
                    height=height,
                )

                # Also save as GIF (smaller, for preview)
                output_gif_path = video_dir / "output.gif"
                self._frames_to_gif(frame_paths, output_gif_path, fps=fps)

                # Calculate info
                actual_seed = seed if seed >= 0 and generator else -1

                task_info.update({
                    "status": "completed",
                    "progress": 100,
                    "message": "Video generated successfully!",
                    "output_path": str(output_video_path),
                    "gif_path": str(output_gif_path),
                    "video_url": f"/outputs/{video_id}/output.mp4",
                    "gif_url": f"/outputs/{video_id}/output.gif",
                    "frames_url": f"/outputs/{video_id}/frames/",
                    "num_frames": len(result),
                    "fps": fps,
                    "seed": actual_seed,
                    "elapsed": time.time() - task_info["start_time"],
                    "width": width,
                    "height": height,
                    "prompt": prompt,
                })

            else:
                task_info.update({
                    "status": "error",
                    "progress": 100,
                    "message": "No frames generated",
                    "elapsed": time.time() - task_info["start_time"],
                })

        except Exception as e:
            import traceback
            logger.error(f"Generation failed for {video_id}: {e}")
            logger.error(traceback.format_exc())
            task_info.update({
                "status": "error",
                "progress": task_info.get("progress", 0),
                "message": str(e),
                "elapsed": time.time() - task_info["start_time"],
            })

        return self._generation_tasks[video_id]

    def _generate_img2vid(
        self,
        pipeline,
        image: Image.Image,
        num_frames: int,
        num_inference_steps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        generator,
        progress_callback: Callable,
    ):
        """Generate video from image using SVD pipeline."""
        # SVD has a step callback
        def step_callback(step, timestep, latents):
            progress = (step + 1) / num_inference_steps
            progress_callback(progress, f"Step {step + 1}/{num_inference_steps}")
            return latents

        with torch.inference_mode():
            result = pipeline(
                image=image,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                generator=generator,
                callback=step_callback,
                callback_steps=1,
            )

        return result.frames[0]

    def _generate_text2vid(
        self,
        pipeline,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        num_inference_steps: int,
        width: int,
        height: int,
        generator,
        cfg_scale: float,
        progress_callback: Callable,
    ):
        """Generate video from text prompt."""
        def step_callback(step, timestep, latents):
            progress = (step + 1) / num_inference_steps
            progress_callback(progress, f"Step {step + 1}/{num_inference_steps}")
            return latents

        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                generator=generator,
                guidance_scale=cfg_scale,
                callback=step_callback,
                callback_steps=1,
            )

        return result.frames[0]

    def _generate_generic(
        self,
        pipeline,
        prompt: str,
        negative_prompt: str,
        image: Image.Image,
        num_frames: int,
        num_inference_steps: int,
        width: int,
        height: int,
        generator,
        cfg_scale: float,
        progress_callback: Callable,
    ):
        """Generic pipeline generation."""
        kwargs = {
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }

        if prompt:
            kwargs["prompt"] = prompt
            kwargs["negative_prompt"] = negative_prompt or None
            kwargs["guidance_scale"] = cfg_scale

        if image:
            kwargs["image"] = image

        if hasattr(pipeline, "num_frames"):
            kwargs["num_frames"] = num_frames

        if hasattr(pipeline, "width"):
            kwargs["width"] = width

        if hasattr(pipeline, "height"):
            kwargs["height"] = height

        def step_callback(step, timestep, latents):
            progress = (step + 1) / num_inference_steps
            progress_callback(progress, f"Step {step + 1}/{num_inference_steps}")
            return latents

        kwargs["callback"] = step_callback
        kwargs["callback_steps"] = 1

        with torch.inference_mode():
            result = pipeline(**kwargs)

        # Extract frames
        if hasattr(result, "frames"):
            return result.frames[0]
        elif isinstance(result, list) and all(isinstance(f, Image.Image) for f in result):
            return result
        elif isinstance(result, Image.Image):
            return [result]
        else:
            return [result]

    def _frames_to_video(
        self,
        frame_paths: list,
        output_path: Path,
        fps: int = 6,
        width: int = 1024,
        height: int = 576,
    ) -> bool:
        """Convert frames to MP4 video using ffmpeg."""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_paths[0].replace("frame_00000", "frame_%05d"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-preset", "medium",
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                # Fallback: try simpler command
                cmd_simple = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", frame_paths[0].replace("frame_00000", "frame_%05d"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(output_path),
                ]
                subprocess.run(cmd_simple, capture_output=True, text=True, timeout=300)

            return output_path.exists()

        except FileNotFoundError:
            logger.warning("ffmpeg not found, skipping video encoding")
            return False
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timeout")
            return False
        except Exception as e:
            logger.error(f"Video encoding failed: {e}")
            return False

    def _frames_to_gif(
        self,
        frame_paths: list,
        output_path: Path,
        fps: int = 6,
        max_size: tuple = (512, 512),
    ) -> bool:
        """Convert frames to GIF using ffmpeg."""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_paths[0].replace("frame_00000", "frame_%05d"),
                "-vf", f"scale={max_size[0]}:{max_size[1]}:force_original_aspect_ratio=decrease",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return output_path.exists()

        except Exception as e:
            logger.error(f"GIF encoding failed: {e}")
            return False

    def get_task_status(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a generation task."""
        return self._generation_tasks.get(video_id)

    def list_outputs(self) -> list:
        """List all generated videos."""
        outputs = []
        if not OUTPUTS_DIR.exists():
            return outputs

        for video_dir in sorted(OUTPUTS_DIR.iterdir(), reverse=True):
            if video_dir.is_dir():
                video_file = video_dir / "output.mp4"
                gif_file = video_dir / "output.gif"

                if video_file.exists() or gif_file.exists():
                    task_info = self._generation_tasks.get(video_dir.name, {})
                    outputs.append({
                        "video_id": video_dir.name,
                        "video_url": f"/outputs/{video_dir.name}/output.mp4",
                        "gif_url": f"/outputs/{video_dir.name}/output.gif",
                        "status": task_info.get("status", "completed"),
                        "prompt": task_info.get("prompt", ""),
                        "num_frames": task_info.get("num_frames", 0),
                        "fps": task_info.get("fps", 0),
                        "width": task_info.get("width", 0),
                        "height": task_info.get("height", 0),
                        "elapsed": task_info.get("elapsed", 0),
                        "created_at": video_dir.stat().st_ctime,
                    })

        return outputs

    def delete_output(self, video_id: str) -> bool:
        """Delete a generated video."""
        video_dir = OUTPUTS_DIR / video_id
        if video_dir.exists():
            import shutil
            shutil.rmtree(video_dir)
            if video_id in self._generation_tasks:
                del self._generation_tasks[video_id]
            return True
        return False


# Global generator instance
video_generator = VideoGenerator()
