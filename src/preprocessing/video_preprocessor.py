"""
Video preprocessing module for extracting frames from video files.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class VideoPreprocessor:
    """Preprocesses video files by extracting frames."""

    def __init__(
        self, fps: int = 30, max_frames: Optional[int] = None, frame_skip: int = 1
    ):
        """
        Initialize the video preprocessor.

        Args:
            fps: Target frames per second
            max_frames: Maximum number of frames to extract (None for all)
            frame_skip: Process every Nth frame
        """
        self.fps = fps
        self.max_frames = max_frames
        self.frame_skip = frame_skip

    def extract_frames(
        self, video_path: str, output_dir: str, resolution: Tuple[int, int] = (512, 512)
    ) -> List[str]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the input video file
            output_dir: Directory to save extracted frames

        Returns:
            List of paths to extracted frames
        """
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video FPS: {video_fps}, Total frames: {total_frames}")

        frame_paths = []
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames based on frame_skip parameter
            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue

            # Resize frame to the fixed resolution before saving
            frame_resized = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
            frame_name = f"frame_{saved_count:04d}.png"
            frame_path = os.path.join(output_dir, frame_name)

            # Save frame
            cv2.imwrite(frame_path, frame_resized)
            frame_paths.append(frame_path)

            saved_count += 1
            frame_count += 1

            # Check if we've reached max_frames
            if self.max_frames and saved_count >= self.max_frames:
                break

        cap.release()
        print(f"Extracted {saved_count} frames to {output_dir}")

        return frame_paths

    def extract_frames_without_mask(
        self,
        video_path: str,
        mask: np.ndarray,
        output_dir: str,
        resolution: Tuple[int, int] = (512, 512),
    ) -> List[str]:
        """
        Extract frames from a video file without applying any mask.

        Args:
            video_path: Path to the input video file
            output_dir: Directory to save extracted frames
        Returns:
            List of paths to extracted frames
        """
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video FPS: {video_fps}, Total frames: {total_frames}")

        frame_paths = []
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames based on frame_skip parameter
            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue

            # Apply mask to frame
            if mask is not None:
                frame = cv2.bitwise_and(frame, frame, mask=mask[frame_count])

            # Resize frame to the fixed resolution before saving
            frame_resized = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
            frame_name = f"frame_{saved_count:04d}.png"
            frame_path = os.path.join(output_dir, frame_name)

            # Save frame
            cv2.imwrite(frame_path, frame_resized)
            frame_paths.append(frame_path)

            saved_count += 1
            frame_count += 1

            # Check if we've reached max_frames
            if self.max_frames and saved_count >= self.max_frames:
                break

        cap.release()
        print(f"Extracted {saved_count} frames to {output_dir}")

        return frame_paths

    def resize_frames(
        self, frame_paths: List[str], target_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        """
        Resize frames to target size.

        Args:
            frame_paths: List of paths to frame images
            target_size: Target size (width, height)

        Returns:
            List of resized frames as numpy arrays
        """
        resized_frames = []

        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Could not read frame {frame_path}")
                continue

            resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized)

        return resized_frames
