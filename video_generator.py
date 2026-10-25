"""Video Generator for 3D Volume Slicing Sequences.

This module renders 240-frame MP4 videos from projected 2D plane sequences with:
- Linear trajectory interpolation for arbitrary keyframe counts.
- Resolution standardization and intensity normalization (uint8, uint16, uint32).
- Video encoding via OpenCV VideoWriter.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from projection_engine import ProjectionEngine, ProjectionMethod


class VideoGenerator:
    """Generates MP4 videos from 3D volume projections and plane trajectories."""

    DEFAULT_FRAME_COUNT = 240
    DEFAULT_FPS = 30.0

    @classmethod
    def interpolate_planes(
        cls,
        keyframes: np.ndarray,
        target_count: int = DEFAULT_FRAME_COUNT,
    ) -> np.ndarray:
        """Linearly interpolate plane coordinates across time to yield exactly target_count frames.

        Args:
            keyframes: Array of keyframe plane points of shape (K, P, 3), where P is 3 or 4.
            target_count: Total number of frames required (default: 240).

        Returns:
            np.ndarray of shape (target_count, P, 3) containing interpolated coordinates.

        Raises:
            ValueError: If keyframes array has invalid shape.
        """
        arr = np.asarray(keyframes, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[1] not in (3, 4):
            raise ValueError(
                f"Expected keyframes of shape (K, 3, 3) or (K, 4, 3), got {arr.shape}"
            )

        k, p_count, _ = arr.shape

        if k == target_count:
            return arr.copy()

        if k == 1:
            # Replicate single plane definition for all frames
            return np.repeat(arr, target_count, axis=0)

        # Multi-keyframe linear interpolation
        t_keys = np.linspace(0.0, 1.0, k)
        t_target = np.linspace(0.0, 1.0, target_count)

        interpolated = np.zeros((target_count, p_count, 3), dtype=np.float64)

        for p_idx in range(p_count):
            for coord_idx in range(3):
                interpolated[:, p_idx, coord_idx] = np.interp(
                    t_target,
                    t_keys,
                    arr[:, p_idx, coord_idx],
                )

        return interpolated

    @classmethod
    def calculate_canvas_size(
        cls,
        planes: np.ndarray,
        min_dim: int = 64,
    ) -> Tuple[int, int]:
        """Compute the uniform (height, width) bounding canvas required across all frames.

        Args:
            planes: Array of shape (N, P, 3) representing plane definitions.
            min_dim: Minimum dimension for height or width.

        Returns:
            Tuple of (canvas_height, canvas_width).
        """
        max_h = min_dim
        max_w = min_dim

        for plane in planes:
            p_count = plane.shape[0]
            if p_count == 3:
                a, b, c = plane[0], plane[1], plane[2]
                d_ab = float(np.linalg.norm(b - a))
                d_bc = float(np.linalg.norm(c - b))
                max_h = max(max_h, int(np.ceil(d_ab)))
                max_w = max(max_w, int(np.ceil(d_bc)))
            else:
                a1, a2, b, c = plane[0], plane[1], plane[2], plane[3]
                d_ab = float(np.linalg.norm(b - a1))
                d_bc = float(np.linalg.norm(c - b))
                d_a1a2 = float(np.linalg.norm(a2 - a1))
                max_h = max(max_h, int(np.ceil(d_ab)))
                max_w = max(max_w, int(np.ceil(max(d_bc, d_a1a2))))

        # Ensure dimensions are even numbers (requirement for certain MP4 video codecs)
        if max_h % 2 != 0:
            max_h += 1
        if max_w % 2 != 0:
            max_w += 1

        return max_h, max_w

    @classmethod
    def normalize_frame_to_uint8(
        cls,
        frame: np.ndarray,
        global_max: Optional[float] = None,
    ) -> np.ndarray:
        """Normalize a 2D intensity frame to an 8-bit unsigned integer (uint8) image.

        Args:
            frame: 2D numpy array of intensities (uint8, uint16, uint32, or float).
            global_max: Optional maximum scalar for consistent intensity scaling.

        Returns:
            2D numpy array with dtype np.uint8 in range [0, 255].
        """
        if frame.dtype == np.uint8:
            return frame

        if global_max is None or global_max <= 0:
            f_max = float(frame.max())
        else:
            f_max = float(global_max)

        if f_max <= 0:
            return np.zeros(frame.shape, dtype=np.uint8)

        # Scale intensity to [0, 255]
        norm = (frame.astype(np.float32) / f_max) * 255.0
        return np.clip(norm, 0.0, 255.0).astype(np.uint8)

    @classmethod
    def render_video(
        cls,
        volume: np.ndarray,
        plane_keyframes: np.ndarray,
        output_filepath: Union[str, Path],
        frame_count: int = DEFAULT_FRAME_COUNT,
        fps: float = DEFAULT_FPS,
        method: Union[str, ProjectionMethod] = ProjectionMethod.VECTORIZED,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Render a sequence of planar cuts through volume into an MP4 video file.

        Args:
            volume: 3D numpy ndarray representing the voxel matrix M (sx, sy, sz).
            plane_keyframes: Keyframe plane points of shape (K, 3, 3) or (K, 4, 3).
            output_filepath: Output destination path for the .mp4 video file.
            frame_count: Target number of frames to generate (default: 240).
            fps: Frame rate for video playback (default: 30.0 FPS).
            method: 'vectorized' (default) or 'stepping'.
            progress_callback: Optional callback invoked with (current_frame, total_frames).

        Returns:
            Dictionary containing rendering statistics and metadata.

        Raises:
            RuntimeError: If VideoWriter fails to open or output cannot be written.
        """
        out_path = Path(output_filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Trajectory interpolation to exactly frame_count frames
        planes = cls.interpolate_planes(plane_keyframes, target_count=frame_count)

        # 2. Compute uniform canvas size across all frames
        canvas_h, canvas_w = cls.calculate_canvas_size(planes)

        # 3. Initialize OpenCV VideoWriter
        # OpenCV expects (width, height)
        frame_size = (canvas_w, canvas_h)
        fourcc_code = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc_code, fps, frame_size, isColor=True)

        if not writer.isOpened():
            # Try fallback codecs: avc1 or XVID
            for alt_fourcc in ("avc1", "XVID", "MJPG"):
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*alt_fourcc),
                    fps,
                    frame_size,
                    isColor=True,
                )
                if writer.isOpened():
                    break

        if not writer.isOpened():
            raise RuntimeError(
                f"Failed to open OpenCV VideoWriter for destination: {out_path.resolve()}"
            )

        # Calculate volume dynamic range for uniform scaling
        # For non-uint8 volumes (e.g. 16-bit medical scans), use the 99.8th percentile
        # to prevent single high-intensity outlier voxels from dimming the video.
        if volume.size > 0:
            if volume.dtype != np.uint8 and np.any(volume > 0):
                nonzero_voxels = volume[volume > 0]
                vol_max = float(np.percentile(nonzero_voxels, 99.8))
            else:
                vol_max = float(volume.max())
        else:
            vol_max = 255.0

        t_start = time.perf_counter()

        try:
            for idx in range(frame_count):
                plane_pts = planes[idx]

                # Project 2D plane from 3D volume
                raw_frame = ProjectionEngine.project(
                    volume=volume,
                    points=plane_pts,
                    method=method,
                    target_size=(canvas_h, canvas_w),
                )

                # Normalize to 8-bit unsigned integer
                uint8_frame = cls.normalize_frame_to_uint8(raw_frame, global_max=vol_max)

                # Convert single-channel grayscale to 3-channel BGR for VideoWriter
                bgr_frame = cv2.cvtColor(uint8_frame, cv2.COLOR_GRAY2BGR)

                writer.write(bgr_frame)

                if progress_callback is not None:
                    progress_callback(idx + 1, frame_count)

        finally:
            writer.release()

        elapsed_time = time.perf_counter() - t_start
        actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
        file_size = out_path.stat().st_size if out_path.exists() else 0

        return {
            "output_path": str(out_path.resolve()),
            "frame_count": frame_count,
            "fps": fps,
            "duration_seconds": frame_count / fps,
            "frame_width": canvas_w,
            "frame_height": canvas_h,
            "elapsed_seconds": elapsed_time,
            "projection_fps": actual_fps,
            "file_size_bytes": file_size,
        }
