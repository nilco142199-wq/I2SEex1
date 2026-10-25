#!/usr/bin/env python3
"""Exercise 0/1 Entry Point: 3D Volume Slicing and Video Generation Pipeline.

CLI Usage:
    python Ex0.py <file.nrrd> <file.csv> <file.mp4>

Arguments:
    <file.nrrd> Path to the input 3D NRRD voxel volume file (uint8, uint16, uint32).
    <file.csv>  Path to the input CSV containing plane coordinate definitions.
    <file.mp4>  Path to the output MP4 video file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from nrrd_parser import CSVParser, NRRDParser
from projection_engine import ProjectionMethod
from video_generator import VideoGenerator


def print_progress(current: int, total: int, prefix: str = "Rendering frames: ") -> None:
    """Print an interactive ASCII progress bar to stdout."""
    bar_len = 40
    fraction = current / total
    filled_len = int(bar_len * fraction)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)
    percent = fraction * 100.0
    sys.stdout.write(f"\r{prefix}[{bar}] {current}/{total} ({percent:5.1f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def run_pipeline(
    nrrd_path: str,
    csv_path: str,
    output_mp4: str,
    frame_count: int = 240,
    fps: float = 30.0,
    method: str = "vectorized",
) -> int:
    """Execute the end-to-end 3D volume slicing and video generation pipeline.

    Args:
        nrrd_path: Path to the .nrrd 3D volume file.
        csv_path: Path to the .csv plane definitions file.
        output_mp4: Path for the generated .mp4 video file.
        frame_count: Number of frames to generate (default: 240).
        fps: Playback frame rate (default: 30.0).
        method: Projection algorithm ('vectorized' or 'stepping').

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    print("=" * 70)
    print(" 3D Volume Plane Projection & Video Generator (I2SE Exercise 1)")
    print("=" * 70)

    # 1. Validate Input Paths
    nrrd_file = Path(nrrd_path)
    csv_file = Path(csv_path)
    mp4_file = Path(output_mp4)

    # Smart path fallback for NRRD file
    if not nrrd_file.is_file():
        alternates = [
            Path("data") / nrrd_file.name,
            Path("docs") / nrrd_file.name,
            Path(".") / nrrd_file.name,
        ]
        for alt in alternates:
            if alt.is_file():
                print(f"Notice: '{nrrd_file}' not found directly, resolved to '{alt}'")
                nrrd_file = alt
                break

    if not nrrd_file.is_file():
        print(f"Error: NRRD volume file not found: {nrrd_path}", file=sys.stderr)
        return 1

    # Smart path fallback for CSV file
    if not csv_file.is_file():
        alternates = [
            Path("data") / csv_file.name,
            Path("docs") / csv_file.name,
            Path(".") / csv_file.name,
        ]
        for alt in alternates:
            if alt.is_file():
                print(f"Notice: '{csv_file}' not found directly, resolved to '{alt}'")
                csv_file = alt
                break

    if not csv_file.is_file():
        print(f"Error: CSV plane points file not found: {csv_path}", file=sys.stderr)
        return 1

    # 2. Ingest 3D Volume
    print(f"\n[1/4] Ingesting 3D NRRD volume from: {nrrd_file.name}")
    try:
        t0 = time.perf_counter()
        volume, header = NRRDParser.load(nrrd_file)
        load_time = time.perf_counter() - t0
        print(f"      Dimensions : {volume.shape} (sx, sy, sz)")
        print(f"      Data Type  : {volume.dtype} (Range: [{volume.min()}, {volume.max()}])")
        print(f"      Load Time  : {load_time * 1000.0:.2f} ms")
    except Exception as exc:
        print(f"Error loading NRRD file: {exc}", file=sys.stderr)
        return 1

    # 3. Ingest Plane Trajectory Keyframes
    print(f"\n[2/4] Parsing plane coordinates from: {csv_file.name}")
    try:
        keyframes = CSVParser.load_points(csv_file)
        pt_type = "Triangle (3 points)" if keyframes.shape[1] == 3 else "Trapezoid (4 points)"
        print(f"      Loaded     : {len(keyframes)} keyframe definition(s)")
        print(f"      Geometry   : {pt_type}")
    except Exception as exc:
        print(f"Error loading CSV file: {exc}", file=sys.stderr)
        return 1

    # 4. Trajectory Interpolation
    print(f"\n[3/4] Interpolating plane trajectory to {frame_count} frames")
    if len(keyframes) < frame_count:
        print(
            f"      Interpolating smoothly from {len(keyframes)} keyframes -> {frame_count} frames"
        )
    else:
        print(f"      Using {frame_count} frames directly")

    # 5. Render Video Sequence
    print(
        f"\n[4/4] Executing {method} projection & rendering MP4 video ({fps} FPS)"
    )
    try:
        stats = VideoGenerator.render_video(
            volume=volume,
            plane_keyframes=keyframes,
            output_filepath=mp4_file,
            frame_count=frame_count,
            fps=fps,
            method=method,
            progress_callback=print_progress,
        )

        print("\n" + "-" * 70)
        print(" Video Generation Successful!")
        print(f"   Output Destination : {stats['output_path']}")
        print(f"   Total Frames       : {stats['frame_count']} frames ({stats['fps']} FPS)")
        print(f"   Video Duration     : {stats['duration_seconds']:.2f} seconds")
        print(f"   Frame Resolution   : {stats['frame_width']}x{stats['frame_height']} pixels")
        print(f"   Render Time        : {stats['elapsed_seconds']:.3f} seconds")
        print(f"   Average Throughput : {stats['projection_fps']:.1f} FPS")
        print(f"   File Size          : {stats['file_size_bytes'] / 1024:.2f} KB")
        print("-" * 70)
        return 0

    except Exception as exc:
        print(f"\nError during video generation: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    """CLI Entry point accepting positional arguments as specified."""
    # Support positional syntax: python Ex0.py <file.nrrd> <file.csv> <file.mp4>
    # Also support flags --fps, --method, --frames if passed
    parser = argparse.ArgumentParser(
        description="3D Volume Slicing to 2D Planar Image & MP4 Video Generator",
        usage="python Ex0.py <file.nrrd> <file.csv> <file.mp4> [--method {vectorized,stepping}] [--fps FPS]",
    )

    parser.add_argument("nrrd", help="Path to input .nrrd 3D volume file")
    parser.add_argument("csv", help="Path to input .csv plane definition file")
    parser.add_argument("mp4", help="Path to output .mp4 video destination")
    parser.add_argument(
        "--method",
        choices=["vectorized", "stepping"],
        default="vectorized",
        help="Projection engine execution mode (default: vectorized)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video frames per second (default: 30.0)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=240,
        help="Number of video frames to render (default: 240)",
    )

    args = parser.parse_args()

    exit_code = run_pipeline(
        nrrd_path=args.nrrd,
        csv_path=args.csv,
        output_mp4=args.mp4,
        frame_count=args.frames,
        fps=args.fps,
        method=args.method,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
