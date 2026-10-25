"""Unit tests and performance benchmarks for 3D Projection Engine and Video Generator."""

from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import pytest

from projection_engine import PlaneGeometry, ProjectionEngine, ProjectionMethod
from video_generator import VideoGenerator


class TestProjectionEngine:
    """Test suite for ProjectionEngine (vectorized and stepping implementations)."""

    @pytest.fixture
    def test_volume(self) -> np.ndarray:
        """Create a known synthetic 3D volume with gradient values."""
        vol = np.zeros((50, 50, 50), dtype=np.uint8)
        # Create a sphere of radius 15 at center (25, 25, 25)
        x = np.arange(50)[:, None, None]
        y = np.arange(50)[None, :, None]
        z = np.arange(50)[None, None, :]
        dist_sq = (x - 25) ** 2 + (y - 25) ** 2 + (z - 25) ** 2
        vol[dist_sq <= 15**2] = 200
        vol[dist_sq <= 8**2] = 255
        return vol

    def test_triangle_plane_geometry(self) -> None:
        """Verify PlaneGeometry calculations for an isosceles triangle."""
        # a = apex (10, 20, 30), b = (40, 20, 30), c = (40, 60, 30)
        pts = np.array([
            [10.0, 20.0, 30.0],
            [40.0, 20.0, 30.0],
            [40.0, 60.0, 30.0],
        ])
        geom = PlaneGeometry.from_points(pts)
        assert geom.is_triangle is True
        assert pytest.approx(geom.d_ab, 1e-4) == 30.0
        assert pytest.approx(geom.d_bc, 1e-4) == 40.0
        assert np.allclose(geom.v_ab, [1.0, 0.0, 0.0])

    def test_quad_plane_geometry(self) -> None:
        """Verify PlaneGeometry calculations for a trapezoid/quad."""
        pts = np.array([
            [10.0, 20.0, 0.0],
            [10.0, 40.0, 0.0],
            [50.0, 10.0, 0.0],
            [50.0, 50.0, 0.0],
        ])
        geom = PlaneGeometry.from_points(pts)
        assert geom.is_triangle is False
        assert pytest.approx(geom.d_a1a2, 1e-4) == 20.0
        assert pytest.approx(geom.d_bc, 1e-4) == 40.0

    def test_algorithmic_equivalence(self, test_volume: np.ndarray) -> None:
        """Assert exact algorithmic equivalence between stepping and vectorized modes."""
        pts = np.array([
            [10.0, 10.0, 10.0],
            [40.0, 10.0, 40.0],
            [40.0, 40.0, 40.0],
        ])
        e_step = ProjectionEngine.project(test_volume, pts, method=ProjectionMethod.STEPPING)
        e_vec = ProjectionEngine.project(test_volume, pts, method=ProjectionMethod.VECTORIZED)

        assert e_step.shape == e_vec.shape
        diff = np.max(np.abs(e_step.astype(int) - e_vec.astype(int)))
        assert diff == 0, f"Max difference between stepping and vectorized is {diff}"

    def test_quad_algorithmic_equivalence(self, test_volume: np.ndarray) -> None:
        """Assert equivalence for 4-point trapezoid cuts."""
        pts = np.array([
            [15.0, 15.0, 10.0],
            [15.0, 35.0, 10.0],
            [45.0, 10.0, 40.0],
            [45.0, 40.0, 40.0],
        ])
        e_step = ProjectionEngine.project(test_volume, pts, method="stepping")
        e_vec = ProjectionEngine.project(test_volume, pts, method="vectorized")

        assert e_step.shape == e_vec.shape
        assert np.array_equal(e_step, e_vec)

    def test_out_of_bounds_handling(self, test_volume: np.ndarray) -> None:
        """Verify that plane coordinates outside the matrix bounds evaluate to v = 0."""
        # Plane placed entirely outside the [0, 50) volume
        pts_outside = np.array([
            [100.0, 100.0, 100.0],
            [150.0, 100.0, 100.0],
            [150.0, 150.0, 100.0],
        ])
        e_out = ProjectionEngine.project(test_volume, pts_outside, method="vectorized")
        assert np.all(e_out == 0), "All out-of-bounds voxels must be 0"

        # Plane partially intersecting the volume
        pts_partial = np.array([
            [-10.0, 25.0, 25.0],
            [60.0, 10.0, 25.0],
            [60.0, 40.0, 25.0],
        ])
        e_partial = ProjectionEngine.project(test_volume, pts_partial, method="vectorized")
        # Check that center contains sphere values while margins contain 0
        assert np.any(e_partial > 0), "Intersecting portion must sample non-zero voxels"
        assert np.any(e_partial == 0), "Portion outside must evaluate to 0"

    def test_target_size_padding(self, test_volume: np.ndarray) -> None:
        """Verify that passing target_size produces exact canvas dimensions."""
        pts = np.array([
            [10.0, 10.0, 10.0],
            [30.0, 10.0, 10.0],
            [30.0, 30.0, 10.0],
        ])
        target_size = (100, 120)
        e_padded = ProjectionEngine.project(
            test_volume, pts, method="vectorized", target_size=target_size
        )
        assert e_padded.shape == target_size

    def test_500cubed_volume_performance_benchmark(self) -> None:
        """Benchmark projection speed on a 500x500x500 voxel volume.

        Requirement: Minimum 30+ FPS, target 100+ FPS.
        """
        # Create 500^3 volume
        sx, sy, sz = 500, 500, 500
        vol_500 = np.random.randint(0, 255, size=(sx, sy, sz), dtype=np.uint8)

        # Realistic diagonal slicing plane (approx 300x300 pixels)
        pts = np.array([
            [50.0, 50.0, 50.0],
            [350.0, 50.0, 350.0],
            [350.0, 350.0, 350.0],
        ])

        # Warmup
        _ = ProjectionEngine.project(vol_500, pts, method="vectorized")

        # Benchmark 50 iterations
        n_iters = 50
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = ProjectionEngine.project(vol_500, pts, method="vectorized")
        elapsed = time.perf_counter() - t0

        fps = n_iters / elapsed
        time_per_frame_ms = (elapsed / n_iters) * 1000.0

        print(f"\n[BENCHMARK] 500^3 volume: {fps:.1f} FPS ({time_per_frame_ms:.2f} ms/frame)")
        assert fps >= 30.0, f"Failed minimum 30 FPS requirement: got {fps:.1f} FPS"
        assert fps >= 100.0, f"Failed 100+ FPS target requirement: got {fps:.1f} FPS"


class TestVideoGenerator:
    """Test suite for trajectory interpolation and video synthesis."""

    def test_linear_interpolation_fewer_than_240(self) -> None:
        """Test interpolating 3 keyframes to exactly 240 frames."""
        kfs = np.array([
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]],
            [[0.0, 0.0, 50.0], [10.0, 0.0, 50.0], [10.0, 10.0, 50.0]],
            [[0.0, 0.0, 100.0], [10.0, 0.0, 100.0], [10.0, 10.0, 100.0]],
        ])
        interpolated = VideoGenerator.interpolate_planes(kfs, target_count=240)

        assert interpolated.shape == (240, 3, 3)
        # First frame matches first keyframe
        np.testing.assert_allclose(interpolated[0], kfs[0], atol=1e-5)
        # Last frame matches last keyframe
        np.testing.assert_allclose(interpolated[-1], kfs[-1], atol=1e-5)
        # Midpoint frame matches middle keyframe
        np.testing.assert_allclose(interpolated[119], kfs[1], atol=1.0)

    def test_single_plane_interpolation(self) -> None:
        """Test that 1 keyframe replicates 240 times."""
        kfs = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
        interpolated = VideoGenerator.interpolate_planes(kfs, target_count=240)
        assert interpolated.shape == (240, 3, 3)
        for i in range(240):
            np.testing.assert_allclose(interpolated[i], kfs[0])

    def test_end_to_end_mp4_generation(self) -> None:
        """Verify rendering a 240-frame MP4 video from volume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_video = Path(tmpdir) / "test_output.mp4"
            vol = np.zeros((40, 40, 40), dtype=np.uint8)
            vol[10:30, 10:30, 10:30] = 180

            kfs = np.array([
                [[5.0, 5.0, 5.0], [35.0, 5.0, 5.0], [35.0, 35.0, 5.0]],
                [[5.0, 5.0, 35.0], [35.0, 5.0, 35.0], [35.0, 35.0, 35.0]],
            ])

            stats = VideoGenerator.render_video(
                volume=vol,
                plane_keyframes=kfs,
                output_filepath=out_video,
                frame_count=240,
                fps=30.0,
            )

            assert stats["frame_count"] == 240
            assert stats["duration_seconds"] == 8.0
            assert out_video.exists()
            assert out_video.stat().st_size > 0

            # Verify using cv2.VideoCapture that file contains 240 frames
            cap = cv2.VideoCapture(str(out_video))
            assert cap.isOpened(), "Generated video must be decodable by OpenCV"
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            assert total_frames == 240
