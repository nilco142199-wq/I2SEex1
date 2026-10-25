"""Unit tests for NRRD and CSV parsers."""

from pathlib import Path
import tempfile

import nrrd
import numpy as np
import pytest

from nrrd_parser import CSVParser, NRRDParser


class TestNRRDParser:
    """Test suite for NRRD volume loading, validation, and generation."""

    def test_load_valid_sample(self) -> None:
        """Test loading the project sample.nrrd volume."""
        sample_path = Path("data/sample.nrrd")
        assert sample_path.exists(), "data/sample.nrrd must exist"

        volume, header = NRRDParser.load(sample_path)
        assert isinstance(volume, np.ndarray)
        assert volume.ndim == 3
        assert volume.shape == (128, 128, 128)
        assert volume.dtype == np.uint8
        assert "dimension" in header or "sizes" in header or "type" in header

    def test_load_mrhead_scan(self) -> None:
        """Test loading MRHead_1.nrrd medical volume if present."""
        mrhead_path = Path("data/MRHead_1.nrrd")
        if not mrhead_path.exists():
            mrhead_path = Path("docs/MRHead_1.nrrd")

        if mrhead_path.exists():
            volume, header = NRRDParser.load(mrhead_path)
            assert volume.ndim == 3
            assert volume.shape == (256, 256, 130)
            assert volume.min() >= 0

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_load_various_unsigned_dtypes(self, dtype: type) -> None:
        """Test loading NRRD volumes with 8, 16, and 32-bit unsigned integers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / f"test_{dtype.__name__}.nrrd"
            data = np.arange(64, dtype=dtype).reshape(4, 4, 4)
            NRRDParser.save(file_path, data)

            loaded_data, _ = NRRDParser.load(file_path)
            assert loaded_data.shape == (4, 4, 4)
            assert loaded_data.dtype == dtype
            assert np.array_equal(loaded_data, data)

    def test_file_not_found(self) -> None:
        """Test that attempting to load a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            NRRDParser.load("non_existent_volume_path.nrrd")

    def test_invalid_dimensions(self) -> None:
        """Test that 2D or 4D arrays are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 2D file
            file_path_2d = Path(tmpdir) / "test_2d.nrrd"
            nrrd.write(str(file_path_2d), np.zeros((10, 10), dtype=np.uint8))
            with pytest.raises(ValueError, match="Expected 3D volume"):
                NRRDParser.load(file_path_2d)

            # 4D file
            file_path_4d = Path(tmpdir) / "test_4d.nrrd"
            nrrd.write(str(file_path_4d), np.zeros((2, 2, 2, 2), dtype=np.uint8))
            with pytest.raises(ValueError, match="Expected 3D volume"):
                NRRDParser.load(file_path_4d)

    def test_synthetic_volume_generation(self) -> None:
        """Test the synthetic volume generator structure."""
        vol = NRRDParser.generate_synthetic_volume(shape=(32, 32, 32), dtype=np.uint16)
        assert vol.shape == (32, 32, 32)
        assert vol.dtype == np.uint16
        assert vol.max() > 0
        assert vol.min() == 0


class TestCSVParser:
    """Test suite for CSV plane coordinates parser."""

    def test_load_valid_sample_csv(self) -> None:
        """Test parsing the sample points.csv file."""
        csv_path = Path("data/points.csv")
        assert csv_path.exists(), "data/points.csv must exist"

        points = CSVParser.load_points(csv_path)
        assert isinstance(points, np.ndarray)
        assert points.ndim == 3
        assert points.shape[1] in (3, 4)  # 3 points (triangle) or 4 points (quad)
        assert points.shape[2] == 3  # (x, y, z)
        assert len(points) >= 1

    def test_parse_3point_format(self) -> None:
        """Test parsing CSV with 9 columns (3 points per row)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "triangle.csv"
            with open(csv_path, "w") as f:
                f.write("ax,ay,az,bx,by,bz,cx,cy,cz\n")
                f.write("1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0\n")
                f.write("10, 20, 30, 40, 50, 60, 70, 80, 90\n")

            pts = CSVParser.load_points(csv_path)
            assert pts.shape == (2, 3, 3)
            assert pts[0, 0, 0] == 1.0
            assert pts[1, 2, 2] == 90.0

    def test_parse_4point_format(self) -> None:
        """Test parsing CSV with 12 columns (4 points per row)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "quad.csv"
            with open(csv_path, "w") as f:
                f.write("a1x,a1y,a1z,a2x,a2y,a2z,bx,by,bz,cx,cy,cz\n")
                f.write("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\n")

            pts = CSVParser.load_points(csv_path)
            assert pts.shape == (1, 4, 3)
            assert pts[0, 3, 2] == 12.0

    def test_round_trip_save_and_load(self) -> None:
        """Test saving points to CSV and parsing them back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "roundtrip.csv"
            original = np.array([
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ])
            CSVParser.save_points(csv_path, original)
            loaded = CSVParser.load_points(csv_path)
            np.testing.assert_allclose(original, loaded, atol=1e-3)

    def test_csv_file_not_found(self) -> None:
        """Test error when CSV does not exist."""
        with pytest.raises(FileNotFoundError):
            CSVParser.load_points("non_existent_points.csv")
