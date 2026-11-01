"""NRRD and CSV Data Parsers for 3D Volume Slicing.

This module provides robust loaders and writers for:
- 3D Nearly Raw Raster Data (.nrrd) volume matrices with 8, 16, and 32-bit unsigned integers.
- Planar coordinate definition files (.csv) containing 3-point or 4-point specifications.
- Synthetic 3D volume generation for testing and benchmarking.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nrrd
import numpy as np


class NRRDParser:
    """Parser and validator for 3D Nearly Raw Raster Data (.nrrd) files."""

    SUPPORTED_DTYPES = (
        np.uint8,
        np.uint16,
        np.uint32,
    )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load and validate a 3D volume from a .nrrd file.

        Args:
            filepath: Path to the .nrrd file.

        Returns:
            Tuple containing:
                - data: 3D numpy ndarray representing the voxel matrix M (sx, sy, sz).
                - header: Dictionary containing NRRD metadata headers.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not 3-dimensional or has an unsupported dtype.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"NRRD file not found: {path.resolve()}")

        data, header = nrrd.read(str(path))

        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D volume matrix, got {data.ndim}D array with shape {data.shape}"
            )

        # Validate or cast unsigned integer types
        if data.dtype not in cls.SUPPORTED_DTYPES:
            # If float or signed int from some exporters, verify if convertible
            if np.issubdtype(data.dtype, np.unsignedinteger):
                pass
            elif np.issubdtype(data.dtype, np.integer) and data.min() >= 0:
                # Convert to unsigned if within positive range
                if data.max() <= np.iinfo(np.uint8).max:
                    data = data.astype(np.uint8)
                elif data.max() <= np.iinfo(np.uint16).max:
                    data = data.astype(np.uint16)
                else:
                    data = data.astype(np.uint32)
            else:
                raise ValueError(
                    f"Unsupported volume data type '{data.dtype}'. "
                    f"Expected 8-bit, 16-bit, or 32-bit unsigned integer (uint8, uint16, uint32)."
                )

        return data, header

    @classmethod
    def save(
        cls,
        filepath: Union[str, Path],
        data: np.ndarray,
        header: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a 3D numpy ndarray to a .nrrd file.

        Args:
            filepath: Destination path for the .nrrd file.
            data: 3D numpy ndarray to write.
            header: Optional dictionary containing NRRD header fields.

        Raises:
            ValueError: If data is not a 3D numpy ndarray.
        """
        if data.ndim != 3:
            raise ValueError(f"Data must be 3-dimensional, got shape {data.shape}")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        nrrd.write(str(path), data, header=header)

    @staticmethod
    def generate_synthetic_volume(
        shape: Tuple[int, int, int] = (128, 128, 128),
        dtype: type = np.uint8,
    ) -> np.ndarray:
        """Generate a deterministic synthetic 3D volume containing geometric shapes.

        The volume contains:
        - A central spherical shell with high intensity.
        - An inner solid sphere with medium intensity.
        - Planar cross-axis fiducials along coordinate axes.

        Args:
            shape: Shape (sx, sy, sz) of the 3D matrix.
            dtype: Unsigned integer type (np.uint8, np.uint16, np.uint32).

        Returns:
            3D numpy ndarray with the generated volume.
        """
        sx, sy, sz = shape
        volume = np.zeros(shape, dtype=dtype)

        # Scale intensity ceiling based on data type
        max_val = int(np.iinfo(dtype).max)
        val_outer = int(max_val * 0.9)
        val_inner = int(max_val * 0.5)
        val_axis = int(max_val * 0.75)

        cx, cy, cz = sx / 2.0, sy / 2.0, sz / 2.0
        r_outer = min(cx, cy, cz) * 0.75
        r_inner = min(cx, cy, cz) * 0.40

        # Coordinate grid
        x = np.arange(sx)[:, None, None]
        y = np.arange(sy)[None, :, None]
        z = np.arange(sz)[None, None, :]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2

        # Outer spherical shell
        outer_mask = (dist_sq <= r_outer**2) & (dist_sq >= (r_outer * 0.8) ** 2)
        volume[outer_mask] = val_outer

        # Inner solid core
        inner_mask = dist_sq <= r_inner**2
        volume[inner_mask] = val_inner

        # Coordinate axes bars (cross-axis markers)
        thick = max(1, int(min(sx, sy, sz) * 0.03))
        volume[int(cx) - thick : int(cx) + thick, int(cy) - thick : int(cy) + thick, :] = val_axis
        volume[:, int(cy) - thick : int(cy) + thick, int(cz) - thick : int(cz) + thick] = val_axis
        volume[int(cx) - thick : int(cx) + thick, :, int(cz) - thick : int(cz) + thick] = val_axis

        return volume


class CSVParser:
    """Parser for CSV files specifying plane coordinates in 3D space."""

    @classmethod
    def load_points(cls, filepath: Union[str, Path]) -> np.ndarray:
        """Load plane definitions from a CSV file.

        Supports two formats:
        - 3 points per plane (triangle): [ax, ay, az, bx, by, bz, cx, cy, cz] (9 values)
        - 4 points per plane (trapezoid/quad): [a1x, a1y, a1z, a2x, a2y, a2z, bx, by, bz, cx, cy, cz] (12 values)

        Also supports multi-line formats where coordinates are listed as rows of (x, y, z).

        Args:
            filepath: Path to the CSV file.

        Returns:
            np.ndarray of shape (K, num_points, 3), where num_points is 3 or 4,
            and K is the number of plane definitions.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV format cannot be parsed into 3D plane coordinates.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path.resolve()}")

        rows: List[List[float]] = []
        with open(path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                cleaned = [cell.strip() for cell in row if cell.strip() != ""]
                if not cleaned:
                    continue
                # Skip header rows that contain non-numeric data
                try:
                    numeric_row = [float(val) for val in cleaned]
                    rows.append(numeric_row)
                except ValueError:
                    # Header row detected, skip
                    continue

        if not rows:
            raise ValueError(f"No valid numeric plane coordinates found in {path}")

        # Check column dimensions
        first_len = len(rows[0])

        if first_len == 9:
            # Each row is a triangle: a(x,y,z), b(x,y,z), c(x,y,z)
            planes = []
            for r in rows:
                if len(r) != 9:
                    raise ValueError(f"Inconsistent column count. Expected 9, got {len(r)}")
                arr = np.array(r, dtype=np.float64).reshape(3, 3)
                planes.append(arr)
            return np.array(planes, dtype=np.float64)

        if first_len == 12:
            # Each row is a quad/trapezoid: a1, a2, b, c
            planes = []
            for r in rows:
                if len(r) != 12:
                    raise ValueError(f"Inconsistent column count. Expected 12, got {len(r)}")
                arr = np.array(r, dtype=np.float64).reshape(4, 3)
                planes.append(arr)
            return np.array(planes, dtype=np.float64)

        if first_len == 3:
            # 3 numbers per row: (x, y, z)
            # Group into 3-point planes (or 4-point planes if divisible by 4)
            total_points = len(rows)
            if total_points % 3 == 0 and total_points % 4 != 0:
                k = total_points // 3
                return np.array(rows, dtype=np.float64).reshape(k, 3, 3)
            elif total_points % 4 == 0 and total_points % 3 != 0:
                k = total_points // 4
                return np.array(rows, dtype=np.float64).reshape(k, 4, 3)
            elif total_points % 3 == 0:
                k = total_points // 3
                return np.array(rows, dtype=np.float64).reshape(k, 3, 3)
            else:
                raise ValueError(
                    f"Row count {total_points} with 3 columns is not divisible by 3 or 4"
                )

        raise ValueError(
            f"Unsupported CSV format with {first_len} columns per row. "
            f"Expected 9 columns (3 points), 12 columns (4 points), or 3 columns (x, y, z)."
        )

    @classmethod
    def save_points(cls, filepath: Union[str, Path], points: np.ndarray) -> None:
        """Save plane definitions to a CSV file in row-per-plane format.

        Args:
            filepath: Destination path for the CSV file.
            points: np.ndarray of shape (K, 3, 3) or (K, 4, 3).

        Raises:
            ValueError: If points array does not have expected dimensions.
        """
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] not in (3, 4):
            raise ValueError(
                f"Points array must have shape (K, 3, 3) or (K, 4, 3), got {points.shape}"
            )

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        k, p_count, _ = points.shape
        if p_count == 3:
            header = ["ax", "ay", "az", "bx", "by", "bz", "cx", "cy", "cz"]
        else:
            header = ["a1x", "a1y", "a1z", "a2x", "a2y", "a2z", "bx", "by", "bz", "cx", "cy", "cz"]

        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for plane in points:
                flat_row = plane.flatten().tolist()
                writer.writerow([f"{val:.4f}" for val in flat_row])
