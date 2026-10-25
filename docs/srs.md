# Software Requirements Specification (SRS)
## Exercise 1: 3D Volume Plane Projection to 2D Image & MP4 Video Generation

---

### Document Control
- **Document Version**: 1.0.0
- **Status**: Approved / Architectural Baseline
- **Course**: Introduction to Software Engineering (I2SE) - Exercise 1
- **Domain**: 3D Medical & Scientific Volume Slicing, Vector Mathematics, High-Performance Video Synthesis

---

## 1. Introduction

### 1.1 Project Scope & Overview
Modern scientific and medical imaging modalities—such as Computed Tomography (CT), Magnetic Resonance Imaging (MRI), and 3D ultrasound—produce volumetric data sets structured as 3D discrete scalar fields (voxel matrices). Analyzing these volumes often requires slicing the 3D domain along arbitrary, oblique planes to produce 2D planar images (Multi-Planar Reconstruction, or MPR).

This software system provides an optimized, modular pipeline that:
1. Loads 3D voxel matrices from Nearly Raw Raster Data (`.nrrd`) files supporting multi-bit unsigned integer formats (8-bit, 16-bit, and 32-bit).
2. Parses spatial trajectory keypoints from Comma-Separated Values (`.csv`) files defining planar cuts (triangles or trapezoids/rectangles).
3. Executes a mathematical vector-stepping projection algorithm with vectorized acceleration to sample the 3D volume onto a 2D discrete matrix with high computational throughput.
4. Linearly interpolates plane geometries across time to generate smooth planar motion and encodes the resulting sequence into an MP4 video consisting of exactly 240 frames at a standard frame rate (e.g., 30 FPS for an 8-second video).

### 1.2 Problem Definition
Given:
- A discrete 3D matrix $M \in \mathbb{N}^{s_x \times s_y \times s_z}$ where each voxel coordinate $(x, y, z) \in [0, s_x - 1] \times [0, s_y - 1] \times [0, s_z - 1]$ stores an unsigned intensity value $v$.
- A spatial planar domain defined by either:
  - **3 points** forming a planar triangle $a, b, c \in \mathbb{R}^3$, or
  - **4 points** forming a planar trapezoid / quad $a_1, a_2, b, c \in \mathbb{R}^3$.

The objective is to compute a 2D discrete image $E \in \mathbb{R}^{H \times W}$ representing the planar cross-section of $M$ across the plane. For every spatial sampling coordinate $p = (p_x, p_y, p_z) \in \mathbb{R}^3$ on the plane:
$$v = \begin{cases} 
M[\text{round}(p_x), \text{round}(p_y), \text{round}(p_z)], & \text{if } p \in [0, s_x - 1] \times [0, s_y - 1] \times [0, s_z - 1] \\
0, & \text{if } p \notin M \text{ (Out-of-Bounds)}
\end{cases}$$

---

## 2. System Architecture & Components

The system follows a modular, pipeline-oriented architecture consisting of four decoupled core components inside `src/` and a unified CLI orchestration layer.

```
+-----------------------------------------------------------------------------------+
|                                 Ex0.py / src/Ex0.py                               |
|                          (CLI Orchestration & Pipeline)                           |
+-------------------+-------------------------------+-------------------------------+
                    |                               |
                    v                               v
        +-----------------------+       +-----------------------+
        |      NRRD Loader      |       |      CSV Parser       |
        |  (src/nrrd_parser.py) |       |  (src/nrrd_parser.py) |
        +-----------+-----------+       +-----------+-----------+
                    |                               |
                    | 3D Matrix M                   | Plane Keyframes (K, P, 3)
                    +---------------+---------------+
                                    |
                                    v
                        +-----------------------+
                        |    Video Generator    |
                        |(src/video_generator.py|
                        +-----------+-----------+
                                    |
                                    | Interpolates to 240 frames
                                    |
                                    v
                        +-----------------------+
                        |   Projection Engine   |
                        |(src/projection_engine)|
                        +-----------+-----------+
                                    |
                                    | 240x 2D Frames E (H x W)
                                    |
                                    v
                        +-----------------------+
                        |   cv2.VideoWriter     |
                        |    (MP4 Output)       |
                        +-----------------------+
```

### 2.1 NRRD Loader (`src/nrrd_parser.py`)
- **Responsibilities**:
  - Load volumetric datasets stored in `.nrrd` (Nearly Raw Raster Data) format.
  - Parse metadata headers (volume dimensions, field data types, endianness, voxel spacing).
  - Support unsigned integer voxel precisions:
    - 8-bit unsigned integer (`uint8`, $[0, 255]$)
    - 16-bit unsigned integer (`uint16`, $[0, 65535]$)
    - 32-bit unsigned integer (`uint32`, $[0, 2^{32}-1]$)
  - Validate matrix dimensionality ($D = 3$) and raise structured errors for corrupted or unsupported inputs.
  - Provide synthetic volumetric generators for automated testing and benchmarking.

### 2.2 CSV Parser (`src/nrrd_parser.py`)
- **Responsibilities**:
  - Ingest planar coordinate definitions from `.csv` files.
  - Parse both 3-point planar configurations ($a, b, c$) and 4-point configurations ($a_1, a_2, b, c$).
  - Support both row-wise coordinate tuples (9 or 12 numerical elements per row) and multi-line coordinate blocks.
  - Validate spatial bounds, numerical finiteness, and geometric consistency (ensuring points do not collapse to a single point).

### 2.3 Projection Engine (`src/projection_engine.py`)
- **Responsibilities**:
  - Project a 3D matrix $M$ onto a discrete 2D matrix $E$.
  - Implement the fundamental **vector stepping algorithm**:
    1. Given plane points $a, b, c$ (or trapezoid $a_1, a_2, b, c$):
       - If 3 points: initialize apex $a_1 = a, a_2 = a$.
       - Compute leg distances: $D_{ab} = \|b - a_1\|$, $D_{ac} = \|c - a_2\|$.
       - Compute base distance: $D_{bc} = \|c - b\|$.
       - Calculate normalized direction vectors:
         $$v(a_1, b) = \frac{b - a_1}{D_{ab}}, \quad v(a_2, c) = \frac{c - a_2}{D_{ac}}$$
    2. Allocate output 2D matrix $E$ with spatial dimensions $H = \lceil D_{ab} \rceil, W = \lceil D_{bc} \rceil$.
    3. Step along the outer loop $y \in [0, H-1]$:
       - Update leg endpoints:
         $$a_1(y) = a_1 + y \cdot v(a_1, b), \quad a_2(y) = a_2 + y \cdot v(a_2, c)$$
       - Calculate current transverse width: $d_{12}(y) = \|a_2(y) - a_1(y)\|$.
       - Compute unit step vector along the transverse slice:
         $$dp = \frac{a_2(y) - a_1(y)}{d_{12}(y)} \quad (\text{for } d_{12}(y) > 0)$$
       - Step along the inner loop $x_1 \in [0, \lfloor d_{12}(y) \rfloor]$:
         - Sampling point: $p = a_1(y) + x_1 \cdot dp$.
         - Sample value $v = M[\text{round}(p)]$ if $p \in M$, else $v = 0$.
         - Store sampled value into $E[y, x_1]$.
  - Implement high-performance **NumPy vectorization**:
    - Precompute grid coordinate matrices using NumPy array broadcasting.
    - Perform vectorized bounds checking and batch voxel indexing to eliminate Python interpreter overhead.
    - Maintain algorithmic equivalence between iterative stepping and vectorized modes.

### 2.4 Video Generator (`src/video_generator.py`)
- **Responsibilities**:
  - Accept volumetric matrix $M$ and a sequence of keyframe plane definitions.
  - Guarantee output video contains **exactly 240 frames**.
  - Perform **linear trajectory interpolation**:
    - If the input CSV contains fewer than 240 plane definitions (e.g., $K < 240$), interpolate each coordinate component $P_i(t)$ linearly across the 240 time steps using continuous parameter $t \in [0, 1]$.
    - If $K = 1$, replicate the single plane for 240 frames.
    - If $K > 240$, resample uniformly to 240 frames.
  - Maintain uniform frame resolution across the video stream by calculating global bounding dimensions and applying zero-padding.
  - Normalize multi-bit voxel values to 8-bit RGB/BGR range $[0, 255]$ for video codec compliance.
  - Encode frames into an MP4 container utilizing `opencv-python` (`cv2.VideoWriter`) with standard MPEG-4 / H.264 fourcc codecs.

### 2.5 CLI Orchestration Layer (`src/Ex0.py` & root launcher `Ex0.py`)
- **CLI Interface**:
  ```bash
  python Ex0.py <file.nrrd> <file.csv> <file.mp4>
  ```
- **Error Handling & Reporting**:
  - Verify file paths and emit descriptive diagnostics.
  - Display progress indicators and performance metrics (FPS, total execution time).
  - Return exit status code 0 on success, non-zero on failure.

---

## 3. Performance & Non-Functional Requirements

### 3.1 Performance Requirements
| Metric | Specification Target | Minimum Required | Rationale |
| :--- | :--- | :--- | :--- |
| **Projection Speed** | **100+ FPS** | **30+ FPS** | Real-time / interactive slicing throughput on standard hardware for a $500^3$ voxel matrix. |
| **Batch Render Time** | $< 2.4$ seconds for 240 frames | $< 8.0$ seconds | Slicing 240 video frames must complete promptly without blocking interactive user workflows. |
| **Vectorization Gain** | $> 10\times$ speedup over pure Python loops | $> 5\times$ speedup | NumPy vectorized coordinate broadcasting avoids scalar Python overhead. |

### 3.2 Robustness & Reliability
- **Out-of-Bounds (OOB) Safety**:
  - If any coordinate of sample point $p = (p_x, p_y, p_z)$ satisfies $p_x < 0$, $p_x \ge s_x$, $p_y < 0$, $p_y \ge s_y$, $p_z < 0$, or $p_z \ge s_z$, the system must return $v = 0$.
  - Under no circumstances should out-of-bounds sampling raise an `IndexError` or crash the runtime.
- **Degenerate Plane Handling**:
  - When $d_{12}(y) = 0$ (e.g. at the apex of a triangle), division by zero must be prevented by setting $dp = \vec{0}$ and sampling only the single apex coordinate.
- **Data Type Fidelity**:
  - Preserve voxel dynamic range during projection.
  - Apply monotonic contrast scaling during 8-bit video conversion without clipping non-zero details.

### 3.3 Portability & Code Quality
- **Python Compatibility**: Python 3.9+ through Python 3.15+.
- **Clean Architecture**: Decoupled modules with strict separation of concerns.
- **Type Hinting**: 100% type hinting using standard `typing` annotations (`np.ndarray`, `Tuple`, `Optional`, `Union`, `Path`).
- **Documentation**: Comprehensive PEP 257 docstrings for all classes and public functions.

---

## 4. Testing Strategy

### 4.1 Unit Testing (`pytest`)
- **NRRD Parsing Tests (`test_nrrd.py`)**:
  - Load and verify 8-bit, 16-bit, and 32-bit integer volumes.
  - Verify rejection of non-existent files and invalid headers.
  - Verify CSV coordinate parsing across 3-point and 4-point formats.
- **Projection Tests (`test_projection.py`)**:
  - Verify triangle and trapezoid slicing against known analytical shapes (e.g. 3D spheres).
  - Verify that out-of-bounds coordinates return exact zero.
  - Verify numerical equivalence between `project_plane_vectorized` and `project_plane_stepping`.
- **Video Generator Tests**:
  - Verify linear interpolation creates exactly 240 frames.
  - Verify output `.mp4` file is generated, non-empty, and contains 240 decodable frames.

### 4.2 Performance Benchmarking
- Automated benchmark test measuring FPS for a $500 \times 500 \times 500$ voxel volume.
- Validation that the vectorized projection engine achieves $\ge 100$ FPS.

### 4.3 Edge Case Coverage
- Plane passing entirely outside volume $M$ $\to$ Frame filled with zeros.
- Oblique cut intersecting multiple faces of the volume bounding cube.
- Single-plane CSV ($K = 1$) interpolated smoothly to 240 frames.
- Multi-point CSV ($K > 240$) resampled correctly.

### 4.4 CLI Test Execution & Verification

Automated testing is executed via the `pytest` CLI runner from the project root directory.

#### Test Execution Commands:
```bash
# 1. Activate project virtual environment
source .venv/bin/activate

# 2. Run complete test suite (all 23 unit tests & benchmarks)
pytest -v

# 3. Run complete test suite with live stdout & benchmark metrics
pytest -v -s

# 4. Run tests by module
pytest tests/test_nrrd.py -v          # NRRD and CSV format parsers
pytest tests/test_projection.py -v    # 3D projection, equivalence, video synthesis

# 5. Run only performance benchmark with live telemetry
pytest tests/test_projection.py -k "benchmark" -s

# 6. Direct execution without manual venv activation
.venv/bin/pytest -v -s
```

#### Acceptance Criteria & Expected Results:
- **Test Pass Rate**: 100% (23/23 tests passing with 0 failures).
- **Throughput Benchmark**: Slicing the $500^3$ voxel matrix must output $\ge 100.0$ FPS in `test_500cubed_volume_performance_benchmark`.
- **Equivalence Verification**: $\max |E_{\text{stepping}} - E_{\text{vec}}| = 0$ across all test slices.
- **Container Verification**: Output MP4 must open and contain exactly 240 valid frames.
