#!/usr/bin/env python3
"""
GPU/CPU raster stack builder for CIR, DTM and DOM.

The script:
- Uses the CIR raster as the reference grid, CRS, resolution and extent.
- Reads DTM and DOM tiles from one or more folders.
- Uses only DTM/DOM tiles that overlap the CIR raster extent.
- Virtually mosaics the overlapping DTM/DOM tiles with GDAL VRT.
- Resamples DTM/DOM to the CIR grid with bilinear resampling.
- Computes:
    * CHM = DOM - DTM, negative values clamped to 0
    * CHM_Gaus5
    * NDVI
    * Slope_Gaus5 in degrees
    * TPI20, TPI50 and TPI70 using square moving windows in pixels
- Writes a standard multiband GeoTIFF, not a COG.
- Supports OUTPUT_COMPRESSION = "LZW" or "NONE".
- Supports OUTPUT_DTYPE = "float32" or "uint8".
- In uint8 mode, each band is explicitly scaled to 0–254 and 255 is reserved for nodata.
- Can process with either GPU/CuPy or CPU/SciPy.
- Handles temp/output files on different Windows drive letters safely.

Output band order:
    1  NIR
    2  Red
    3  Green
    4  TPI20
    5  TPI50
    6  TPI70
    7  CHM_Gaus5
    8  CHM
    9  NDVI
    10 Slope_Gaus5

Edit the USER CONFIGURATION block before running.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
from scipy.ndimage import uniform_filter as scipy_uniform_filter
from tqdm import tqdm

try:
    from osgeo import gdal
except ImportError as exc:
    raise SystemExit(
        "Kunne ikke importere osgeo.gdal. Installer GDAL Python bindings i miljøet "
        "som kjører scriptet."
    ) from exc


# CuPy is imported lazily only if PROCESSING_BACKEND = "cpu".
cp = None
cupy_gaussian_filter = None
cupy_uniform_filter = None


def _get_required_env_path(name: str) -> Path:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Miljøvariabelen {name} mangler. Sett den før kjøring.")
    return Path(value)


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


# =============================================================================
# USER CONFIGURATION — EDIT THESE VALUES BEFORE RUNNING
# =============================================================================

# ---------------------------------------------------------------------------
# Areas to process
# ---------------------------------------------------------------------------
# NB: This list currently contains "10_" twice, so it will be processed twice.
# Remove the duplicate if that was not intended.
AREA_LIST = ["04_", "06_"]

# Alternative full 02–12 batch:
# AREA_LIST = ["02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "10_", "11_", "12_"]

# If True, the script logs a failed area and continues to the next.
# If False, it stops immediately on the first area error.
CONTINUE_ON_AREA_ERROR = True

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
# Required env vars (auto-loaded from .env if present):
#   BASE_PATH=<path to segmentation root>
#   TOPO_PATH=<path to topography root>
KILDE = _get_required_env_path("BASE_PATH")
TOPO_PATH = _get_required_env_path("TOPO_PATH")

# ---------------------------------------------------------------------------
# Mapping from area code to height-data folder codes
# ---------------------------------------------------------------------------
HEIGHTDATA_BY_AREA = {
    "01_": ["14-1819"],
    "02_": ["12-14", "13-14"],
    "03_": ["1213-13", "12-14"],
    "04_": ["1213-13"],
    "05_": ["1213-12"],
    "06_": ["1213-11", "1213-12"],
    "07_": ["11-11", "1213-11"],
    "08_": ["11-10", "11-11"],
    "09_": ["10-10", "11-10"],
    "10_": ["10-10"],
    "11_": ["10-11"],
    "12_": ["10-12", "11-12"],
}

# ---------------------------------------------------------------------------
# Input CIR naming
# ---------------------------------------------------------------------------
CIR_FILENAME = "merged_1m.tif"

# ---------------------------------------------------------------------------
# CIR band mapping
# ---------------------------------------------------------------------------
# Correct CIR mapping for this dataset:
#   1 = NIR, 2 = Red, 3 = Green
NIR_BAND = 1
RED_BAND = 2
GREEN_BAND = 3

# ---------------------------------------------------------------------------
# Output and temporary naming
# ---------------------------------------------------------------------------
OUTPUT_FILENAME_TEMPLATE = "{area}_cir_topo_stack.tif"

# Optional temp root. If set, each area gets its own temp subfolder under this path.
# Example:
# TEMP_DIR_BASE = KILDE / "_temp"
TEMP_DIR_BASE: Path | None = None

# ---------------------------------------------------------------------------
# Processing backend
# ---------------------------------------------------------------------------
# "gpu" = CuPy/CUDA
# "cpu" = NumPy/SciPy
PROCESSING_BACKEND = "cpu"

# ---------------------------------------------------------------------------
# Output options
# ---------------------------------------------------------------------------
OVERWRITE_OUTPUT = True

# Allowed values are strictly "LZW" or "NONE".
OUTPUT_COMPRESSION = "LZW"

# Allowed values are strictly "float32" or "uint8".
# "float32" preserves the original computed values.
# "uint8" writes compact 8-bit output using UINT8_SCALE_RANGES below.
OUTPUT_DTYPE = "uint8"

# Float32 output nodata value.
FLOAT32_NODATA_VALUE = -9999.0

# UInt8 output nodata value.
# 255 is reserved for nodata; valid scaled values use 0–254.
UINT8_NODATA_VALUE = 255

# Value ranges used when OUTPUT_DTYPE = "uint8".
# Values below/above the stated range are clipped before scaling to 0–254.
# Adjust CHM/TPI ranges if your study areas contain larger values than expected.
UINT8_SCALE_RANGES = {
    "NIR": (0.0, 255.0),
    "Red": (0.0, 255.0),
    "Green": (0.0, 255.0),
    "TPI20": (-5.0, 5.0),
    "TPI50": (-10.0, 10.0),
    "TPI70": (-15.0, 15.0),
    "CHM_Gaus5": (0.0, 50.0),
    "CHM": (0.0, 50.0),
    "NDVI": (-1.0, 1.0),
    "Slope_Gaus5": (0.0, 90.0),
}

# ---------------------------------------------------------------------------
# Raster processing
# ---------------------------------------------------------------------------
BLOCK_SIZE = 8192
GAUSS_SIGMA_M = 5.0

# TPI square window sizes in pixels.
TPI_WINDOW_PIXELS = (20, 50, 70)

# GDAL in-memory cache.
GDAL_CACHE_MB = 32768

# Parallel scanning of tile metadata.
SCAN_WORKERS = min(32, max(4, (os.cpu_count() or 8) // 2))

# ---------------------------------------------------------------------------
# Progress prints
# ---------------------------------------------------------------------------
# Print progress during tile overlap scanning every N candidate files.
TILE_SCAN_PRINT_EVERY = 500

# Print progress during raster processing every N windows.
WINDOW_PRINT_EVERY = 5

# Supported input tile extensions.
SUPPORTED_RASTER_SUFFIXES = {".tif", ".tiff", ".vrt"}


# =============================================================================
# OUTPUT STACK DESCRIPTION
# =============================================================================
BAND_DESCRIPTIONS = (
    "NIR",
    "Red",
    "Green",
    "TPI20",
    "TPI50",
    "TPI70",
    "CHM_Gaus5",
    "CHM",
    "NDVI",
    "Slope_Gaus5",
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass(frozen=True)
class Config:
    area: str
    area_path: Path

    cir_path: Path
    dtm_dirs: tuple[Path, ...]
    dom_dirs: tuple[Path, ...]
    out_path: Path
    tmp_dir: Path | None

    nir_band: int
    red_band: int
    green_band: int

    processing_backend: str
    overwrite_output: bool
    output_compression: str
    output_dtype: str
    float32_nodata: float
    uint8_nodata: int
    uint8_scale_ranges: dict[str, tuple[float, float]]

    block_size: int
    gauss_sigma_m: float
    tpi_window_pixels: tuple[int, ...]
    gdal_cache_mb: int

    scan_workers: int
    tile_scan_print_every: int
    window_print_every: int


@dataclass(frozen=True)
class RasterFootprint:
    path: Path
    bounds_in_cir_crs: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class ProcessingGeometry:
    xres: float
    yres: float
    sigma_px_x: float
    sigma_px_y: float
    tpi_window_sizes: tuple[int, ...]
    max_tpi_radius_px: int
    halo_px: int


# =============================================================================
# CONFIGURATION AND RUNTIME SETUP
# =============================================================================
def build_heightdata_dirs(area: str) -> tuple[list[Path], list[Path]]:
    if area not in HEIGHTDATA_BY_AREA:
        raise KeyError(
            f"Område '{area}' finnes ikke i HEIGHTDATA_BY_AREA. "
            f"Kjente områder: {', '.join(HEIGHTDATA_BY_AREA)}"
        )

    folder_codes = HEIGHTDATA_BY_AREA[area]

    dtm_dirs = [
        TOPO_PATH / "DTM1_UTM33_hele_norge" / f"DTM1_{folder_code}_UTM33"
        for folder_code in folder_codes
    ]

    dom_dirs = [
        TOPO_PATH / "DOM1_UTM33_hele_norge" / f"DOM1_{folder_code}_UTM33"
        for folder_code in folder_codes
    ]

    return dtm_dirs, dom_dirs


def load_config(area: str) -> Config:
    area_path = KILDE / area
    dtm_dirs, dom_dirs = build_heightdata_dirs(area)

    tmp_dir = None
    if TEMP_DIR_BASE is not None:
        tmp_dir = TEMP_DIR_BASE / area

    return Config(
        area=area,
        area_path=area_path,
        cir_path=area_path / "orto" / CIR_FILENAME,
        dtm_dirs=tuple(dtm_dirs),
        dom_dirs=tuple(dom_dirs),
        out_path=area_path / "output" / OUTPUT_FILENAME_TEMPLATE.format(area=area),
        tmp_dir=tmp_dir,
        nir_band=int(NIR_BAND),
        red_band=int(RED_BAND),
        green_band=int(GREEN_BAND),
        processing_backend=PROCESSING_BACKEND.strip().lower(),
        overwrite_output=bool(OVERWRITE_OUTPUT),
        output_compression=OUTPUT_COMPRESSION.strip().upper(),
        output_dtype=OUTPUT_DTYPE.strip().lower(),
        float32_nodata=float(FLOAT32_NODATA_VALUE),
        uint8_nodata=int(UINT8_NODATA_VALUE),
        uint8_scale_ranges={
            str(name): (float(bounds[0]), float(bounds[1]))
            for name, bounds in UINT8_SCALE_RANGES.items()
        },
        block_size=int(BLOCK_SIZE),
        gauss_sigma_m=float(GAUSS_SIGMA_M),
        tpi_window_pixels=tuple(int(v) for v in TPI_WINDOW_PIXELS),
        gdal_cache_mb=int(GDAL_CACHE_MB),
        scan_workers=int(SCAN_WORKERS),
        tile_scan_print_every=int(TILE_SCAN_PRINT_EVERY),
        window_print_every=int(WINDOW_PRINT_EVERY),
    )


def configure_runtime(gdal_cache_mb: int) -> None:
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt")
    os.environ.setdefault("GDAL_CACHEMAX", str(gdal_cache_mb))
    os.environ.setdefault("CPL_DEBUG", "OFF")

    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("GDAL_CACHEMAX", str(gdal_cache_mb))
    gdal.UseExceptions()


def initialize_processing_backend(processing_backend: str) -> None:
    global cp, cupy_gaussian_filter, cupy_uniform_filter

    if processing_backend not in {"gpu", "cpu"}:
        raise ValueError('PROCESSING_BACKEND må være enten "gpu" eller "cpu".')

    if processing_backend == "cpu":
        print("[BACKEND] Bruker CPU/SciPy-beregninger. CuPy importeres ikke.")
        return

    print("[BACKEND] Bruker GPU/CuPy-beregninger. Importerer CuPy ...")
    try:
        import cupy as _cp
        from cupyx.scipy.ndimage import gaussian_filter as _cupy_gaussian_filter
        from cupyx.scipy.ndimage import uniform_filter as _cupy_uniform_filter
    except Exception as exc:
        raise SystemExit(
            "Kunne ikke importere CuPy i GPU-modus. "
            "Sett PROCESSING_BACKEND = 'cpu' for å kjøre uten GPU, "
            "eller reparer CuPy/CUDA-installasjonen."
        ) from exc

    cp = _cp
    cupy_gaussian_filter = _cupy_gaussian_filter
    cupy_uniform_filter = _cupy_uniform_filter

    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    gpu_count = cp.cuda.runtime.getDeviceCount()
    print(f"[BACKEND] CuPy klar. Antall CUDA-enheter: {gpu_count}.")


def validate_config(config: Config) -> None:
    if not config.cir_path.exists():
        raise FileNotFoundError(f"Fant ikke CIR-fil: {config.cir_path}")

    if not config.dtm_dirs:
        raise ValueError("DTM_TILE_DIRS er tom.")
    if not config.dom_dirs:
        raise ValueError("DOM_TILE_DIRS er tom.")

    for directory in (*config.dtm_dirs, *config.dom_dirs):
        if not directory.exists():
            raise FileNotFoundError(f"Fant ikke mappe: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Forventet mappe, fikk: {directory}")

    if config.output_compression not in {"LZW", "NONE"}:
        raise ValueError('OUTPUT_COMPRESSION må være enten "LZW" eller "NONE".')

    if config.output_dtype not in {"float32", "uint8"}:
        raise ValueError('OUTPUT_DTYPE må være enten "float32" eller "uint8".')

    if not (0 <= config.uint8_nodata <= 255):
        raise ValueError("UINT8_NODATA_VALUE må være mellom 0 og 255.")

    if config.output_dtype == "uint8":
        missing_ranges = [
            band for band in BAND_DESCRIPTIONS if band not in config.uint8_scale_ranges
        ]
        if missing_ranges:
            raise ValueError(
                "UINT8_SCALE_RANGES mangler skaleringsintervall for: "
                + ", ".join(missing_ranges)
            )

        for band_name in BAND_DESCRIPTIONS:
            lower, upper = config.uint8_scale_ranges[band_name]
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(
                    f"Ugyldig skaleringsintervall for {band_name}: "
                    f"({lower}, {upper}). Forventet lower < upper."
                )

    if config.block_size <= 0:
        raise ValueError("BLOCK_SIZE må være > 0.")

    if config.gauss_sigma_m <= 0:
        raise ValueError("GAUSS_SIGMA_M må være > 0.")

    if not config.tpi_window_pixels:
        raise ValueError("TPI_WINDOW_PIXELS må inneholde minst én vindusstørrelse.")
    if any(v <= 0 for v in config.tpi_window_pixels):
        raise ValueError("Alle TPI-vindusstørrelser må være > 0.")

    if len(set(config.tpi_window_pixels)) != len(config.tpi_window_pixels):
        raise ValueError("TPI_WINDOW_PIXELS inneholder duplikater.")

    if config.tile_scan_print_every <= 0:
        raise ValueError("TILE_SCAN_PRINT_EVERY må være > 0.")
    if config.window_print_every <= 0:
        raise ValueError("WINDOW_PRINT_EVERY må være > 0.")


def validate_cir(config: Config) -> None:
    with rasterio.open(config.cir_path) as cir:
        if cir.crs is None:
            raise ValueError("CIR mangler CRS.")
        if cir.count < max(config.nir_band, config.red_band, config.green_band):
            raise ValueError(
                f"CIR har bare {cir.count} bånd, men bandmappingen krever minst "
                f"{max(config.nir_band, config.red_band, config.green_band)} bånd."
            )
        if cir.transform.a == 0 or cir.transform.e == 0:
            raise ValueError("CIR har ugyldig transform/oppløsning.")


# =============================================================================
# TILE DISCOVERY AND VRT CREATION
# =============================================================================
def iter_raster_files(directories: Sequence[Path]) -> list[Path]:
    files: list[Path] = []

    for directory in directories:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_RASTER_SUFFIXES:
                files.append(path)

    unique_files = sorted(set(files))
    if not unique_files:
        raise FileNotFoundError(
            f"Fant ingen rasterfiler i: {', '.join(str(directory) for directory in directories)}"
        )
    return unique_files


def rectangles_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    a_left, a_bottom, a_right, a_top = a
    b_left, b_bottom, b_right, b_top = b

    return not (
        a_right <= b_left or a_left >= b_right or a_top <= b_bottom or a_bottom >= b_top
    )


def tile_bounds_in_cir_crs(path: Path, cir_crs) -> RasterFootprint:
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return RasterFootprint(path, None)

            bounds = transform_bounds(
                src.crs,
                cir_crs,
                *src.bounds,
                densify_pts=21,
            )
            return RasterFootprint(path, bounds)

    except RasterioIOError:
        return RasterFootprint(path, None)


def find_overlapping_tiles(
    paths: Sequence[Path],
    cir_crs,
    cir_bounds: tuple[float, float, float, float],
    workers: int,
    label: str,
    print_every: int,
) -> list[Path]:
    print(f"[{label}] Starter overlappskontroll for {len(paths):,} kandidatfliser ...")
    started = time.perf_counter()

    out: list[Path] = []
    checked = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        footprints = pool.map(lambda p: tile_bounds_in_cir_crs(p, cir_crs), paths)

        for checked, footprint in enumerate(
            tqdm(
                footprints,
                total=len(paths),
                desc=f"Sjekker overlap {label}",
                unit="fil",
            ),
            start=1,
        ):
            if footprint.bounds_in_cir_crs is not None:
                if rectangles_overlap(footprint.bounds_in_cir_crs, cir_bounds):
                    out.append(footprint.path)

            if checked % print_every == 0 or checked == len(paths):
                elapsed = time.perf_counter() - started
                print(
                    f"[{label}] Kontrollert {checked:,}/{len(paths):,} fliser | "
                    f"overlappende: {len(out):,} | {elapsed:,.1f} s"
                )

    if not out:
        raise RuntimeError(f"Ingen {label}-fliser overlapper CIR-området.")

    elapsed = time.perf_counter() - started
    print(
        f"[{label}] Ferdig overlappskontroll: {len(out):,} av {len(paths):,} fliser brukes "
        f"({elapsed:,.1f} s)."
    )
    return out


def build_vrt(tile_paths: Sequence[Path], vrt_path: Path) -> Path:
    print(f"[VRT] Bygger {vrt_path.name} fra {len(tile_paths):,} fliser ...")
    started = time.perf_counter()

    options = gdal.BuildVRTOptions(resampleAlg="bilinear")
    vrt = gdal.BuildVRT(
        str(vrt_path), [str(path) for path in tile_paths], options=options
    )
    if vrt is None:
        raise RuntimeError(f"Klarte ikke å bygge VRT: {vrt_path}")

    vrt.FlushCache()
    vrt = None

    elapsed = time.perf_counter() - started
    print(f"[VRT] Ferdig {vrt_path.name} ({elapsed:,.1f} s).")
    return vrt_path


# =============================================================================
# WINDOW GEOMETRY
# =============================================================================
def make_processing_geometry(
    cir_ds: rasterio.io.DatasetReader,
    gauss_sigma_m: float,
    tpi_window_pixels: tuple[int, ...],
) -> ProcessingGeometry:
    xres = abs(float(cir_ds.transform.a))
    yres = abs(float(cir_ds.transform.e))

    if xres <= 0 or yres <= 0:
        raise ValueError("Ugyldig pixelstørrelse i CIR.")

    sigma_px_x = gauss_sigma_m / xres
    sigma_px_y = gauss_sigma_m / yres

    max_tpi_window = max(tpi_window_pixels)

    # For even-sized windows, scipy/cupy filters use their standard centering convention.
    max_tpi_radius_px = int(math.ceil(max_tpi_window / 2.0))

    # gaussian_filter default truncate=4.0.
    gaussian_halo = int(math.ceil(4.0 * max(sigma_px_x, sigma_px_y))) + 2
    tpi_halo = max_tpi_radius_px + 2
    halo_px = max(gaussian_halo, tpi_halo, 4)

    return ProcessingGeometry(
        xres=xres,
        yres=yres,
        sigma_px_x=sigma_px_x,
        sigma_px_y=sigma_px_y,
        tpi_window_sizes=tpi_window_pixels,
        max_tpi_radius_px=max_tpi_radius_px,
        halo_px=halo_px,
    )


def block_windows(width: int, height: int, block_size: int) -> Iterable[Window]:
    for row_off in range(0, height, block_size):
        window_height = min(block_size, height - row_off)

        for col_off in range(0, width, block_size):
            window_width = min(block_size, width - col_off)
            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=window_width,
                height=window_height,
            )


def expanded_window(core: Window, halo: int, width: int, height: int) -> Window:
    col0 = max(0, int(core.col_off) - halo)
    row0 = max(0, int(core.row_off) - halo)
    col1 = min(width, int(core.col_off + core.width) + halo)
    row1 = min(height, int(core.row_off + core.height) + halo)

    return Window(
        col_off=col0,
        row_off=row0,
        width=col1 - col0,
        height=row1 - row0,
    )


def crop_slices(core: Window, expanded: Window) -> tuple[slice, slice]:
    row_start = int(core.row_off - expanded.row_off)
    col_start = int(core.col_off - expanded.col_off)
    row_end = row_start + int(core.height)
    col_end = col_start + int(core.width)

    return slice(row_start, row_end), slice(col_start, col_end)


# =============================================================================
# GPU IMPLEMENTATION
# =============================================================================
def to_gpu_float32(arr: np.ndarray, nodata_value: float | int | None):
    gpu = cp.asarray(arr, dtype=cp.float32)

    if nodata_value is not None:
        gpu = cp.where(gpu == cp.float32(nodata_value), cp.nan, gpu)

    gpu = cp.where(cp.isfinite(gpu), gpu, cp.nan)
    return gpu


def gpu_normalized_gaussian_filter(values, sigma_y: float, sigma_x: float):
    valid = cp.isfinite(values)
    numeric = cp.where(valid, values, cp.float32(0.0))
    weights = valid.astype(cp.float32)

    filtered_values = cupy_gaussian_filter(
        numeric, sigma=(sigma_y, sigma_x), mode="nearest"
    )
    filtered_weights = cupy_gaussian_filter(
        weights, sigma=(sigma_y, sigma_x), mode="nearest"
    )

    result = filtered_values / cp.maximum(filtered_weights, cp.float32(1e-6))
    result = cp.where(filtered_weights > 0, result, cp.nan)
    result = cp.where(valid, result, cp.nan)
    return result.astype(cp.float32, copy=False)


def gpu_weighted_local_mean(values, window_size: int):
    valid = cp.isfinite(values)
    numeric = cp.where(valid, values, cp.float32(0.0))
    weights = valid.astype(cp.float32)

    local_sum_like = cupy_uniform_filter(
        numeric,
        size=(window_size, window_size),
        mode="nearest",
    )
    local_weight_like = cupy_uniform_filter(
        weights,
        size=(window_size, window_size),
        mode="nearest",
    )

    mean = local_sum_like / cp.maximum(local_weight_like, cp.float32(1e-6))
    mean = cp.where(local_weight_like > 0, mean, cp.nan)
    return mean.astype(cp.float32, copy=False)


def gpu_slope_degrees(dtm, xres: float, yres: float):
    dz_dy, dz_dx = cp.gradient(dtm, cp.float32(yres), cp.float32(xres))
    slope = cp.degrees(cp.arctan(cp.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)))
    slope = cp.where(cp.isfinite(dtm), slope, cp.nan)
    return slope.astype(cp.float32, copy=False)


def gpu_compute_stack(
    nir_np: np.ndarray,
    red_np: np.ndarray,
    green_np: np.ndarray,
    dtm_np: np.ndarray,
    dom_np: np.ndarray,
    cir_nodata: float | int | None,
    dtm_nodata: float | int | None,
    dom_nodata: float | int | None,
    geom: ProcessingGeometry,
    output_nodata: float,
) -> list[np.ndarray]:
    nir = to_gpu_float32(nir_np, cir_nodata)
    red = to_gpu_float32(red_np, cir_nodata)
    green = to_gpu_float32(green_np, cir_nodata)
    dtm = to_gpu_float32(dtm_np, dtm_nodata)
    dom = to_gpu_float32(dom_np, dom_nodata)

    input_valid = (
        cp.isfinite(nir)
        & cp.isfinite(red)
        & cp.isfinite(green)
        & cp.isfinite(dtm)
        & cp.isfinite(dom)
    )

    chm = dom - dtm
    chm = cp.maximum(chm, cp.float32(0.0))
    chm = cp.where(input_valid, chm, cp.nan)

    chm_gaus5 = gpu_normalized_gaussian_filter(
        chm,
        sigma_y=geom.sigma_px_y,
        sigma_x=geom.sigma_px_x,
    )

    denominator = nir + red
    ndvi = cp.full_like(denominator, cp.nan, dtype=cp.float32)
    valid_ndvi = (
        input_valid
        & cp.isfinite(denominator)
        & (cp.abs(denominator) > cp.float32(1e-6))
    )
    cp.divide(
        nir - red,
        denominator,
        out=ndvi,
        where=valid_ndvi,
    )

    dtm_for_topo = cp.where(input_valid, dtm, cp.nan)

    tpi_layers = []
    for window_size in geom.tpi_window_sizes:
        local_mean = gpu_weighted_local_mean(dtm_for_topo, window_size)
        tpi_layer = dtm_for_topo - local_mean
        tpi_layer = cp.where(input_valid, tpi_layer, cp.nan)
        tpi_layers.append(tpi_layer)

    slope = gpu_slope_degrees(dtm_for_topo, geom.xres, geom.yres)
    slope_gaus5 = gpu_normalized_gaussian_filter(
        slope,
        sigma_y=geom.sigma_px_y,
        sigma_x=geom.sigma_px_x,
    )
    slope_gaus5 = cp.where(input_valid, slope_gaus5, cp.nan)

    layers = [
        nir,
        red,
        green,
        *tpi_layers,
        chm_gaus5,
        chm,
        ndvi,
        slope_gaus5,
    ]

    output_layers: list[np.ndarray] = []

    for layer in layers:
        output_layers.append(cp.asnumpy(layer.astype(cp.float32, copy=False)))

    return output_layers


# =============================================================================
# CPU IMPLEMENTATION
# =============================================================================
def to_cpu_float32(arr: np.ndarray, nodata_value: float | int | None) -> np.ndarray:
    cpu = np.asarray(arr, dtype=np.float32)

    if nodata_value is not None:
        cpu = np.where(cpu == np.float32(nodata_value), np.nan, cpu)

    cpu = np.where(np.isfinite(cpu), cpu, np.nan)
    return cpu.astype(np.float32, copy=False)


def cpu_normalized_gaussian_filter(
    values: np.ndarray,
    sigma_y: float,
    sigma_x: float,
) -> np.ndarray:
    valid = np.isfinite(values)
    numeric = np.where(valid, values, np.float32(0.0)).astype(np.float32, copy=False)
    weights = valid.astype(np.float32)

    filtered_values = scipy_gaussian_filter(
        numeric, sigma=(sigma_y, sigma_x), mode="nearest"
    )
    filtered_weights = scipy_gaussian_filter(
        weights, sigma=(sigma_y, sigma_x), mode="nearest"
    )

    result = filtered_values / np.maximum(filtered_weights, np.float32(1e-6))
    result = np.where(filtered_weights > 0, result, np.nan)
    result = np.where(valid, result, np.nan)
    return result.astype(np.float32, copy=False)


def cpu_weighted_local_mean(values: np.ndarray, window_size: int) -> np.ndarray:
    valid = np.isfinite(values)
    numeric = np.where(valid, values, np.float32(0.0)).astype(np.float32, copy=False)
    weights = valid.astype(np.float32)

    local_sum_like = scipy_uniform_filter(
        numeric,
        size=(window_size, window_size),
        mode="nearest",
    )
    local_weight_like = scipy_uniform_filter(
        weights,
        size=(window_size, window_size),
        mode="nearest",
    )

    mean = local_sum_like / np.maximum(local_weight_like, np.float32(1e-6))
    mean = np.where(local_weight_like > 0, mean, np.nan)
    return mean.astype(np.float32, copy=False)


def cpu_slope_degrees(dtm: np.ndarray, xres: float, yres: float) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(dtm, np.float32(yres), np.float32(xres))
    slope = np.degrees(np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)))
    slope = np.where(np.isfinite(dtm), slope, np.nan)
    return slope.astype(np.float32, copy=False)


def cpu_compute_stack(
    nir_np: np.ndarray,
    red_np: np.ndarray,
    green_np: np.ndarray,
    dtm_np: np.ndarray,
    dom_np: np.ndarray,
    cir_nodata: float | int | None,
    dtm_nodata: float | int | None,
    dom_nodata: float | int | None,
    geom: ProcessingGeometry,
    output_nodata: float,
) -> list[np.ndarray]:
    nir = to_cpu_float32(nir_np, cir_nodata)
    red = to_cpu_float32(red_np, cir_nodata)
    green = to_cpu_float32(green_np, cir_nodata)
    dtm = to_cpu_float32(dtm_np, dtm_nodata)
    dom = to_cpu_float32(dom_np, dom_nodata)

    input_valid = (
        np.isfinite(nir)
        & np.isfinite(red)
        & np.isfinite(green)
        & np.isfinite(dtm)
        & np.isfinite(dom)
    )

    chm = dom - dtm
    chm = np.maximum(chm, np.float32(0.0))
    chm = np.where(input_valid, chm, np.nan)

    chm_gaus5 = cpu_normalized_gaussian_filter(
        chm,
        sigma_y=geom.sigma_px_y,
        sigma_x=geom.sigma_px_x,
    )

    denominator = nir + red
    ndvi = np.full_like(denominator, np.nan, dtype=np.float32)
    valid_ndvi = (
        input_valid
        & np.isfinite(denominator)
        & (np.abs(denominator) > np.float32(1e-6))
    )
    np.divide(
        nir - red,
        denominator,
        out=ndvi,
        where=valid_ndvi,
    )

    dtm_for_topo = np.where(input_valid, dtm, np.nan)

    tpi_layers: list[np.ndarray] = []
    for window_size in geom.tpi_window_sizes:
        local_mean = cpu_weighted_local_mean(dtm_for_topo, window_size)
        tpi_layer = dtm_for_topo - local_mean
        tpi_layer = np.where(input_valid, tpi_layer, np.nan)
        tpi_layers.append(tpi_layer.astype(np.float32, copy=False))

    slope = cpu_slope_degrees(dtm_for_topo, geom.xres, geom.yres)
    slope_gaus5 = cpu_normalized_gaussian_filter(
        slope,
        sigma_y=geom.sigma_px_y,
        sigma_x=geom.sigma_px_x,
    )
    slope_gaus5 = np.where(input_valid, slope_gaus5, np.nan)

    layers = [
        nir,
        red,
        green,
        *tpi_layers,
        chm_gaus5,
        chm,
        ndvi,
        slope_gaus5,
    ]

    output_layers: list[np.ndarray] = []

    for layer in layers:
        output_layers.append(layer.astype(np.float32, copy=False))

    return output_layers


# =============================================================================
# OUTPUT ENCODING
# =============================================================================
def encode_layers_for_output(
    layers: list[np.ndarray],
    config: Config,
) -> list[np.ndarray]:
    if len(layers) != len(BAND_DESCRIPTIONS):
        raise RuntimeError(
            f"Antall lag matcher ikke BAND_DESCRIPTIONS: "
            f"{len(layers)} vs {len(BAND_DESCRIPTIONS)}."
        )

    if config.output_dtype == "float32":
        encoded_layers: list[np.ndarray] = []
        for layer in layers:
            encoded = np.where(
                np.isfinite(layer),
                layer,
                np.float32(config.float32_nodata),
            ).astype(np.float32, copy=False)
            encoded_layers.append(encoded)
        return encoded_layers

    if config.output_dtype == "uint8":
        encoded_layers = []

        for band_name, layer in zip(BAND_DESCRIPTIONS, layers, strict=False):
            lower, upper = config.uint8_scale_ranges[band_name]

            encoded = np.full(
                layer.shape,
                config.uint8_nodata,
                dtype=np.uint8,
            )

            valid = np.isfinite(layer)
            if np.any(valid):
                clipped = np.clip(layer[valid], lower, upper)
                scaled = np.rint(((clipped - lower) / (upper - lower)) * 254.0).astype(
                    np.uint8
                )
                encoded[valid] = scaled

            encoded_layers.append(encoded)

        return encoded_layers

    raise ValueError('OUTPUT_DTYPE må være enten "float32" eller "uint8".')


# =============================================================================
# OUTPUT GEO TIFF CREATION
# =============================================================================
def create_output_profile(
    cir: rasterio.io.DatasetReader,
    config: Config,
) -> dict:
    if config.output_dtype == "float32":
        output_dtype = "float32"
        output_nodata = float(config.float32_nodata)
        lzw_predictor = 3
    elif config.output_dtype == "uint8":
        output_dtype = "uint8"
        output_nodata = int(config.uint8_nodata)
        lzw_predictor = 2
    else:
        raise ValueError('OUTPUT_DTYPE må være enten "float32" eller "uint8".')

    profile = cir.profile.copy()
    profile.update(
        driver="GTiff",
        dtype=output_dtype,
        count=len(BAND_DESCRIPTIONS),
        nodata=output_nodata,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        bigtiff="IF_SAFER",
        interleave="band",
    )

    if config.output_compression == "LZW":
        profile.update(
            compress="LZW",
            predictor=lzw_predictor,
        )
    elif config.output_compression == "NONE":
        profile.update(
            compress="NONE",
        )
    else:
        raise ValueError('OUTPUT_COMPRESSION må være enten "LZW" eller "NONE".')

    return profile


def write_stack_gtiff(
    cir_path: Path,
    dtm_vrt_path: Path,
    dom_vrt_path: Path,
    tmp_gtiff: Path,
    config: Config,
) -> ProcessingGeometry:
    print(
        f"[STACK] Starter {config.processing_backend.upper()}-beregning "
        "og skriving av standard GeoTIFF ..."
    )
    stack_started = time.perf_counter()

    with rasterio.open(cir_path) as cir:
        geom = make_processing_geometry(
            cir,
            config.gauss_sigma_m,
            config.tpi_window_pixels,
        )
        profile = create_output_profile(
            cir,
            config,
        )

        print(
            f"[STACK] CIR-grid: {cir.width:,} x {cir.height:,} piksler | "
            f"oppløsning: {geom.xres:g} x {geom.yres:g} | "
            f"halo: {geom.halo_px} piksler"
        )
        print(
            "[STACK] TPI-vinduer: "
            + ", ".join(f"{size} x {size}" for size in geom.tpi_window_sizes)
            + " piksler."
        )
        print(
            f"[STACK] Gaussian sigma: {config.gauss_sigma_m:g} m -> "
            f"{geom.sigma_px_x:.3f} x {geom.sigma_px_y:.3f} piksler."
        )

        vrt_kwargs = {
            "crs": cir.crs,
            "transform": cir.transform,
            "width": cir.width,
            "height": cir.height,
            "resampling": Resampling.bilinear,
        }

        with (
            rasterio.open(dtm_vrt_path) as dtm_src,
            rasterio.open(dom_vrt_path) as dom_src,
        ):
            with (
                WarpedVRT(dtm_src, **vrt_kwargs) as dtm,
                WarpedVRT(dom_src, **vrt_kwargs) as dom,
            ):
                total_windows = math.ceil(cir.width / config.block_size) * math.ceil(
                    cir.height / config.block_size
                )

                print(
                    f"[STACK] Prosesserer {total_windows:,} vinduer "
                    f"med blokkstørrelse {config.block_size:,} piksler."
                )

                process_started = time.perf_counter()

                with rasterio.open(tmp_gtiff, "w", **profile) as dst:
                    for band_index, description in enumerate(
                        BAND_DESCRIPTIONS, start=1
                    ):
                        dst.set_band_description(band_index, description)

                    dst.update_tags(
                        SOFTWARE="cir_topo_stack_gpu.py",
                        OUTPUT_BANDS=", ".join(BAND_DESCRIPTIONS),
                        CIR_INPUT=str(cir_path),
                        CIR_BAND_MAPPING=(
                            f"NIR={config.nir_band}, "
                            f"Red={config.red_band}, "
                            f"Green={config.green_band}"
                        ),
                        DTM_VRT=str(dtm_vrt_path),
                        DOM_VRT=str(dom_vrt_path),
                        TPI_WINDOW_PIXELS=", ".join(
                            f"{size}x{size}" for size in geom.tpi_window_sizes
                        ),
                        GAUSSIAN_SIGMA_M=str(config.gauss_sigma_m),
                        SLOPE_UNITS="degrees",
                        PROCESSING_BACKEND=config.processing_backend.upper(),
                        OUTPUT_COMPRESSION=config.output_compression,
                        OUTPUT_DTYPE=config.output_dtype.upper(),
                        UINT8_NODATA_VALUE=str(config.uint8_nodata),
                        UINT8_SCALE_RANGES="; ".join(
                            f"{band}={bounds[0]}:{bounds[1]}"
                            for band, bounds in config.uint8_scale_ranges.items()
                        ),
                    )

                    windows = block_windows(cir.width, cir.height, config.block_size)

                    for window_index, core in enumerate(
                        tqdm(
                            windows,
                            total=total_windows,
                            desc="Prosesserer vinduer",
                            unit="vindu",
                        ),
                        start=1,
                    ):
                        ext = expanded_window(core, geom.halo_px, cir.width, cir.height)
                        rows, cols = crop_slices(core, ext)

                        nir_np = cir.read(
                            config.nir_band,
                            window=ext,
                            out_dtype="float32",
                            masked=False,
                        )
                        red_np = cir.read(
                            config.red_band,
                            window=ext,
                            out_dtype="float32",
                            masked=False,
                        )
                        green_np = cir.read(
                            config.green_band,
                            window=ext,
                            out_dtype="float32",
                            masked=False,
                        )
                        dtm_np = dtm.read(
                            1, window=ext, out_dtype="float32", masked=False
                        )
                        dom_np = dom.read(
                            1, window=ext, out_dtype="float32", masked=False
                        )

                        if config.processing_backend == "gpu":
                            layers = gpu_compute_stack(
                                nir_np=nir_np,
                                red_np=red_np,
                                green_np=green_np,
                                dtm_np=dtm_np,
                                dom_np=dom_np,
                                cir_nodata=cir.nodata,
                                dtm_nodata=dtm.nodata,
                                dom_nodata=dom.nodata,
                                geom=geom,
                                output_nodata=config.float32_nodata,
                            )
                        else:
                            layers = cpu_compute_stack(
                                nir_np=nir_np,
                                red_np=red_np,
                                green_np=green_np,
                                dtm_np=dtm_np,
                                dom_np=dom_np,
                                cir_nodata=cir.nodata,
                                dtm_nodata=dtm.nodata,
                                dom_nodata=dom.nodata,
                                geom=geom,
                                output_nodata=config.float32_nodata,
                            )

                        encoded_layers = encode_layers_for_output(layers, config)

                        for band_index, layer in enumerate(encoded_layers, start=1):
                            dst.write(layer[rows, cols], band_index, window=core)

                        del (
                            encoded_layers,
                            layers,
                            nir_np,
                            red_np,
                            green_np,
                            dtm_np,
                            dom_np,
                        )

                        if config.processing_backend == "gpu":
                            cp.get_default_memory_pool().free_all_blocks()

                        if (
                            window_index % config.window_print_every == 0
                            or window_index == total_windows
                        ):
                            elapsed = time.perf_counter() - process_started
                            avg_seconds_per_window = elapsed / max(1, window_index)
                            remaining_seconds = avg_seconds_per_window * max(
                                0, total_windows - window_index
                            )
                            print(
                                f"[STACK] Vindu {window_index:,}/{total_windows:,} ferdig | "
                                f"{elapsed:,.1f} s brukt | "
                                f"estimert resterende ved samme takt: {remaining_seconds:,.1f} s"
                            )

    elapsed = time.perf_counter() - stack_started
    print(f"[STACK] Standard GeoTIFF ferdig ({elapsed:,.1f} s): {tmp_gtiff}")
    return geom


def move_gtiff_to_output(tmp_gtiff: Path, out_path: Path, overwrite: bool) -> None:
    """
    Move the finished GeoTIFF to its final destination.

    On Windows, os.replace() cannot move files across drive letters.
    Example: C:/Temp -> D:/output raises WinError 17.
    We try os.replace() first because it is fast on the same filesystem, then
    fall back to shutil.move() for cross-drive moves.
    """
    print(f"[OUTPUT] Flytter standard GeoTIFF til endelig output: {out_path}")
    started = time.perf_counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output finnes allerede: {out_path}. "
                "Sett OVERWRITE_OUTPUT = True for å overskrive."
            )
        out_path.unlink()

    try:
        os.replace(tmp_gtiff, out_path)
        method = "os.replace"

    except OSError as exc:
        cross_device = (
            getattr(exc, "winerror", None) == 17
            or getattr(exc, "errno", None) == getattr(os, "EXDEV", 18)
            or getattr(exc, "errno", None) == 18
        )

        if not cross_device:
            raise

        print(
            "[OUTPUT] Tempfil og output ligger på ulike diskstasjoner. "
            "Bruker shutil.move i stedet."
        )
        shutil.move(str(tmp_gtiff), str(out_path))
        method = "shutil.move"

    elapsed = time.perf_counter() - started
    print(f"[OUTPUT] Ferdig flytting av GeoTIFF via {method} ({elapsed:,.1f} s).")


def validate_output(out_path: Path) -> None:
    print(f"[CHECK] Validerer output: {out_path}")

    with rasterio.open(out_path) as ds:
        if ds.count != len(BAND_DESCRIPTIONS):
            raise RuntimeError(
                f"Output har {ds.count} bånd, forventet {len(BAND_DESCRIPTIONS)}."
            )

        descriptions = tuple(ds.descriptions)
        if descriptions != BAND_DESCRIPTIONS:
            raise RuntimeError(f"Uventede båndbeskrivelser: {descriptions}")

    print("[CHECK] Output ser riktig ut.")


# =============================================================================
# MAIN
# =============================================================================
def process_area(area: str) -> None:
    area_started = time.perf_counter()
    config = load_config(area)

    print()
    print("#" * 100)
    print(f"START PROCESSING AREA: {area}")
    print("#" * 100)
    print(f"[CONFIG] CIR: {config.cir_path}")
    print(
        f"[CONFIG] CIR bandmapping: "
        f"NIR={config.nir_band}, Red={config.red_band}, Green={config.green_band}"
    )
    print(f"[CONFIG] DTM-mapper: {', '.join(str(path) for path in config.dtm_dirs)}")
    print(f"[CONFIG] DOM-mapper: {', '.join(str(path) for path in config.dom_dirs)}")
    print(f"[CONFIG] Output: {config.out_path}")
    print(
        f"[CONFIG] Outputformat: standard GeoTIFF | "
        f"datatype: {config.output_dtype.upper()} | "
        f"komprimering: {config.output_compression}"
    )
    if config.output_dtype == "uint8":
        print(
            f"[CONFIG] UInt8 nodata: {config.uint8_nodata} | "
            "gyldige skalerte verdier bruker 0–254."
        )
    print(
        f"[CONFIG] Tempmappe: "
        f"{config.tmp_dir if config.tmp_dir is not None else 'system default'}"
    )
    print(f"[CONFIG] Blokkstørrelse: {config.block_size:,}")
    print(f"[CONFIG] GDAL cache: {config.gdal_cache_mb:,} MB")
    print(f"[CONFIG] Scan workers: {config.scan_workers}")
    print(f"[CONFIG] Beregningsbackend: {config.processing_backend.upper()}")
    print(f"[CONFIG] Print hver {config.window_print_every} prosesseringsvinduer.")
    print(
        "[CONFIG] TPI-vinduer: "
        + ", ".join(f"{size} x {size}" for size in config.tpi_window_pixels)
        + " piksler."
    )
    print("#" * 100)

    print("[INIT] Validerer konfigurasjon ...")
    validate_config(config)

    print("[INIT] Validerer CIR-raster ...")
    validate_cir(config)

    config.out_path.parent.mkdir(parents=True, exist_ok=True)

    if config.out_path.exists() and not config.overwrite_output:
        raise FileExistsError(
            f"Output finnes allerede: {config.out_path}. "
            "Sett OVERWRITE_OUTPUT = True for å overskrive."
        )

    with rasterio.open(config.cir_path) as cir:
        cir_bounds = tuple(cir.bounds)
        cir_crs = cir.crs
        print(
            f"[CIR] CRS: {cir_crs} | "
            f"størrelse: {cir.width:,} x {cir.height:,} | "
            f"bounds: {cir_bounds}"
        )

    print("[SCAN] Leser kandidatfiler fra DTM-mapper ...")
    dtm_candidates = iter_raster_files(config.dtm_dirs)
    print(f"[SCAN] Fant {len(dtm_candidates):,} DTM-kandidatfiler.")

    print("[SCAN] Leser kandidatfiler fra DOM-mapper ...")
    dom_candidates = iter_raster_files(config.dom_dirs)
    print(f"[SCAN] Fant {len(dom_candidates):,} DOM-kandidatfiler.")

    dtm_tiles = find_overlapping_tiles(
        dtm_candidates,
        cir_crs,
        cir_bounds,
        config.scan_workers,
        "DTM",
        config.tile_scan_print_every,
    )
    dom_tiles = find_overlapping_tiles(
        dom_candidates,
        cir_crs,
        cir_bounds,
        config.scan_workers,
        "DOM",
        config.tile_scan_print_every,
    )

    print(f"[SCAN] Bruker {len(dtm_tiles):,} DTM-fliser som overlapper CIR.")
    print(f"[SCAN] Bruker {len(dom_tiles):,} DOM-fliser som overlapper CIR.")

    temp_parent = str(config.tmp_dir) if config.tmp_dir is not None else None

    if config.tmp_dir is not None:
        config.tmp_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f"cir_topo_stack_{area}_", dir=temp_parent
    ) as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        print(f"[TEMP] Midlertidig arbeidsmappe: {tmpdir}")

        dtm_vrt = build_vrt(dtm_tiles, tmpdir / "dtm_overlap.vrt")
        dom_vrt = build_vrt(dom_tiles, tmpdir / "dom_overlap.vrt")
        tmp_gtiff = tmpdir / "stack_intermediate.tif"

        geom = write_stack_gtiff(
            config.cir_path,
            dtm_vrt,
            dom_vrt,
            tmp_gtiff,
            config,
        )

        move_gtiff_to_output(
            tmp_gtiff,
            config.out_path,
            config.overwrite_output,
        )

        validate_output(config.out_path)

    total_elapsed = time.perf_counter() - area_started

    print("=" * 100)
    print(f"[AREA DONE] {area} ferdig.")
    print(f"[AREA DONE] Output: {config.out_path}")
    print(f"[AREA DONE] Kjøretid: {total_elapsed:,.1f} s")
    print("[AREA DONE] Båndrekkefølge:")
    for index, band_name in enumerate(BAND_DESCRIPTIONS, start=1):
        print(f"  {index}: {band_name}")
    print(
        "[AREA DONE] TPI-vinduer: "
        + ", ".join(f"{size} x {size}" for size in geom.tpi_window_sizes)
        + " piksler"
    )
    print(
        f"[AREA DONE] Gaussian sigma: {config.gauss_sigma_m:g} m "
        f"(~{geom.sigma_px_x:.3f} x {geom.sigma_px_y:.3f} piksler)"
    )
    print("=" * 100)


def main() -> int:
    total_started = time.perf_counter()

    print("=" * 100)
    print("CIR + topografisk rasterstack — flerområdeprosessering")
    print("=" * 100)
    print(f"[BATCH] Områder i kø: {', '.join(AREA_LIST)}")
    print(f"[BATCH] Antall oppføringer: {len(AREA_LIST)}")
    print(f"[BATCH] Beregningsbackend: {PROCESSING_BACKEND.upper()}")
    print(f"[BATCH] Fortsett ved områdefeil: {CONTINUE_ON_AREA_ERROR}")
    print("=" * 100)

    print(
        "[INIT] Konfigurerer GDAL og prosesseringsbackend én gang for hele batchen ..."
    )
    configure_runtime(GDAL_CACHE_MB)
    initialize_processing_backend(PROCESSING_BACKEND.strip().lower())

    successful_areas: list[str] = []
    failed_areas: list[tuple[str, str]] = []

    for area_index, area in enumerate(AREA_LIST, start=1):
        print()
        print("=" * 100)
        print(f"[BATCH] Starter område {area_index}/{len(AREA_LIST)}: {area}")
        print("=" * 100)

        try:
            process_area(area)
            successful_areas.append(area)

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            failed_areas.append((area, error_message))

            print("!" * 100)
            print(f"[AREA ERROR] Område {area} feilet.")
            print(f"[AREA ERROR] {error_message}")
            print("[AREA ERROR] Traceback:")
            traceback.print_exc()
            print("!" * 100)

            if not CONTINUE_ON_AREA_ERROR:
                raise

    total_elapsed = time.perf_counter() - total_started

    print()
    print("=" * 100)
    print("[BATCH DONE] Samlet status")
    print("=" * 100)
    print(f"[BATCH DONE] Vellykkede områder: {len(successful_areas)}/{len(AREA_LIST)}")
    if successful_areas:
        print(f"[BATCH DONE] OK: {', '.join(successful_areas)}")

    print(f"[BATCH DONE] Feilede områder: {len(failed_areas)}/{len(AREA_LIST)}")
    for area, message in failed_areas:
        print(f"[BATCH DONE] FEIL {area}: {message}")

    print(f"[BATCH DONE] Total kjøretid: {total_elapsed:,.1f} s")
    print("=" * 100)

    return 0 if not failed_areas else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as err:
        raise SystemExit("Avbrutt av bruker.") from err
