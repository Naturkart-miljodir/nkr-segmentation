# nkr-segmentation | Segmentation for ground truth collection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Norwegian Title:** Segmentering for innsamling av bakkesannheter for lavland og fjell.
**Project Owner:** Norwegian Environment Agency (NEA).
**Repository Purpose:** Provide reproducible preprocessing and eCognition rulesets for segmentation used in ground-truth data collection.

---

**Table of Contents:**

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [License](#license)

## Project Overview

**English:** This repository contains the segmentation workflow used to generate object segments for ground-truth collection. The workflow combines a Python preprocessing script that builds a standardized 10-band CIR/topography raster stack and two eCognition rulesets for project setup and segmentation execution.

**Norwegian:** Dette repositoriet inneholder segmenteringsløpet, brukt til å lage objekter for innsamling av bakkesannheter. Arbeidsflyten kombinerer et Python-skript for å bygge en standardisert 10-bånds CIR/topografi-stack og to eCognition-ruleset for oppsett og segmentering.

---

### Key Features

- Builds a 10-band raster stack from CIR, DTM, and DOM using CPU (SciPy) or GPU (CuPy).
- Produces derived layers: TPI20, TPI50, TPI70, CHM, CHM_Gaus5, NDVI, and Slope_Gaus5.
- Includes eCognition setup ruleset for layer aliasing and project preparation.
- Includes eCognition segmentation ruleset with separate Segment A and Segment B parameterization.

### Resulting Products

| Product                     | Description                                                         |
| --------------------------- | ------------------------------------------------------------------- |
| 10-band preprocessing stack | Standard GeoTIFF stack used as segmentation input.                  |
| Setup ruleset               | eCognition setup process for AR5 import and layer alias assignment. |
| Segmentation ruleset        | eCognition segmentation process defining Segment A and Segment B.   |

### Repository Structure

```text
nkr-segmentation/
├── docs/
│   ├── teknisk_dokumentasjon_segmentering_2026.docx
│   └── teknisk_dokumentasjon_segmentering_2026.md
├── rulesets/
│   ├── segment2026_ruleset_setup.dcp
│   └── segment2026_ruleset_segment.dcp
├── scripts/
│   └── cir_topo_stack_gpu.py
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- `uv`
- GDAL and `osgeo.gdal` bindings
- Optional: `cupy` for GPU processing
- eCognition (licensed separately) to run `.dcp` rulesets

### Installation

1. Clone the repository.
2. Create environment file from template.
3. Synchronize dependencies with uv.
4. Ensure eCognition is installed and licensed separately.

Example setup with uv:

```bash
cp .env.example .env
uv sync
```

Optional groups:

```bash
# include GPU dependency
uv sync --group gpu

# include development tooling
uv sync --group dev
```

### Workflow

1. Copy `.env.example` to `.env` and set `BASE_PATH` and `TOPO_PATH`.
2. Verify area and processing settings in `scripts/cir_topo_stack_gpu.py` (for example `AREA_LIST`, `PROCESSING_BACKEND`, and output options).
3. Run preprocessing script to generate the 10-band stack.
4. Open eCognition and run setup ruleset, then segmentation ruleset.

Example:

```bash
uv run python scripts/cir_topo_stack_gpu.py
```

## Documentation

For detailed technical documentation, refer to: [Teknisk_dokumentasjon_segmentering_2026.md](docs/Teknisk_dokumentasjon_segmentering_2026.md)

## License

This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Note on third-party dependencies: This repository does not include eCognition software. eCognition is a separate proprietary dependency and must be licensed independently by the user.

---
