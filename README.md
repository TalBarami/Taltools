# Taltools

Utilities for computer vision, data processing, I/O, and logging.

## Installation

**From GitHub (remote machines):**
```bash
pip install git+https://github.com/TalBarami/Taltools.git
```

**Local development (editable — changes take effect immediately):**
```bash
git clone https://github.com/TalBarami/Taltools.git
cd Taltools
pip install -e .
```

## Updating

**Remote machines:**
```bash
pip install --upgrade git+https://github.com/TalBarami/Taltools.git
```

**Local editable install:**
```bash
git pull  # changes are live immediately, no reinstall needed
```

## Subpackages

- `taltools.cv` — video/image processing, bounding boxes, 3D landmarks
- `taltools.io` — file I/O (JSON, pickle), OmegaConf configuration
- `taltools.data` — pandas DataFrame utilities
- `taltools.logging` — console, file, and Neptune.ai loggers

## Usage

```python
from taltools.cv import get_video_properties
from taltools.io import read_json, write_json
from taltools.data import gaussian_smoothing
from taltools.logging import PrintLogger
```
