"""Download and cache pretrained PAT weights."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from urllib.request import urlopen


DEFAULT_WEIGHTS_URL = (
    "https://www.dropbox.com/scl/fi/ha9b0cj4b3gvcfq4etc6h/"
    "weight_only_encoder_large_90_unsmoothed_mse_all.h5?rlkey="
    "sbu5fd9p56qawnquz4w6stjzr&st=aewhfwq5&dl=1"
)
DEFAULT_WEIGHTS_NAME = "WEIGHTS_encoder_large_90_unsmoothed_mse_all.h5"


def download_weights(weights_path: str | Path | None = None, *, url: str = DEFAULT_WEIGHTS_URL) -> Path:
    """Return local encoder weights, downloading them when absent."""
    path = Path(weights_path) if weights_path is not None else default_weights_path()
    path = path.expanduser()
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".part")
    try:
        with urlopen(url, timeout=60) as response, temporary_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if not temporary_path.exists() or temporary_path.stat().st_size == 0:
            raise RuntimeError("Downloaded weights file is empty.")
        temporary_path.replace(path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download PAT weights from {url!r}.") from error
    return path


def default_weights_path() -> Path:
    """Return the platform-appropriate cache path for the default weights."""
    if sys.platform == "win32":
        cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "pypat" / DEFAULT_WEIGHTS_NAME
