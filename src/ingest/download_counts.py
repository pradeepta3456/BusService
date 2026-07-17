"""Download the DfT GB Road Traffic Counts datasets.

Two datasets, both key-free, both Open Government Licence v3.0:

  raw_counts - one row per (count point, direction, date, hour). This is the
      dependent data: 5.3M rows, ~1GB uncompressed. Manual counts run a 12-hour
      day (07:00-18:59), so a count point contributes 12 rows per direction per
      counted date.
  aadf - Annual Average Daily Flow per count point per year. Used as a
      link-level context table, not as the target.

Plain `requests` here rather than Spark: two HTTP GETs are I/O-bound and there
is nothing to parallelise.

Run:
    python -m src.ingest.download_counts
"""

from __future__ import annotations

import io
import logging
import sys
import zipfile

import requests

from src.config import load_config, resolve

LOGGER = logging.getLogger("download_counts")

CHUNK = 1 << 20


def fetch_and_extract(url: str, expected_suffix: str = ".csv") -> bool:
    """Download a zip and extract its CSV into data/raw."""
    raw_dir = resolve("raw")

    response = requests.head(url, allow_redirects=True, timeout=60)
    size_mb = int(response.headers.get("Content-Length", 0)) / 1048576
    LOGGER.info("Fetching %s (%.1f MB)", url.rsplit("/", 1)[-1], size_mb)

    buffer = io.BytesIO()
    with requests.get(url, stream=True, timeout=600) as stream:
        stream.raise_for_status()
        for chunk in stream.iter_content(CHUNK):
            buffer.write(chunk)

    with zipfile.ZipFile(buffer) as archive:
        for member in archive.namelist():
            if not member.endswith(expected_suffix):
                continue
            target = raw_dir / member.rsplit("/", 1)[-1]
            if target.exists() and target.stat().st_size > 0:
                LOGGER.info("Already present, skipping: %s", target.name)
                continue
            with archive.open(member) as source, open(target, "wb") as sink:
                while block := source.read(CHUNK):
                    sink.write(block)
            LOGGER.info("Wrote %s (%.1f MB)", target.name, target.stat().st_size / 1048576)
    return True


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stdout
    )
    cfg = load_config()["sources"]
    raw_dir = resolve("raw")

    if (raw_dir / "dft_traffic_counts_raw_counts.csv").exists():
        LOGGER.info("Raw counts already downloaded; skipping.")
    else:
        fetch_and_extract(cfg["raw_counts"])

    if (raw_dir / "dft_traffic_counts_aadf.csv").exists():
        LOGGER.info("AADF already downloaded; skipping.")
    else:
        fetch_and_extract(cfg["aadf"])

    LOGGER.info("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
