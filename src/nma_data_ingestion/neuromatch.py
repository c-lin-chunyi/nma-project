"""Download, load, and validate the Neuromatch Allen Visual Behavior parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm.auto import tqdm


DEFAULT_DATA_DIR = Path("data/raw")


@dataclass(frozen=True)
class DatasetSpec:
    """Remote dataset metadata."""

    name: str
    url: str
    filename: str
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSummary:
    """Small validation summary for the ingestion smoke test."""

    path: Path | None
    rows: int
    columns: int
    sessions: int
    experiments: int
    cre_lines: tuple[str, ...]
    omitted_counts: dict[str, int]
    is_change_counts: dict[str, int]
    rewarded_counts: dict[str, int]


NEUROMATCH_DATASET = DatasetSpec(
    name="neuromatch",
    url="https://ndownloader.figshare.com/files/28470255",
    filename="allen_visual_behavior_2p_change_detection_familiar_novel_image_sets.parquet",
    required_columns=(
        "trace",
        "trace_timestamps",
        "image_name",
        "cell_specimen_id",
        "rewarded",
        "omitted",
        "is_change",
        "cre_line",
        "ophys_session_id",
        "ophys_experiment_id",
    ),
)


def download_neuromatch_dataset(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    overwrite: bool = False,
    validate: bool = False,
    chunk_size: int = 1024 * 1024,
    timeout: int = 60,
) -> Path:
    """Download the Neuromatch parquet dataset and return the local path.

    The file is streamed to ``*.part`` and atomically renamed when complete.
    Existing files are reused unless ``overwrite`` is true.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    target_path = data_dir / NEUROMATCH_DATASET.filename
    partial_path = target_path.with_suffix(target_path.suffix + ".part")

    if target_path.exists() and not overwrite:
        if validate:
            validate_neuromatch_dataset(target_path)
        return target_path

    if partial_path.exists():
        partial_path.unlink()

    response = requests.get(
        NEUROMATCH_DATASET.url,
        stream=True,
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with partial_path.open("wb") as file_obj:
        with tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=NEUROMATCH_DATASET.filename,
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                file_obj.write(chunk)
                progress.update(len(chunk))

    partial_path.replace(target_path)

    if validate:
        validate_neuromatch_dataset(target_path)

    return target_path


def load_neuromatch_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Neuromatch parquet dataset into a dataframe."""

    dataset_path = Path(path) if path is not None else DEFAULT_DATA_DIR / NEUROMATCH_DATASET.filename
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run `nma-download-data --validate` first."
        )
    return pd.read_parquet(dataset_path)


def validate_neuromatch_dataset(data_or_path: pd.DataFrame | str | Path) -> DatasetSummary:
    """Validate required columns and return a compact dataset summary."""

    path: Path | None
    if isinstance(data_or_path, pd.DataFrame):
        path = None
        data = data_or_path
    else:
        path = Path(data_or_path)
        data = load_neuromatch_dataset(path)

    missing_columns = sorted(set(NEUROMATCH_DATASET.required_columns) - set(data.columns))
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required column(s): {columns}")

    if data.empty:
        raise ValueError("Dataset is empty.")

    return DatasetSummary(
        path=path,
        rows=len(data),
        columns=len(data.columns),
        sessions=_nunique(data, "ophys_session_id"),
        experiments=_nunique(data, "ophys_experiment_id"),
        cre_lines=_sorted_unique_strings(data["cre_line"]),
        omitted_counts=_value_counts(data["omitted"]),
        is_change_counts=_value_counts(data["is_change"]),
        rewarded_counts=_value_counts(data["rewarded"]),
    )


def _nunique(data: pd.DataFrame, column: str) -> int:
    return int(data[column].nunique(dropna=True))


def _sorted_unique_strings(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted(str(value) for value in pd.Series(values).dropna().unique()))


def _value_counts(values: Iterable[object]) -> dict[str, int]:
    counts = pd.Series(values).value_counts(dropna=False).sort_index()
    return {str(index): int(count) for index, count in counts.items()}
