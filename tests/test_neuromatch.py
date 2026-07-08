from pathlib import Path

import pandas as pd
import pytest

from nma_data_ingestion.neuromatch import (
    NEUROMATCH_DATASET,
    download_neuromatch_dataset,
    validate_neuromatch_dataset,
)


def _tiny_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trace": [[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]],
            "trace_timestamps": [[-1.25, 0.0, 1.5], [-1.25, 0.0, 1.5]],
            "image_name": ["im001", "omitted"],
            "cell_specimen_id": [100, 101],
            "rewarded": [True, False],
            "omitted": [False, True],
            "is_change": [True, False],
            "cre_line": ["Vip-IRES-Cre", "Sst-IRES-Cre"],
            "ophys_session_id": [1, 1],
            "ophys_experiment_id": [10, 11],
        }
    )


def test_validate_neuromatch_dataset_from_dataframe() -> None:
    summary = validate_neuromatch_dataset(_tiny_dataset())

    assert summary.path is None
    assert summary.rows == 2
    assert summary.sessions == 1
    assert summary.experiments == 2
    assert summary.cre_lines == ("Sst-IRES-Cre", "Vip-IRES-Cre")
    assert summary.omitted_counts == {"False": 1, "True": 1}


def test_validate_neuromatch_dataset_from_parquet(tmp_path: Path) -> None:
    dataset_path = tmp_path / NEUROMATCH_DATASET.filename
    _tiny_dataset().to_parquet(dataset_path)

    summary = validate_neuromatch_dataset(dataset_path)

    assert summary.path == dataset_path
    assert summary.rows == 2


def test_validate_neuromatch_dataset_rejects_missing_columns() -> None:
    data = _tiny_dataset().drop(columns=["trace"])

    with pytest.raises(ValueError, match="missing required column"):
        validate_neuromatch_dataset(data)


def test_download_neuromatch_dataset_skips_existing_file(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / NEUROMATCH_DATASET.filename
    dataset_path.write_bytes(b"already here")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("requests.get should not be called when file exists")

    monkeypatch.setattr("nma_data_ingestion.neuromatch.requests.get", fail_if_called)

    result = download_neuromatch_dataset(tmp_path)

    assert result == dataset_path
    assert dataset_path.read_bytes() == b"already here"


def test_download_neuromatch_dataset_overwrites_via_part_file(tmp_path: Path, monkeypatch) -> None:
    class FakeResponse:
        headers = {"content-length": "10"}

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int):
            yield b"new "
            yield b"bytes"

    dataset_path = tmp_path / NEUROMATCH_DATASET.filename
    dataset_path.write_bytes(b"old")

    monkeypatch.setattr(
        "nma_data_ingestion.neuromatch.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )

    result = download_neuromatch_dataset(tmp_path, overwrite=True)

    assert result == dataset_path
    assert dataset_path.read_bytes() == b"new bytes"
    assert not dataset_path.with_suffix(dataset_path.suffix + ".part").exists()
