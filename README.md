# NMA Dataset Ingestion

The default pipeline downloads the Neuromatch-provided parquet file:

- `allen_visual_behavior_2p_change_detection_familiar_novel_image_sets.parquet`
- Source: `https://ndownloader.figshare.com/files/28470255`
- Destination: `data/raw/`

Full AllenSDK/NWB ingestion is currently not part of the default setup because it pulls much larger files and adds heavier dependencies. We will add it later after the whole workflow is stable.

## Repo Layout

```text
notebooks/                  Notebooks
src/nma_data_ingestion/     Download, load, validate helpers and CLI
data/raw/                   Downloaded source data, ignored by git
data/processed/             Analysis-ready derived data, ignored by git
data/cache/                 Local caches, ignored by git
tests/                      Unit tests for ingestion behavior
```

## Conda Setup

```bash
conda env create -f environment.yml
conda activate nma-data
pip install -e .
nma-download-data --dataset neuromatch --validate
```

Then open `notebooks/00_ingestion_smoke_test.ipynb` and run the cells.

## Colab Setup

In Colab:

```python
!git clone <repo-url> nma-project
%cd nma-project
!pip install -r requirements-colab.txt
!pip install -e .
!nma-download-data --dataset neuromatch --validate
```

If the repo is already uploaded or mounted in Colab, start from `%cd nma-project`.

## Python Usage

```python
from nma_data_ingestion import (
    download_neuromatch_dataset,
    load_neuromatch_dataset,
    validate_neuromatch_dataset,
)

path = download_neuromatch_dataset(validate=True)
data = load_neuromatch_dataset(path)
summary = validate_neuromatch_dataset(data)
print(summary)
```

## CLI

```bash
nma-download-data --dataset neuromatch --validate
nma-download-data --dataset neuromatch --overwrite --validate
nma-download-data --dataset neuromatch --data-dir /path/to/data/raw --validate
```

The downloader writes to a temporary `.part` file first, then renames it after a successful download. Existing files are reused unless `--overwrite` is passed.

## Expected Dataset Columns

The validator checks for the columns needed by the initial hit/omission analysis:

- `trace`
- `trace_timestamps`
- `image_name`
- `cell_specimen_id`
- `rewarded`
- `omitted`
- `is_change`
- `cre_line`
- `ophys_session_id`
- `ophys_experiment_id`

It also reports row count, session count, experiment count, cre lines, and basic trial-label counts.

## Tests

```bash
pytest
```

The unit tests use tiny synthetic parquet files and mocked network responses, so they do not require downloading the real dataset.
