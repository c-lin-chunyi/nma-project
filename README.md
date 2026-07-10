# NMA Dataset Ingestion

The default pipeline downloads the Neuromatch-provided parquet file:

- `allen_visual_behavior_2p_change_detection_familiar_novel_image_sets.parquet`
- Source: `https://ndownloader.figshare.com/files/28470255`
- Destination: `data/raw/`

AllenSDK/NWB ingestion is available as an optional setup because it pulls much larger files and adds heavier dependencies.

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

## Optional AllenSDK Setup

Use this only when you need to build a small Neuromatch-like parquet from raw AllenSDK/NWB data. It is separate from `environment.yml` because AllenSDK pins older scientific Python packages.

```bash
conda env create -f environment-allensdk.yml
conda activate nma-allensdk
nma-build-allen-parquet --validate
```

`--validate` means "build the pilot parquet if needed, then validate its schema and
summary." The default builder is capped at one successful experiment per cre line and 
at most three NWB download/load attempts per
cre line. With the default three cre lines, that means at most nine experiment NWB files
unless you explicitly raise `--max-attempts-per-cre-line`.

If an existing `nma-allensdk` env was created before the `pynwb`/`hdmf` pins were added,
refresh it with:

```bash
python -m pip install "pynwb==2.8.3" "hdmf>=4,<5"
```

The default AllenSDK command writes:

```text
data/processed/allen_visp_sst_vip_slc17a7_pilot.parquet
```

It caches AllenSDK metadata and NWB files under:

```text
data/cache/allensdk/
```

By default it selects one active VISp experiment per cre line for:

- `Sst-IRES-Cre`
- `Vip-IRES-Cre`
- `Slc17a7-IRES2-Cre`

The AllenSDK command shows tqdm progress for experiment-level NWB download/load attempts and per-cell alignment. AllenSDK also emits its own byte-level progress for individual NWB files when it downloads them, for example `behavior_ophys_experiment_...nwb: 254M/254M`. Use `--max-attempts-per-cre-line 1` for a strict one-NWB-per-cre-line run with no fallback, or `--no-progress` for cleaner batch logs around the repo-level progress bars.

## Colab Setup

Open `notebooks/00_ingestion_smoke_test.ipynb` in Colab and run the cells top-to-bottom. The first code cell installs any missing Python dependencies inside the notebook and will use the repo package if the repo is available.

If the notebook is opened standalone, it still downloads, validates, summarizes, and plots the dataset using notebook-local fallback helpers. If the full repo is cloned or mounted, it imports the shared helpers from `src/nma_data_ingestion/`.

Use `notebooks/01_colab_drive_parquet_demo.ipynb` for a standalone Colab demo that
downloads an already-built AllenSDK-derived parquet from a shared Google Drive link.
Set `GDRIVE_URL` in the download cell to a link such as:

```text
https://drive.google.com/file/d/FILE_ID/view?usp=sharing
```

The notebook installs `gdown` when needed and downloads the file to its local
`data/` directory. The Drive file must be shared with anyone who has the link.

For AllenSDK work in Colab, install the heavier optional requirements from the repo root:

```python
!pip install -r requirements-allensdk-colab.txt
!pip install -e .
```

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
nma-build-allen-parquet --validate
nma-build-allen-parquet --targeted-structure VISp --cre-lines Sst-IRES-Cre Vip-IRES-Cre Slc17a7-IRES2-Cre --max-experiments-per-cre-line 1 --validate
nma-build-allen-parquet --max-attempts-per-cre-line 1 --validate
```

The downloader writes to a temporary `.part` file first, then renames it after a successful download. Existing files are reused unless `--overwrite` is passed.

## Expected Dataset Columns

The Neuromatch validator checks for the columns needed by the initial hit/omission analysis:

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

The AllenSDK-derived parquet includes all 31 Neuromatch parquet columns where they can
be derived from AllenSDK. In particular, it fills per-stimulus running speed and pupil
area summaries, plus session/genotype/reporter/driver metadata from the AllenSDK
metadata table and NWB metadata. It also adds Allen-only analysis columns:

- `hit`
- `behavior_outcome`
- `experience_level`

## Tests

```bash
pytest
```

The unit tests use tiny synthetic parquet files and mocked network responses, so they do not require downloading the real dataset.
