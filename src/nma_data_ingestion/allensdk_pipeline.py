"""Build a Neuromatch-like analysis parquet from AllenSDK Visual Behavior data."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DEFAULT_ALLEN_CACHE_DIR = Path("data/cache/allensdk")
DEFAULT_ALLEN_OUTPUT_PATH = Path("data/processed/allen_visp_sst_vip_slc17a7_pilot.parquet")
DEFAULT_TARGETED_STRUCTURE = "VISp"
DEFAULT_CRE_LINES = ("Sst-IRES-Cre", "Vip-IRES-Cre", "Slc17a7-IRES2-Cre")
DEFAULT_TRACE_COLUMN = "dff"
DEFAULT_T_BEFORE = 1.25
DEFAULT_T_AFTER = 1.5
DEFAULT_OUTPUT_SAMPLING_RATE = 31.0
DEFAULT_MAX_ATTEMPTS_PER_CRE_LINE = 3

NEUROMATCH_COMPATIBLE_COLUMNS = (
    "stimulus_presentations_id",
    "cell_specimen_id",
    "trace",
    "trace_timestamps",
    "mean_response",
    "baseline_response",
    "image_name",
    "image_index",
    "is_change",
    "omitted",
    "mean_running_speed",
    "mean_pupil_area",
    "response_latency",
    "rewarded",
    "ophys_experiment_id",
    "imaging_depth",
    "targeted_structure",
    "cre_line",
    "session_type",
    "session_number",
    "mouse_id",
    "ophys_session_id",
    "ophys_container_id",
    "behavior_session_id",
    "full_genotype",
    "reporter_line",
    "driver_line",
    "indicator",
    "sex",
    "age_in_days",
    "exposure_level",
)

ALLEN_ONLY_COLUMNS = (
    "hit",
    "behavior_outcome",
    "experience_level",
)

ALLEN_PARQUET_REQUIRED_COLUMNS = NEUROMATCH_COMPATIBLE_COLUMNS + ALLEN_ONLY_COLUMNS


@dataclass(frozen=True)
class AllenParquetSummary:
    """Compact validation summary for AllenSDK-derived analysis parquet files."""

    path: Path | None
    rows: int
    columns: int
    cells: int
    stimuli: int
    sessions: int
    experiments: int
    cre_lines: tuple[str, ...]
    mouse_count: int


def get_allensdk_cache(cache_dir: str | Path = DEFAULT_ALLEN_CACHE_DIR) -> Any:
    """Create a VisualBehaviorOphysProjectCache with lazy AllenSDK import."""

    try:
        from allensdk.brain_observatory.behavior.behavior_project_cache import (
            VisualBehaviorOphysProjectCache,
        )
    except ImportError as exc:
        raise ImportError(
            "AllenSDK is not installed. Create the optional env with "
            "`conda env create -f environment-allensdk.yml`."
        ) from exc

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)


def load_allensdk_metadata(cache: Any) -> dict[str, pd.DataFrame]:
    """Load AllenSDK ophys metadata tables used by the selector."""

    return {
        "ophys_sessions": cache.get_ophys_session_table(),
        "ophys_experiments": cache.get_ophys_experiment_table(),
        "ophys_cells": cache.get_ophys_cells_table(),
    }


def select_allensdk_experiments(
    experiment_table: pd.DataFrame,
    *,
    targeted_structure: str = DEFAULT_TARGETED_STRUCTURE,
    cre_lines: Sequence[str] = DEFAULT_CRE_LINES,
    max_experiments_per_cre_line: int = 1,
) -> pd.DataFrame:
    """Select deterministic candidate experiments by structure and cre line."""

    if max_experiments_per_cre_line < 1:
        raise ValueError("max_experiments_per_cre_line must be >= 1.")

    candidates = select_allensdk_experiment_candidates(
        experiment_table,
        targeted_structure=targeted_structure,
        cre_lines=cre_lines,
    )

    return (
        candidates.groupby("cre_line", group_keys=False)
        .head(max_experiments_per_cre_line)
        .reset_index(drop=True)
    )


def select_allensdk_experiment_candidates(
    experiment_table: pd.DataFrame,
    *,
    targeted_structure: str = DEFAULT_TARGETED_STRUCTURE,
    cre_lines: Sequence[str] = DEFAULT_CRE_LINES,
) -> pd.DataFrame:
    """Return sorted active-behavior candidate experiments for fallback selection."""

    table = _normalize_index(experiment_table, "ophys_experiment_id")
    required = {"ophys_experiment_id", "cre_line", "targeted_structure"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Experiment table is missing required column(s): {', '.join(missing)}")

    filtered = table[
        (table["targeted_structure"] == targeted_structure)
        & (table["cre_line"].isin(tuple(cre_lines)))
    ].copy()

    if "passive" in filtered.columns:
        filtered = filtered[~filtered["passive"].fillna(False).astype(bool)]
    elif "behavior_type" in filtered.columns:
        filtered = filtered[filtered["behavior_type"].fillna("").astype(str).str.lower() == "active_behavior"]
    elif "session_type" in filtered.columns:
        filtered = filtered[~filtered["session_type"].fillna("").astype(str).str.contains("passive", case=False)]

    sort_columns = [
        column
        for column in (
            "cre_line",
            "mouse_id",
            "ophys_session_id",
            "ophys_container_id",
            "ophys_experiment_id",
        )
        if column in filtered.columns
    ]
    filtered = filtered.sort_values(sort_columns)

    return filtered.reset_index(drop=True)


def build_allensdk_analysis_parquet(
    *,
    output_path: str | Path = DEFAULT_ALLEN_OUTPUT_PATH,
    cache_dir: str | Path = DEFAULT_ALLEN_CACHE_DIR,
    targeted_structure: str = DEFAULT_TARGETED_STRUCTURE,
    cre_lines: Sequence[str] = DEFAULT_CRE_LINES,
    max_experiments_per_cre_line: int = 1,
    max_attempts_per_cre_line: int = DEFAULT_MAX_ATTEMPTS_PER_CRE_LINE,
    trace_column: str = DEFAULT_TRACE_COLUMN,
    t_before: float = DEFAULT_T_BEFORE,
    t_after: float = DEFAULT_T_AFTER,
    output_sampling_rate: float = DEFAULT_OUTPUT_SAMPLING_RATE,
    overwrite: bool = False,
    validate: bool = False,
    show_progress: bool = True,
) -> Path:
    """Download selected AllenSDK experiments and write a derived analysis parquet."""

    if max_attempts_per_cre_line < max_experiments_per_cre_line:
        raise ValueError(
            "max_attempts_per_cre_line must be >= max_experiments_per_cre_line."
        )

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        if validate:
            validate_allen_analysis_parquet(output_path)
        return output_path

    cache = get_allensdk_cache(cache_dir)
    metadata = load_allensdk_metadata(cache)
    candidates = select_allensdk_experiment_candidates(
        metadata["ophys_experiments"],
        targeted_structure=targeted_structure,
        cre_lines=cre_lines,
    )

    if candidates.empty:
        raise ValueError("No AllenSDK experiments matched the requested selector.")

    candidates_by_cre = _selected_by_cre_line(candidates)
    output_parts: list[pd.DataFrame] = []
    successful_counts = {cre_line: 0 for cre_line in cre_lines}

    attempt_failures: list[tuple[str, int, str]] = []
    with tqdm(
        desc="AllenSDK NWB attempts",
        unit="attempt",
        bar_format="{desc}: {n_fmt} attempts [{elapsed}, {rate_fmt}{postfix}]",
        disable=not show_progress,
    ) as experiment_progress:
        for cre_line in cre_lines:
            attempts_for_cre_line = 0
            for _, row in candidates_by_cre.get(cre_line, []):
                if successful_counts[cre_line] >= max_experiments_per_cre_line:
                    break
                if attempts_for_cre_line >= max_attempts_per_cre_line:
                    break

                experiment_id = int(row["ophys_experiment_id"])
                attempts_for_cre_line += 1
                experiment_progress.set_postfix_str(f"{cre_line} {experiment_id}")

                try:
                    # AllenSDK downloads and caches the NWB here when it is not already local.
                    experiment = cache.get_behavior_ophys_experiment(experiment_id)
                    stimulus_table = get_change_detection_stimulus_table(experiment)
                    if stimulus_table.empty:
                        attempt_failures.append(
                            (cre_line, experiment_id, "no change-detection stimulus rows")
                        )
                        continue

                    part = build_analysis_dataframe_for_experiment(
                        experiment,
                        stimulus_table=stimulus_table,
                        experiment_metadata=row,
                        trace_column=trace_column,
                        t_before=t_before,
                        t_after=t_after,
                        output_sampling_rate=output_sampling_rate,
                        show_progress=show_progress,
                    )
                    if part.empty:
                        attempt_failures.append((cre_line, experiment_id, "empty derived output"))
                        continue

                    output_parts.append(part)
                    successful_counts[cre_line] += 1
                except Exception as exc:
                    if _is_nwb_dependency_error(exc):
                        raise RuntimeError(
                            "AllenSDK downloaded the NWB but could not read it because the "
                            "installed pynwb/hdmf versions are incompatible. Rebuild the "
                            "optional environment from environment-allensdk.yml, or run "
                            "`python -m pip install 'pynwb==2.8.3' 'hdmf>=4,<5'` "
                            "inside the nma-allensdk environment."
                        ) from exc
                    message = _brief_exception_message(exc)
                    attempt_failures.append((cre_line, experiment_id, message))
                    if show_progress:
                        experiment_progress.write(
                            f"Skipping AllenSDK experiment {experiment_id} ({cre_line}): {message}"
                        )
                finally:
                    experiment_progress.update(1)

    built_cre_lines = set(pd.concat(output_parts)["cre_line"].unique()) if output_parts else set()
    missing_cre_lines = sorted(set(cre_lines) - built_cre_lines)
    if missing_cre_lines:
        message = "Could not build usable AllenSDK output for cre line(s): " + ", ".join(
            missing_cre_lines
        )
        failure_summary = _format_attempt_failures(attempt_failures)
        if failure_summary:
            message = f"{message}. Attempted experiment failures: {failure_summary}"
        raise ValueError(message)

    output = pd.concat(output_parts, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    if validate:
        validate_allen_analysis_parquet(output_path)

    return output_path


def get_change_detection_stimulus_table(experiment: Any) -> pd.DataFrame:
    """Return the change-detection stimulus rows with stable presentation ids."""

    stimulus_table = experiment.stimulus_presentations.copy()
    if "stimulus_block_name" in stimulus_table.columns:
        mask = stimulus_table["stimulus_block_name"].fillna("").astype(str).str.contains(
            "change_detection",
            case=False,
        )
        stimulus_table = stimulus_table[mask].copy()

    if stimulus_table.empty:
        return stimulus_table

    return _normalize_index(stimulus_table, "stimulus_presentations_id")


def build_analysis_dataframe_for_experiment(
    experiment: Any,
    *,
    stimulus_table: pd.DataFrame | None = None,
    experiment_metadata: pd.Series | dict[str, Any] | None = None,
    trace_column: str = DEFAULT_TRACE_COLUMN,
    t_before: float = DEFAULT_T_BEFORE,
    t_after: float = DEFAULT_T_AFTER,
    output_sampling_rate: float = DEFAULT_OUTPUT_SAMPLING_RATE,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build a Neuromatch-like dataframe for one AllenSDK ophys experiment."""

    try:
        import brain_observatory_utilities.datasets.optical_physiology.data_formatting as ophys_formatting
        import brain_observatory_utilities.utilities.general_utilities as utilities
    except ImportError as exc:
        raise ImportError(
            "brain-observatory-utilities is not installed. Use environment-allensdk.yml."
        ) from exc

    stimulus_table = (
        get_change_detection_stimulus_table(experiment)
        if stimulus_table is None
        else _normalize_index(stimulus_table, "stimulus_presentations_id")
    )
    if stimulus_table.empty:
        return pd.DataFrame(columns=ALLEN_PARQUET_REQUIRED_COLUMNS)
    stimulus_table = _add_stimulus_behavior_summaries(stimulus_table, experiment)

    neural_data = ophys_formatting.build_tidy_cell_df(experiment)
    cell_ids = sorted(pd.Series(neural_data["cell_specimen_id"]).dropna().unique())

    rows = []
    event_times = stimulus_table["start_time"].to_numpy()
    event_ids = stimulus_table["stimulus_presentations_id"].to_numpy()
    metadata = _combined_metadata(experiment.metadata, experiment_metadata)

    experiment_id = metadata.get("ophys_experiment_id", "unknown")
    for cell_id in tqdm(
        cell_ids,
        desc=f"Align cells {experiment_id}",
        unit="cell",
        leave=False,
        disable=not show_progress,
    ):
        cell_data = neural_data.query("cell_specimen_id == @cell_id")
        etr = utilities.event_triggered_response(
            data=cell_data,
            t="timestamps",
            y=trace_column,
            event_times=event_times,
            t_before=t_before,
            t_after=t_after,
            output_sampling_rate=output_sampling_rate,
        )
        rows.extend(_event_triggered_rows_for_cell(etr, cell_id, event_ids, trace_column))

    traces = pd.DataFrame(rows)
    if traces.empty:
        return pd.DataFrame(columns=ALLEN_PARQUET_REQUIRED_COLUMNS)

    output = traces.merge(stimulus_table, on="stimulus_presentations_id", how="left")
    output = _add_trial_outcomes(output, getattr(experiment, "trials", None))
    output = _add_experiment_metadata(output, metadata)
    return _coerce_allen_output_schema(output)


def validate_allen_analysis_parquet(data_or_path: pd.DataFrame | str | Path) -> AllenParquetSummary:
    """Validate AllenSDK-derived parquet columns and return a compact summary."""

    path: Path | None
    if isinstance(data_or_path, pd.DataFrame):
        path = None
        data = data_or_path
    else:
        path = Path(data_or_path)
        data = pd.read_parquet(path)

    missing = sorted(set(ALLEN_PARQUET_REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"AllenSDK parquet is missing required column(s): {', '.join(missing)}")
    if data.empty:
        raise ValueError("AllenSDK parquet is empty.")

    return AllenParquetSummary(
        path=path,
        rows=len(data),
        columns=len(data.columns),
        cells=int(data["cell_specimen_id"].nunique(dropna=True)),
        stimuli=int(data["stimulus_presentations_id"].nunique(dropna=True)),
        sessions=int(data["ophys_session_id"].nunique(dropna=True)),
        experiments=int(data["ophys_experiment_id"].nunique(dropna=True)),
        cre_lines=tuple(sorted(str(value) for value in data["cre_line"].dropna().unique())),
        mouse_count=int(data["mouse_id"].nunique(dropna=True)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Neuromatch-like parquet from selected AllenSDK experiments."
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_ALLEN_OUTPUT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_ALLEN_CACHE_DIR)
    parser.add_argument("--targeted-structure", default=DEFAULT_TARGETED_STRUCTURE)
    parser.add_argument("--cre-lines", nargs="+", default=list(DEFAULT_CRE_LINES))
    parser.add_argument("--max-experiments-per-cre-line", type=int, default=1)
    parser.add_argument(
        "--max-attempts-per-cre-line",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS_PER_CRE_LINE,
        help=(
            "Maximum NWB experiment downloads/loads to try per cre line before failing. "
            "This bounds fallback behavior and keeps the default pilot small."
        ),
    )
    parser.add_argument("--trace-column", default=DEFAULT_TRACE_COLUMN)
    parser.add_argument("--t-before", type=float, default=DEFAULT_T_BEFORE)
    parser.add_argument("--t-after", type=float, default=DEFAULT_T_AFTER)
    parser.add_argument("--output-sampling-rate", type=float, default=DEFAULT_OUTPUT_SAMPLING_RATE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars for batch logs or tests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = build_allensdk_analysis_parquet(
        output_path=args.output_path,
        cache_dir=args.cache_dir,
        targeted_structure=args.targeted_structure,
        cre_lines=args.cre_lines,
        max_experiments_per_cre_line=args.max_experiments_per_cre_line,
        max_attempts_per_cre_line=args.max_attempts_per_cre_line,
        trace_column=args.trace_column,
        t_before=args.t_before,
        t_after=args.t_after,
        output_sampling_rate=args.output_sampling_rate,
        overwrite=args.overwrite,
        validate=args.validate,
        show_progress=not args.no_progress,
    )
    print(f"AllenSDK parquet ready: {output_path}")

    if args.validate:
        summary = validate_allen_analysis_parquet(output_path)
        for key, value in asdict(summary).items():
            print(f"{key}: {value}")

    return 0


def _normalize_index(data: pd.DataFrame, id_column: str) -> pd.DataFrame:
    output = data.copy()
    if id_column not in output.columns:
        output = output.reset_index()
        if id_column not in output.columns and "index" in output.columns:
            output = output.rename(columns={"index": id_column})
    return output


def _combined_metadata(
    nwb_metadata: dict[str, Any],
    table_metadata: pd.Series | dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(nwb_metadata)
    if table_metadata is None:
        return metadata
    if isinstance(table_metadata, pd.Series):
        metadata.update(table_metadata.to_dict())
    else:
        metadata.update(table_metadata)
    return metadata


def _add_stimulus_behavior_summaries(
    stimulus_table: pd.DataFrame,
    experiment: Any,
) -> pd.DataFrame:
    output = stimulus_table.copy()
    start_times = pd.to_numeric(output.get("start_time"), errors="coerce").to_numpy(dtype=float)
    end_times = _stimulus_end_times(output, start_times)

    running_speed = getattr(experiment, "running_speed", None)
    output["mean_running_speed"] = _mean_signal_by_interval(
        running_speed,
        value_column="speed",
        start_times=start_times,
        end_times=end_times,
    )

    eye_tracking = getattr(experiment, "eye_tracking", None)
    output["mean_pupil_area"] = _mean_signal_by_interval(
        eye_tracking,
        value_column="pupil_area",
        start_times=start_times,
        end_times=end_times,
        invalid_column="likely_blink",
    )
    return output


def _stimulus_end_times(stimulus_table: pd.DataFrame, start_times: np.ndarray) -> np.ndarray:
    if "end_time" in stimulus_table.columns:
        end_times = pd.to_numeric(stimulus_table["end_time"], errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
    elif "duration" in stimulus_table.columns:
        durations = pd.to_numeric(stimulus_table["duration"], errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
        end_times = start_times + durations
    else:
        end_times = start_times.copy()

    invalid = ~np.isfinite(end_times)
    end_times[invalid] = start_times[invalid]
    return end_times


def _mean_signal_by_interval(
    signal_table: pd.DataFrame | None,
    *,
    value_column: str,
    start_times: np.ndarray,
    end_times: np.ndarray,
    invalid_column: str | None = None,
) -> np.ndarray:
    means = np.full(len(start_times), np.nan, dtype=float)
    if (
        signal_table is None
        or signal_table.empty
        or "timestamps" not in signal_table.columns
        or value_column not in signal_table.columns
    ):
        return means

    timestamps = pd.to_numeric(signal_table["timestamps"], errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )
    values = pd.to_numeric(signal_table[value_column], errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )
    if invalid_column is not None and invalid_column in signal_table.columns:
        invalid = signal_table[invalid_column].fillna(False).astype(bool).to_numpy()
        values[invalid] = np.nan

    finite_times = np.isfinite(timestamps)
    timestamps = timestamps[finite_times]
    values = values[finite_times]
    if timestamps.size == 0:
        return means

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    values = values[order]

    finite_values = np.isfinite(values)
    cumulative_count = np.concatenate([[0], np.cumsum(finite_values.astype(int))])
    cumulative_sum = np.concatenate([[0.0], np.cumsum(np.where(finite_values, values, 0.0))])

    left = np.searchsorted(timestamps, start_times, side="left")
    right = np.searchsorted(timestamps, end_times, side="right")
    counts = cumulative_count[right] - cumulative_count[left]
    sums = cumulative_sum[right] - cumulative_sum[left]
    valid_intervals = counts > 0
    means[valid_intervals] = sums[valid_intervals] / counts[valid_intervals]
    return means


def _selected_by_cre_line(selected: pd.DataFrame) -> dict[str, list[tuple[int, pd.Series]]]:
    grouped: dict[str, list[tuple[int, pd.Series]]] = {}
    for idx, row in selected.iterrows():
        grouped.setdefault(str(row["cre_line"]), []).append((idx, row))
    return grouped


def _is_nwb_dependency_error(exc: Exception) -> bool:
    message = _brief_exception_message(exc, max_length=2000)
    return (
        "Can't instantiate abstract class NWBFile" in message
        and "external_resources" in message
    )


def _brief_exception_message(exc: Exception, *, max_length: int = 220) -> str:
    message = " ".join(f"{type(exc).__name__}: {exc}".split())
    if len(message) <= max_length:
        return message
    return f"{message[: max_length - 3]}..."


def _format_attempt_failures(
    failures: Sequence[tuple[str, int, str]],
    *,
    max_items: int = 6,
) -> str:
    if not failures:
        return ""
    formatted = [
        f"{cre_line} {experiment_id}: {message}"
        for cre_line, experiment_id, message in failures[:max_items]
    ]
    remaining = len(failures) - len(formatted)
    if remaining > 0:
        formatted.append(f"... {remaining} more")
    return "; ".join(formatted)


def _event_triggered_rows_for_cell(
    etr: pd.DataFrame,
    cell_id: int,
    event_ids: np.ndarray,
    value_column_hint: str = DEFAULT_TRACE_COLUMN,
) -> list[dict[str, Any]]:
    event_column = "event_number"
    if event_column not in etr.columns:
        raise ValueError("Event-triggered response output is missing `event_number`.")

    time_column = "time" if "time" in etr.columns else "timestamps"
    value_column = _first_existing_column(
        etr,
        (value_column_hint, "dff", "filtered_events", "events", "response"),
    )
    if time_column not in etr.columns or value_column is None:
        raise ValueError("Event-triggered response output is missing time or response columns.")

    rows: list[dict[str, Any]] = []
    for event_number, event_df in etr.groupby(event_column, sort=True):
        event_idx = int(event_number)
        if event_idx >= len(event_ids):
            continue
        trace_timestamps = event_df[time_column].to_numpy(dtype=float)
        trace = event_df[value_column].to_numpy(dtype=float)
        baseline_mask = trace_timestamps < 0
        response_mask = (trace_timestamps >= 0) & (trace_timestamps <= 0.5)
        rows.append(
            {
                "cell_specimen_id": int(cell_id),
                "stimulus_presentations_id": int(event_ids[event_idx]),
                "trace": trace,
                "trace_timestamps": trace_timestamps,
                "mean_response": _nanmean(trace[response_mask]),
                "baseline_response": _nanmean(trace[baseline_mask]),
            }
        )
    return rows


def _add_trial_outcomes(output: pd.DataFrame, trials: pd.DataFrame | None) -> pd.DataFrame:
    output = output.copy()
    output["hit"] = False
    output["behavior_outcome"] = pd.NA

    if (
        trials is not None
        and not trials.empty
        and "change_time" in trials.columns
        and "start_time" in output.columns
    ):
        trial_table = trials.copy()
        outcome_columns = [
            column
            for column in (
                "hit",
                "miss",
                "false_alarm",
                "correct_reject",
                "response_latency",
                "rewarded",
            )
            if column in trial_table.columns
        ]
        trial_table = trial_table[["change_time", *outcome_columns]].copy()
        trial_table["change_time"] = pd.to_numeric(trial_table["change_time"], errors="coerce")
        trial_table = trial_table.dropna(subset=["change_time"])

        output["_row_order"] = np.arange(len(output))
        output["start_time"] = pd.to_numeric(output["start_time"], errors="coerce")
        mergeable_output = output.dropna(subset=["start_time"])
        unmergeable_output = output[output["start_time"].isna()]

        if trial_table.empty or mergeable_output.empty:
            return output.drop(columns=["_row_order"])

        merged_output = pd.merge_asof(
            mergeable_output.sort_values("start_time"),
            trial_table.sort_values("change_time"),
            left_on="start_time",
            right_on="change_time",
            direction="nearest",
            tolerance=0.75,
            suffixes=("", "_trial"),
        )
        if not unmergeable_output.empty:
            output = pd.concat([merged_output, unmergeable_output], ignore_index=True)
            output = output.sort_values("_row_order")
        else:
            output = merged_output

        if "hit_trial" in output.columns:
            output["hit"] = output["hit_trial"].fillna(False).astype(bool)
        elif "hit_y" in output.columns:
            output["hit"] = output["hit_y"].fillna(False).astype(bool)

        output["behavior_outcome"] = output.apply(_behavior_outcome, axis=1)
        if "response_latency_trial" in output.columns:
            if "response_latency" in output.columns:
                output["response_latency"] = output["response_latency"].combine_first(
                    output["response_latency_trial"]
                )
            else:
                output["response_latency"] = output["response_latency_trial"]
        if "rewarded_trial" in output.columns:
            if "rewarded" in output.columns:
                output["rewarded"] = output["rewarded"].combine_first(output["rewarded_trial"])
            else:
                output["rewarded"] = output["rewarded_trial"]

        output = output.drop(columns=["_row_order"])

    return output


def _behavior_outcome(row: pd.Series) -> str | pd.NA:
    for column, label in (
        ("hit", "hit"),
        ("miss", "miss"),
        ("false_alarm", "false_alarm"),
        ("correct_reject", "correct_reject"),
    ):
        if column in row and _is_true(row.get(column, False)):
            return label
        trial_column = f"{column}_trial"
        if trial_column in row and _is_true(row.get(trial_column, False)):
            return label
    return pd.NA


def _add_experiment_metadata(output: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    output = output.copy()
    metadata_map = {
        "cre_line": "cre_line",
        "mouse_id": "mouse_id",
        "sex": "sex",
        "age_in_days": "age_in_days",
        "ophys_session_id": "ophys_session_id",
        "ophys_experiment_id": "ophys_experiment_id",
        "ophys_container_id": "ophys_container_id",
        "targeted_structure": "targeted_structure",
        "imaging_depth": "imaging_depth",
        "session_type": "session_type",
        "session_number": "session_number",
        "behavior_session_id": "behavior_session_id",
        "full_genotype": "full_genotype",
        "reporter_line": "reporter_line",
        "driver_line": "driver_line",
        "indicator": "indicator",
        "exposure_level": "exposure_level",
        "experience_level": "experience_level",
    }
    for output_column, metadata_key in metadata_map.items():
        if output_column not in output.columns or output[output_column].isna().all():
            _set_constant_column(output, output_column, metadata.get(metadata_key, pd.NA))

    if output["session_number"].isna().all():
        _set_constant_column(
            output,
            "session_number",
            _session_number_from_session_type(metadata.get("session_type", pd.NA)),
        )
    if output["exposure_level"].isna().all():
        _set_constant_column(
            output,
            "exposure_level",
            _exposure_level_from_experience_level(metadata.get("experience_level", pd.NA)),
        )
    return output


def _set_constant_column(output: pd.DataFrame, column: str, value: Any) -> None:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, dict)):
        output[column] = pd.Series([value] * len(output), index=output.index)
    else:
        output[column] = value


def _session_number_from_session_type(session_type: Any) -> float:
    if pd.isna(session_type):
        return float("nan")
    match = re.search(r"OPHYS_(\d+)", str(session_type))
    if not match:
        return float("nan")
    return float(match.group(1))


def _exposure_level_from_experience_level(experience_level: Any) -> Any:
    if pd.isna(experience_level):
        return pd.NA
    value = str(experience_level).strip().lower()
    if value.startswith("familiar"):
        return "familiar"
    if value.startswith("novel"):
        return "novel"
    return value or pd.NA


def _coerce_allen_output_schema(output: pd.DataFrame) -> pd.DataFrame:
    output = output.copy()
    output = _rename_first_existing(
        output,
        {
            "image_name": ("image_name", "image_name_x", "image_name_y"),
            "image_index": ("image_index", "image_index_x", "image_index_y"),
            "is_change": ("is_change", "is_change_x", "is_change_y"),
            "omitted": ("omitted", "omitted_x", "omitted_y"),
            "rewarded": ("rewarded", "rewarded_x", "rewarded_y"),
            "response_latency": (
                "response_latency",
                "response_latency_x",
                "response_latency_y",
            ),
        },
    )

    defaults: dict[str, Any] = {
        "image_name": pd.NA,
        "image_index": pd.NA,
        "is_change": False,
        "omitted": False,
        "mean_running_speed": np.nan,
        "mean_pupil_area": np.nan,
        "rewarded": False,
        "hit": False,
        "behavior_outcome": pd.NA,
        "response_latency": np.nan,
    }
    for column, default in defaults.items():
        if column not in output.columns:
            output[column] = default

    for column in ALLEN_PARQUET_REQUIRED_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA

    return output.loc[:, ALLEN_PARQUET_REQUIRED_COLUMNS]


def _rename_first_existing(output: pd.DataFrame, candidates: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    output = output.copy()
    for target, names in candidates.items():
        if target in output.columns:
            continue
        for name in names:
            if name in output.columns:
                output[target] = output[name]
                break
    return output


def _first_existing_column(data: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in data.columns:
            return candidate
    return None


def _is_true(value: Any) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _nanmean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


if __name__ == "__main__":
    raise SystemExit(main())
