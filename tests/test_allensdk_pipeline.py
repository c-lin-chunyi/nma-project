from pathlib import Path

import pandas as pd
import pytest

from nma_data_ingestion.allensdk_pipeline import (
    ALLEN_PARQUET_REQUIRED_COLUMNS,
    DEFAULT_CRE_LINES,
    DEFAULT_MAX_ATTEMPTS_PER_CRE_LINE,
    NEUROMATCH_COMPATIBLE_COLUMNS,
    _add_experiment_metadata,
    _add_stimulus_behavior_summaries,
    _add_trial_outcomes,
    build_allensdk_analysis_parquet,
    build_parser,
    main,
    select_allensdk_experiment_candidates,
    select_allensdk_experiments,
    validate_allen_analysis_parquet,
)


def _experiment_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ophys_experiment_id": [30, 10, 20, 40, 50],
            "ophys_session_id": [3, 1, 2, 4, 5],
            "ophys_container_id": [300, 100, 200, 400, 500],
            "mouse_id": [3, 1, 2, 4, 5],
            "cre_line": [
                "Slc17a7-IRES2-Cre",
                "Sst-IRES-Cre",
                "Vip-IRES-Cre",
                "Vip-IRES-Cre",
                "Sst-IRES-Cre",
            ],
            "targeted_structure": ["VISp", "VISp", "VISp", "VISl", "VISp"],
            "session_type": [
                "OPHYS_3_images_A",
                "OPHYS_3_images_A",
                "OPHYS_3_images_A",
                "OPHYS_3_images_A",
                "OPHYS_5_images_A_passive",
            ],
        }
    ).set_index("ophys_experiment_id")


def _allen_output() -> pd.DataFrame:
    data = {
        column: [pd.NA]
        for column in ALLEN_PARQUET_REQUIRED_COLUMNS
    }
    data.update(
        {
            "trace": [[0.1, 0.2, 0.3]],
            "trace_timestamps": [[-1.25, 0.0, 1.5]],
            "mean_response": [0.25],
            "baseline_response": [0.1],
            "image_name": ["im001"],
            "image_index": [1],
            "is_change": [True],
            "omitted": [False],
            "rewarded": [True],
            "hit": [True],
            "behavior_outcome": ["hit"],
            "response_latency": [0.3],
            "cell_specimen_id": [123],
            "stimulus_presentations_id": [456],
            "cre_line": ["Slc17a7-IRES2-Cre"],
            "mouse_id": [999],
            "ophys_session_id": [1],
            "ophys_experiment_id": [10],
        }
    )
    return pd.DataFrame(data)


def test_select_allensdk_experiments_picks_one_active_visp_per_cre_line() -> None:
    selected = select_allensdk_experiments(
        _experiment_table(),
        targeted_structure="VISp",
        cre_lines=DEFAULT_CRE_LINES,
        max_experiments_per_cre_line=1,
    )

    assert selected["cre_line"].tolist() == [
        "Slc17a7-IRES2-Cre",
        "Sst-IRES-Cre",
        "Vip-IRES-Cre",
    ]
    assert selected["ophys_experiment_id"].tolist() == [30, 10, 20]


def test_select_allensdk_experiment_candidates_keeps_fallback_pool() -> None:
    table = _experiment_table().reset_index()
    table.loc[table["ophys_experiment_id"] == 50, "session_type"] = "OPHYS_3_images_A"

    candidates = select_allensdk_experiment_candidates(
        table,
        targeted_structure="VISp",
        cre_lines=["Sst-IRES-Cre"],
    )

    assert candidates["ophys_experiment_id"].tolist() == [10, 50]


def test_select_allensdk_experiments_rejects_bad_limit() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        select_allensdk_experiments(_experiment_table(), max_experiments_per_cre_line=0)


def test_validate_allen_analysis_parquet_from_dataframe() -> None:
    summary = validate_allen_analysis_parquet(_allen_output())

    assert summary.rows == 1
    assert summary.cells == 1
    assert summary.cre_lines == ("Slc17a7-IRES2-Cre",)


def test_validate_allen_analysis_parquet_from_file(tmp_path: Path) -> None:
    output_path = tmp_path / "allen.parquet"
    _allen_output().to_parquet(output_path)

    summary = validate_allen_analysis_parquet(output_path)

    assert summary.path == output_path
    assert summary.experiments == 1


def test_validate_allen_analysis_parquet_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required column"):
        validate_allen_analysis_parquet(_allen_output().drop(columns=["trace"]))


def test_allen_schema_contains_all_neuromatch_columns() -> None:
    assert set(NEUROMATCH_COMPATIBLE_COLUMNS).issubset(ALLEN_PARQUET_REQUIRED_COLUMNS)
    assert len(NEUROMATCH_COMPATIBLE_COLUMNS) == 31


def test_add_stimulus_behavior_summaries_computes_interval_means() -> None:
    class FakeExperiment:
        running_speed = pd.DataFrame(
            {
                "timestamps": [0.0, 0.5, 1.0, 1.5, 2.0],
                "speed": [1.0, 3.0, 5.0, 7.0, 9.0],
            }
        )
        eye_tracking = pd.DataFrame(
            {
                "timestamps": [0.0, 0.5, 1.0, 1.5, 2.0],
                "pupil_area": [10.0, 20.0, 30.0, 40.0, 50.0],
                "likely_blink": [False, False, True, False, False],
            }
        )

    stimulus_table = pd.DataFrame(
        {
            "stimulus_presentations_id": [1, 2],
            "start_time": [0.0, 1.0],
            "end_time": [1.0, 2.0],
        }
    )

    result = _add_stimulus_behavior_summaries(stimulus_table, FakeExperiment())

    assert result["mean_running_speed"].tolist() == [pytest.approx(3.0), pytest.approx(7.0)]
    assert result["mean_pupil_area"].tolist() == [pytest.approx(15.0), pytest.approx(45.0)]


def test_add_experiment_metadata_populates_neuromatch_fields() -> None:
    output = pd.DataFrame({"stimulus_presentations_id": [1, 2]})

    result = _add_experiment_metadata(
        output,
        {
            "behavior_session_id": 123,
            "full_genotype": "Sst/wt;Ai148/wt",
            "reporter_line": "Ai148",
            "driver_line": ["Sst-IRES-Cre"],
            "indicator": "GCaMP6f",
            "session_type": "OPHYS_4_images_A",
            "experience_level": "Novel 1",
        },
    )

    assert result["behavior_session_id"].tolist() == [123, 123]
    assert result["driver_line"].tolist() == [["Sst-IRES-Cre"], ["Sst-IRES-Cre"]]
    assert result["session_number"].tolist() == [4.0, 4.0]
    assert result["exposure_level"].tolist() == ["novel", "novel"]
    assert result["experience_level"].tolist() == ["Novel 1", "Novel 1"]


def test_add_trial_outcomes_ignores_null_change_times() -> None:
    output = pd.DataFrame(
        {
            "stimulus_presentations_id": [1, 2],
            "start_time": [1.0, 2.0],
        }
    )
    trials = pd.DataFrame(
        {
            "change_time": [pd.NA, 2.05],
            "hit": [True, True],
            "response_latency": [0.1, 0.2],
            "rewarded": [True, True],
        }
    )

    result = _add_trial_outcomes(output, trials)

    assert result["hit"].tolist() == [False, True]
    assert pd.isna(result.loc[0, "behavior_outcome"])
    assert result.loc[1, "behavior_outcome"] == "hit"
    assert pd.isna(result.loc[0, "response_latency"])
    assert result.loc[1, "response_latency"] == pytest.approx(0.2)


def test_add_trial_outcomes_keeps_rows_when_all_change_times_are_null() -> None:
    output = pd.DataFrame(
        {
            "stimulus_presentations_id": [1, 2],
            "start_time": [1.0, 2.0],
        }
    )
    trials = pd.DataFrame(
        {
            "change_time": [pd.NA, pd.NA],
            "hit": [True, True],
        }
    )

    result = _add_trial_outcomes(output, trials)

    assert len(result) == 2
    assert result["hit"].tolist() == [False, False]
    assert result["behavior_outcome"].isna().all()


def test_allensdk_cli_parser_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.targeted_structure == "VISp"
    assert args.cre_lines == list(DEFAULT_CRE_LINES)
    assert args.max_experiments_per_cre_line == 1
    assert args.max_attempts_per_cre_line == DEFAULT_MAX_ATTEMPTS_PER_CRE_LINE


def test_build_allensdk_analysis_parquet_falls_back_to_next_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeExperiment:
        def __init__(self, experiment_id: int, cre_line: str) -> None:
            self.experiment_id = experiment_id
            self.cre_line = cre_line

    class FakeCache:
        def get_behavior_ophys_experiment(self, experiment_id: int) -> FakeExperiment:
            return FakeExperiment(experiment_id, "Sst-IRES-Cre")

    table = pd.DataFrame(
        {
            "ophys_experiment_id": [10, 50],
            "ophys_session_id": [1, 5],
            "ophys_container_id": [100, 500],
            "mouse_id": [1, 5],
            "cre_line": ["Sst-IRES-Cre", "Sst-IRES-Cre"],
            "targeted_structure": ["VISp", "VISp"],
            "session_type": ["OPHYS_3_images_A", "OPHYS_3_images_A"],
        }
    )
    output_path = tmp_path / "allen.parquet"

    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.get_allensdk_cache",
        lambda cache_dir: FakeCache(),
    )
    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.load_allensdk_metadata",
        lambda cache: {
            "ophys_sessions": pd.DataFrame(),
            "ophys_experiments": table,
            "ophys_cells": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.get_change_detection_stimulus_table",
        lambda experiment: pd.DataFrame()
        if experiment.experiment_id == 10
        else pd.DataFrame({"stimulus_presentations_id": [1], "start_time": [0.0]}),
    )
    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.build_analysis_dataframe_for_experiment",
        lambda experiment, **kwargs: _allen_output().assign(
            ophys_experiment_id=experiment.experiment_id,
            cre_line=experiment.cre_line,
        ),
    )

    result = build_allensdk_analysis_parquet(
        output_path=output_path,
        cache_dir=tmp_path / "cache",
        cre_lines=["Sst-IRES-Cre"],
        validate=True,
        show_progress=False,
    )

    assert result == output_path
    assert pd.read_parquet(output_path)["ophys_experiment_id"].tolist() == [50]


def test_build_allensdk_analysis_parquet_caps_fallback_attempts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    attempted_ids = []

    class FakeExperiment:
        def __init__(self, experiment_id: int) -> None:
            self.experiment_id = experiment_id

    class FakeCache:
        def get_behavior_ophys_experiment(self, experiment_id: int) -> FakeExperiment:
            attempted_ids.append(experiment_id)
            return FakeExperiment(experiment_id)

    table = pd.DataFrame(
        {
            "ophys_experiment_id": [10, 50, 90],
            "ophys_session_id": [1, 5, 9],
            "ophys_container_id": [100, 500, 900],
            "mouse_id": [1, 5, 9],
            "cre_line": ["Sst-IRES-Cre", "Sst-IRES-Cre", "Sst-IRES-Cre"],
            "targeted_structure": ["VISp", "VISp", "VISp"],
            "session_type": ["OPHYS_3_images_A", "OPHYS_3_images_A", "OPHYS_3_images_A"],
        }
    )

    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.get_allensdk_cache",
        lambda cache_dir: FakeCache(),
    )
    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.load_allensdk_metadata",
        lambda cache: {
            "ophys_sessions": pd.DataFrame(),
            "ophys_experiments": table,
            "ophys_cells": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.get_change_detection_stimulus_table",
        lambda experiment: pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="Could not build usable"):
        build_allensdk_analysis_parquet(
            output_path=tmp_path / "allen.parquet",
            cache_dir=tmp_path / "cache",
            cre_lines=["Sst-IRES-Cre"],
            max_attempts_per_cre_line=2,
            show_progress=False,
        )

    assert attempted_ids == [10, 50]


def test_build_allensdk_analysis_parquet_rejects_impossible_attempt_cap(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="max_attempts_per_cre_line"):
        build_allensdk_analysis_parquet(
            output_path=tmp_path / "allen.parquet",
            max_experiments_per_cre_line=2,
            max_attempts_per_cre_line=1,
        )


def test_build_allensdk_analysis_parquet_falls_back_after_load_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeExperiment:
        def __init__(self, experiment_id: int, cre_line: str) -> None:
            self.experiment_id = experiment_id
            self.cre_line = cre_line

    class FakeCache:
        def get_behavior_ophys_experiment(self, experiment_id: int) -> FakeExperiment:
            if experiment_id == 10:
                raise OSError("corrupt candidate")
            return FakeExperiment(experiment_id, "Sst-IRES-Cre")

    table = pd.DataFrame(
        {
            "ophys_experiment_id": [10, 50],
            "ophys_session_id": [1, 5],
            "ophys_container_id": [100, 500],
            "mouse_id": [1, 5],
            "cre_line": ["Sst-IRES-Cre", "Sst-IRES-Cre"],
            "targeted_structure": ["VISp", "VISp"],
            "session_type": ["OPHYS_3_images_A", "OPHYS_3_images_A"],
        }
    )
    output_path = tmp_path / "allen.parquet"

    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.get_allensdk_cache",
        lambda cache_dir: FakeCache(),
    )
    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.load_allensdk_metadata",
        lambda cache: {
            "ophys_sessions": pd.DataFrame(),
            "ophys_experiments": table,
            "ophys_cells": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.get_change_detection_stimulus_table",
        lambda experiment: pd.DataFrame({"stimulus_presentations_id": [1], "start_time": [0.0]}),
    )
    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.build_analysis_dataframe_for_experiment",
        lambda experiment, **kwargs: _allen_output().assign(
            ophys_experiment_id=experiment.experiment_id,
            cre_line=experiment.cre_line,
        ),
    )

    result = build_allensdk_analysis_parquet(
        output_path=output_path,
        cache_dir=tmp_path / "cache",
        cre_lines=["Sst-IRES-Cre"],
        validate=True,
        show_progress=False,
    )

    assert result == output_path
    assert pd.read_parquet(output_path)["ophys_experiment_id"].tolist() == [50]


def test_build_allensdk_analysis_parquet_stops_on_pynwb_hdmf_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeCache:
        def get_behavior_ophys_experiment(self, experiment_id: int):
            raise TypeError(
                "Can't instantiate abstract class NWBFile with abstract method "
                "external_resources"
            )

    table = pd.DataFrame(
        {
            "ophys_experiment_id": [10, 50],
            "ophys_session_id": [1, 5],
            "ophys_container_id": [100, 500],
            "mouse_id": [1, 5],
            "cre_line": ["Sst-IRES-Cre", "Sst-IRES-Cre"],
            "targeted_structure": ["VISp", "VISp"],
            "session_type": ["OPHYS_3_images_A", "OPHYS_3_images_A"],
        }
    )

    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.get_allensdk_cache",
        lambda cache_dir: FakeCache(),
    )
    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.load_allensdk_metadata",
        lambda cache: {
            "ophys_sessions": pd.DataFrame(),
            "ophys_experiments": table,
            "ophys_cells": pd.DataFrame(),
        },
    )

    with pytest.raises(RuntimeError, match="pynwb/hdmf"):
        build_allensdk_analysis_parquet(
            output_path=tmp_path / "allen.parquet",
            cache_dir=tmp_path / "cache",
            cre_lines=["Sst-IRES-Cre"],
            show_progress=False,
        )


def test_allensdk_cli_smoke_with_mocked_builder(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "allen.parquet"

    def fake_build_allensdk_analysis_parquet(**kwargs):
        _allen_output().to_parquet(kwargs["output_path"], index=False)
        return kwargs["output_path"]

    monkeypatch.setattr(
        "nma_data_ingestion.allensdk_pipeline.build_allensdk_analysis_parquet",
        fake_build_allensdk_analysis_parquet,
    )

    exit_code = main(["--output-path", str(output_path), "--validate"])

    assert exit_code == 0
    assert output_path.exists()
