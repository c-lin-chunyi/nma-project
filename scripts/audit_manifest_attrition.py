"""Audit manifest cohort attrition and pilot parquet trial/cell coverage."""

from pathlib import Path

import pandas as pd


RAW = Path("data/raw/visual_behavior_ophys_manifest")
PILOT = Path("data/processed/allen_visp_sst_vip_slc17a7_pilot.parquet")
OUT = Path("data/processed")


def snapshot(label: str, sessions: pd.DataFrame) -> dict[str, object]:
    counts = sessions.groupby("cre_line")["ophys_session_id"].nunique()
    return {
        "stage": label,
        "all": int(sessions["ophys_session_id"].nunique()),
        "Slc17a7-IRES2-Cre": int(counts.get("Slc17a7-IRES2-Cre", 0)),
        "Sst-IRES-Cre": int(counts.get("Sst-IRES-Cre", 0)),
        "Vip-IRES-Cre": int(counts.get("Vip-IRES-Cre", 0)),
    }


def manifest_attrition() -> pd.DataFrame:
    behavior = pd.read_csv(RAW / "behavior_session_table.csv")
    sessions = pd.read_csv(RAW / "ophys_session_table.csv")
    experiments = pd.read_csv(RAW / "ophys_experiment_table.csv")
    cells = pd.read_csv(RAW / "ophys_cells_table.csv", usecols=["ophys_experiment_id"])
    counts = behavior[
        [
            "behavior_session_id",
            "trial_count",
            "go_trial_count",
            "catch_trial_count",
            "engaged_trial_count",
            "hit_trial_count",
            "miss_trial_count",
        ]
    ]
    cohort = sessions.merge(counts, on="behavior_session_id", how="left", validate="one_to_one")
    ladder = [snapshot("current release: all ophys sessions", cohort)]

    cohort = cohort[cohort["behavior_type"].eq("active_behavior")].copy()
    ladder.append(snapshot("active behavior (remove passive)", cohort))

    qc_session_ids = experiments.loc[
        experiments["ophys_experiment_id"].isin(cells["ophys_experiment_id"]),
        "ophys_session_id",
    ].unique()
    cohort = cohort[cohort["ophys_session_id"].isin(qc_session_ids)].copy()
    ladder.append(snapshot("has published QC cell metadata", cohort))

    cohort = cohort[cohort["hit_trial_count"].gt(0) & cohort["miss_trial_count"].gt(0)].copy()
    ladder.append(snapshot("has >=1 total hit and >=1 total miss", cohort))

    cohort["engaged_fraction"] = cohort["engaged_trial_count"] / cohort["trial_count"]
    cohort = cohort[cohort["engaged_fraction"].ge(0.5)].copy()
    ladder.append(snapshot("DISCARDED legacy session engaged_fraction >= 0.5", cohort))
    return pd.DataFrame(ladder)


def pilot_coverage() -> pd.DataFrame:
    columns = [
        "ophys_experiment_id",
        "stimulus_presentations_id",
        "cell_specimen_id",
        "is_change",
        "behavior_outcome",
        "cre_line",
    ]
    data = pd.read_parquet(PILOT, columns=columns)
    experiments = pd.read_csv(
        RAW / "ophys_experiment_table.csv",
        usecols=["ophys_experiment_id", "ophys_session_id", "behavior_session_id"],
    )
    behavior = pd.read_csv(
        RAW / "behavior_session_table.csv",
        usecols=[
            "behavior_session_id",
            "trial_count",
            "go_trial_count",
            "catch_trial_count",
            "hit_trial_count",
            "miss_trial_count",
            "engaged_trial_count",
        ],
    )
    cells = pd.read_csv(RAW / "ophys_cells_table.csv")
    experiment_cells = cells.groupby("ophys_experiment_id")["cell_specimen_id"].nunique()
    cell_sessions = cells.merge(
        experiments[["ophys_experiment_id", "ophys_session_id"]],
        on="ophys_experiment_id",
        how="inner",
    )
    session_cells = cell_sessions.groupby("ophys_session_id")["cell_specimen_id"].nunique()
    manifest = experiments.merge(behavior, on="behavior_session_id", validate="many_to_one")
    manifest = manifest.set_index("ophys_experiment_id")

    output = []
    for experiment_id, experiment in data.groupby("ophys_experiment_id"):
        changes = experiment[
            experiment["is_change"].fillna(False).astype(bool)
            & experiment["behavior_outcome"].isin(["hit", "miss"])
        ]
        trial_meta = changes.drop_duplicates("stimulus_presentations_id")
        outcomes = trial_meta["behavior_outcome"].value_counts()
        cell_sets = [
            set(group["cell_specimen_id"].dropna())
            for _, group in changes.groupby("stimulus_presentations_id")
        ]
        common_cells = set.intersection(*cell_sets) if cell_sets else set()
        metadata = manifest.loc[experiment_id]
        output.append(
            {
                "ophys_experiment_id": int(experiment_id),
                "cre_line": experiment["cre_line"].iloc[0],
                "n_change_trials_in_pilot": int(trial_meta["stimulus_presentations_id"].nunique()),
                "n_hit_in_pilot": int(outcomes.get("hit", 0)),
                "n_miss_in_pilot": int(outcomes.get("miss", 0)),
                "n_hit_in_manifest": int(metadata["hit_trial_count"]),
                "n_miss_in_manifest": int(metadata["miss_trial_count"]),
                "manifest_engaged_fraction": (
                    metadata["engaged_trial_count"] / metadata["trial_count"]
                ),
                "n_cells_in_manifest_experiment": int(experiment_cells.loc[experiment_id]),
                "n_cells_in_manifest_session": int(session_cells.loc[metadata["ophys_session_id"]]),
                "n_cells_union_in_pilot": int(changes["cell_specimen_id"].nunique()),
                "n_cells_strict_trial_intersection": len(common_cells),
            }
        )
    return pd.DataFrame(output).sort_values("ophys_experiment_id")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    attrition = manifest_attrition()
    attrition.to_csv(OUT / "manifest_attrition_ladder.csv", index=False)
    print("Manifest attrition:\n", attrition.to_string(index=False), sep="")
    if PILOT.exists():
        pilot = pilot_coverage()
        pilot.to_csv(OUT / "pilot_trial_cell_coverage.csv", index=False)
        print("\nPilot coverage:\n", pilot.to_string(index=False), sep="")


if __name__ == "__main__":
    main()
