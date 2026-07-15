"""Summarize engaged Visual Behavior Ophys sessions without downloading NWB files."""

from pathlib import Path

import pandas as pd


RAW = Path("data/raw/visual_behavior_ophys_manifest")
OUT = Path("data/processed")


def main() -> None:
    behavior = pd.read_csv(RAW / "behavior_session_table.csv")
    experiments = pd.read_csv(
        RAW / "ophys_experiment_table.csv",
        usecols=["ophys_experiment_id", "ophys_session_id"],
    )
    cells = pd.read_csv(RAW / "ophys_cells_table.csv")

    cell_sessions = cells.merge(experiments, on="ophys_experiment_id", how="inner")
    neurons = (
        cell_sessions.groupby("ophys_session_id", as_index=False)["cell_specimen_id"]
        .nunique()
        .rename(columns={"cell_specimen_id": "n_neurons_per_session"})
    )

    sessions = behavior.merge(neurons, on="ophys_session_id", how="inner")
    sessions = sessions[
        sessions["behavior_type"].eq("active_behavior")
        & (
            sessions["experience_level"].eq("Familiar")
            | sessions["experience_level"].str.startswith("Novel", na=False)
        )
        & sessions["trial_count"].gt(0)
    ].copy()
    sessions["engaged_fraction"] = sessions["engaged_trial_count"] / sessions["trial_count"]
    sessions["novel"] = sessions["experience_level"].str.startswith("Novel")
    sessions["equipment"] = sessions["equipment_name"]
    sessions["n_total_miss"] = sessions["miss_trial_count"]

    summary = (
        sessions.groupby(["cre_line", "novel", "equipment"], dropna=False)
        .agg(
            n_sessions=("ophys_session_id", "nunique"),
            median_n_neurons_per_session=("n_neurons_per_session", "median"),
            median_n_total_miss=("n_total_miss", "median"),
        )
        .reset_index()
        .sort_values(["cre_line", "novel", "equipment"])
    )

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "cre_novel_equipment_active_summary.csv", index=False)
    sessions[
        [
            "behavior_session_id",
            "ophys_session_id",
            "cre_line",
            "novel",
            "experience_level",
            "equipment",
            "engaged_fraction",
            "n_neurons_per_session",
            "n_total_miss",
        ]
    ].to_csv(OUT / "manifest_active_sessions.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
