"""Analyze session engagement metadata without pretending it is trial-level outcome data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


RAW = Path("data/raw/visual_behavior_ophys_manifest")
OUT = Path("data/processed")
FIG = Path("data/processed/figures")
REFERENCE_ENGAGED_FRACTION = 0.722


def eta_squared(data: pd.DataFrame, group: str, value: str) -> float:
    valid = data[[group, value]].dropna()
    grand_mean = valid[value].mean()
    total_ss = ((valid[value] - grand_mean) ** 2).sum()
    between_ss = sum(
        len(values) * (values.mean() - grand_mean) ** 2
        for _, values in valid.groupby(group)[value]
    )
    return float(between_ss / total_ss) if total_ss else np.nan


def load_active_sessions() -> pd.DataFrame:
    behavior = pd.read_csv(RAW / "behavior_session_table.csv")
    experiments = pd.read_csv(
        RAW / "ophys_experiment_table.csv",
        usecols=["ophys_experiment_id", "ophys_session_id"],
    )
    cells = pd.read_csv(RAW / "ophys_cells_table.csv")
    cell_sessions = cells.merge(experiments, on="ophys_experiment_id", how="inner")
    neurons = (
        cell_sessions.groupby("ophys_session_id")["cell_specimen_id"]
        .nunique()
        .rename("n_neurons_per_session")
    )
    sessions = behavior.merge(neurons, on="ophys_session_id", how="inner")
    sessions = sessions[
        sessions["behavior_type"].eq("active_behavior")
        & sessions["trial_count"].gt(0)
        & (
            sessions["experience_level"].eq("Familiar")
            | sessions["experience_level"].str.startswith("Novel", na=False)
        )
    ].copy()
    sessions["novel"] = sessions["experience_level"].str.startswith("Novel")
    # AllenSDK counts engagement over all indexed trials, including aborted
    # trials, so trial_count is the matching denominator.
    sessions["engaged_fraction"] = sessions["engaged_trial_count"] / sessions["trial_count"]
    return sessions


def summarize(sessions: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    keep = [
        "behavior_session_id",
        "ophys_session_id",
        "mouse_id",
        "project_code",
        "cre_line",
        "novel",
        "experience_level",
        "equipment_name",
        "trial_count",
        "go_trial_count",
        "catch_trial_count",
        "engaged_trial_count",
        "engaged_fraction",
        "hit_trial_count",
        "miss_trial_count",
        "n_neurons_per_session",
    ]
    sessions[keep].to_csv(OUT / "manifest_active_session_engagement.csv", index=False)

    by_cre_novel = (
        sessions.groupby(["cre_line", "novel"])
        .apply(
            lambda group: pd.Series(
                {
                    "n_sessions": group["ophys_session_id"].nunique(),
                    "n_mice": group["mouse_id"].nunique(),
                    "mean_engaged_fraction": group["engaged_fraction"].mean(),
                    "median_engaged_fraction": group["engaged_fraction"].median(),
                    "weighted_engaged_fraction": (
                        group["engaged_trial_count"].sum() / group["trial_count"].sum()
                    ),
                    "q25_engaged_fraction": group["engaged_fraction"].quantile(0.25),
                    "q75_engaged_fraction": group["engaged_fraction"].quantile(0.75),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    by_cre_novel.to_csv(OUT / "engagement_by_cre_novel.csv", index=False)

    by_mouse = (
        sessions.groupby(["cre_line", "project_code", "mouse_id"])
        .agg(
            n_sessions=("ophys_session_id", "nunique"),
            mean_engaged_fraction=("engaged_fraction", "mean"),
            median_engaged_fraction=("engaged_fraction", "median"),
        )
        .reset_index()
    )
    by_mouse.to_csv(OUT / "engagement_by_mouse.csv", index=False)

    mouse_cre = (
        by_mouse.groupby("cre_line")
        .agg(
            n_mice=("mouse_id", "nunique"),
            mean_mouse_engaged_fraction=("mean_engaged_fraction", "mean"),
            median_mouse_engaged_fraction=("mean_engaged_fraction", "median"),
            between_mouse_sd=("mean_engaged_fraction", "std"),
        )
        .reset_index()
    )
    mouse_cre.to_csv(OUT / "engagement_mouse_level_by_cre.csv", index=False)

    mouse_level = sessions.sort_values("ophys_session_id").drop_duplicates("mouse_id")
    cre_cohort = pd.crosstab(mouse_level["cre_line"], mouse_level["project_code"])
    cre_cohort.to_csv(OUT / "mouse_cre_by_project_code_crosstab.csv")

    by_cohort = (
        sessions.groupby(["cre_line", "project_code"])
        .agg(
            n_sessions=("ophys_session_id", "nunique"),
            n_mice=("mouse_id", "nunique"),
            mean_engaged_fraction=("engaged_fraction", "mean"),
            median_engaged_fraction=("engaged_fraction", "median"),
        )
        .reset_index()
    )
    by_cohort.to_csv(OUT / "engagement_by_cohort.csv", index=False)

    clustering = pd.DataFrame(
        [
            {"grouping": "mouse_id", "eta_squared": eta_squared(sessions, "mouse_id", "engaged_fraction")},
            {"grouping": "project_code", "eta_squared": eta_squared(sessions, "project_code", "engaged_fraction")},
            {"grouping": "cre_line", "eta_squared": eta_squared(sessions, "cre_line", "engaged_fraction")},
        ]
    )
    clustering.to_csv(OUT / "engagement_clustering_eta_squared.csv", index=False)

    mixed = smf.mixedlm(
        "engaged_fraction ~ C(cre_line) + C(project_code) + novel",
        sessions,
        groups=sessions["mouse_id"],
    ).fit(reml=True)
    mouse_variance = float(mixed.cov_re.iloc[0, 0])
    residual_variance = float(mixed.scale)
    mixed_summary = pd.DataFrame(
        [
            {
                "n_sessions": len(sessions),
                "n_mice": sessions["mouse_id"].nunique(),
                "mouse_random_intercept_variance": mouse_variance,
                "residual_variance": residual_variance,
                "mouse_icc_adjusted_for_cre_cohort_novel": (
                    mouse_variance / (mouse_variance + residual_variance)
                ),
            }
        ]
    )
    mixed_summary.to_csv(OUT / "engagement_mouse_mixed_model.csv", index=False)
    (OUT / "engagement_mouse_mixed_model.txt").write_text(str(mixed.summary()) + "\n")

    slc = sessions[sessions["cre_line"].eq("Slc17a7-IRES2-Cre")].copy()
    equipment = (
        slc.groupby("equipment_name")
        .agg(
            n_sessions=("ophys_session_id", "nunique"),
            n_mice=("mouse_id", "nunique"),
            median_n_neurons_per_session=("n_neurons_per_session", "median"),
            total_session_neuron_observations=("n_neurons_per_session", "sum"),
        )
        .reset_index()
    )
    equipment["session_share"] = equipment["n_sessions"] / equipment["n_sessions"].sum()
    equipment["session_neuron_observation_share"] = (
        equipment["total_session_neuron_observations"]
        / equipment["total_session_neuron_observations"].sum()
    )
    equipment.to_csv(OUT / "slc17a7_equipment_summary.csv", index=False)

    slc["analysis_branch"] = np.where(
        slc["equipment_name"].eq("MESO.1"),
        "MESO robustness",
        "CAM2P primary",
    )
    slc_branch = (
        slc.groupby("analysis_branch")
        .agg(
            n_sessions=("ophys_session_id", "nunique"),
            n_mice=("mouse_id", "nunique"),
            median_n_neurons_per_session=("n_neurons_per_session", "median"),
            median_total_hit=("hit_trial_count", "median"),
            median_total_miss=("miss_trial_count", "median"),
        )
        .reset_index()
    )
    slc_branch.to_csv(OUT / "slc17a7_primary_vs_meso.csv", index=False)

    capabilities = pd.DataFrame(
        [
            {"requested_quantity": "total hit/miss", "available_in_csv": True, "source": "hit_trial_count / miss_trial_count"},
            {"requested_quantity": "engaged trial count", "available_in_csv": True, "source": "engaged_trial_count, Allen >2 rewards/min"},
            {"requested_quantity": "engaged hit/miss intersection", "available_in_csv": False, "source": "requires trial table (NWB)"},
            {"requested_quantity": "mean_hit_rate_engaged", "available_in_csv": False, "source": "requires get_performance_metrics() / NWB"},
            {"requested_quantity": "ophys_frame_rate", "available_in_csv": False, "source": "requires experiment metadata from NWB"},
        ]
    )
    capabilities.to_csv(OUT / "manifest_capabilities.csv", index=False)
    pd.DataFrame(
        [
            {"property": "Allen engagement threshold", "value": "> 2 rewards/min"},
            {"property": "reward-rate trial window", "value": "trial_number-25 : trial_number+25"},
            {"property": "causal", "value": "no; includes current/future trials"},
            {"property": "first 10 trials", "value": "initialized as NaN after inf replacement"},
            {"property": "Piet 72.2% comparable", "value": "no; different threshold, lick-bout rule, and time basis"},
        ]
    ).to_csv(OUT / "allen_engagement_definition_audit.csv", index=False)

    plot_engagement(sessions)
    plot_mouse_clusters(sessions)
    print("Engagement by cre x novel:\n", by_cre_novel.to_string(index=False), sep="")
    print("\nClustering (descriptive eta-squared):\n", clustering.to_string(index=False), sep="")
    print("\nMouse random-intercept model:\n", mixed_summary.to_string(index=False), sep="")
    print("\nMouse-level engagement by cre:\n", mouse_cre.to_string(index=False), sep="")
    print("\nMouse-level cre x project_code:\n", cre_cohort.to_string(), sep="")
    print("\nSlc17a7 equipment:\n", equipment.to_string(index=False), sep="")
    print("\nSlc17a7 analysis branches:\n", slc_branch.to_string(index=False), sep="")


def plot_engagement(sessions: pd.DataFrame) -> None:
    colors = ["#3366cc", "#dc3912", "#109618"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 26)
    for color, (cre, group) in zip(colors, sessions.groupby("cre_line")):
        ax.hist(
            group["engaged_fraction"], bins=bins, density=True, histtype="step",
            linewidth=2, color=color, label=f"{cre} (n={len(group)})",
        )
    ax.axvline(REFERENCE_ENGAGED_FRACTION, color="black", linestyle="--", linewidth=1.5, label="72.2% Piet time-based reference (different definition)")
    ax.set(xlabel="engaged_trial_count / trial_count", ylabel="Density", xlim=(0, 1))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "engaged_fraction_by_cre.png", dpi=180)
    plt.close(fig)


def plot_mouse_clusters(sessions: pd.DataFrame) -> None:
    mouse_order = (
        sessions.groupby("mouse_id")["engaged_fraction"].mean().sort_values().index
    )
    positions = {mouse: index for index, mouse in enumerate(mouse_order)}
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"Slc17a7-IRES2-Cre": "#3366cc", "Sst-IRES-Cre": "#dc3912", "Vip-IRES-Cre": "#109618"}
    rng = np.random.default_rng(0)
    for cre, group in sessions.groupby("cre_line"):
        x = np.array([positions[mouse] for mouse in group["mouse_id"]], dtype=float)
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(x)), group["engaged_fraction"], s=12, alpha=0.55, label=cre, color=colors[cre])
    ax.axhline(REFERENCE_ENGAGED_FRACTION, color="black", linestyle="--", linewidth=1.2)
    ax.set(xlabel="Mouse, ordered by mean engagement", ylabel="Session engaged fraction", ylim=(0, 1))
    ax.set_xticks([])
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "engagement_by_mouse.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    summarize(load_active_sessions())
