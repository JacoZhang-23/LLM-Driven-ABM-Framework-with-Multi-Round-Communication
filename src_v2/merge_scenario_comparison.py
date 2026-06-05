"""
Build a unified scenario comparison across simulation outputs generated at different times.

Typical use case:
    baseline + memory came from one simulation directory,
    prompt_diversity + combined came from a later simulation directory.

Example:
    python merge_scenario_comparison.py \
        --baseline-dir ../data/output/simulation_20260304_164554/baseline \
        --memory-dir ../data/output/simulation_20260304_164554/memory \
        --prompt-diversity-dir ../data/output/simulation_20260311_122827/prompt_diversity \
        --combined-dir ../data/output/simulation_20260311_122827/combined \
        --output-dir ../data/output/simulation_20260311_122827/comparison
"""

import argparse
import os
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "baseline": "#2E86AB",
    "memory": "#E8850C",
    "prompt_diversity": "#388E3C",
    "combined": "#7B1FA2",
}

SCENARIO_ORDER = ["baseline", "memory", "prompt_diversity", "combined"]

SCENARIO_LABELS = {
    "baseline": "Baseline",
    "memory": "Memory",
    "prompt_diversity": "Prompt Diversity",
    "combined": "Combined",
}

REQUIRED_COLUMNS = [
    "Vaccination_Rate",
    "Average_Belief_LLM",
    "Average_Belief_VADER",
    "Belief_Std_Dev_LLM",
    "Belief_Std_Dev_VADER",
]


def load_scenario_result(scenario_name: str, scenario_dir: str):
    mean_path = os.path.join(scenario_dir, "model_data_mean.csv")
    std_path = os.path.join(scenario_dir, "model_data_std.csv")

    if not os.path.exists(mean_path):
        raise FileNotFoundError(f"Missing mean file for {scenario_name}: {mean_path}")
    if not os.path.exists(std_path):
        raise FileNotFoundError(f"Missing std file for {scenario_name}: {std_path}")

    mean_df = pd.read_csv(mean_path, index_col=0)
    std_df = pd.read_csv(std_path, index_col=0)

    for column in REQUIRED_COLUMNS:
        if column not in mean_df.columns:
            raise ValueError(f"{scenario_name} mean data missing column: {column}")
        if column not in std_df.columns:
            raise ValueError(f"{scenario_name} std data missing column: {column}")

    return mean_df, std_df


def create_output_dir(output_dir: str | None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_dir = os.path.join(base_dir, "data", "output", f"comparison_merged_{timestamp}")
    os.makedirs(auto_dir, exist_ok=True)
    return auto_dir


def plot_opinion_and_polarization(output_dir: str, scenario_results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=100)

    for scenario_name in SCENARIO_ORDER:
        mean_df, std_df = scenario_results[scenario_name]
        steps = mean_df.index
        color = COLORS[scenario_name]
        label = SCENARIO_LABELS[scenario_name]

        axes[0].plot(
            steps,
            mean_df["Average_Belief_LLM"],
            marker="o",
            markersize=6,
            linewidth=2.5,
            color=color,
            label=label,
            alpha=0.9,
        )
        axes[0].fill_between(
            steps,
            mean_df["Average_Belief_LLM"] - std_df["Average_Belief_LLM"],
            mean_df["Average_Belief_LLM"] + std_df["Average_Belief_LLM"],
            color=color,
            alpha=0.14,
        )

        axes[1].plot(
            steps,
            mean_df["Belief_Std_Dev_LLM"],
            marker="s",
            markersize=6,
            linewidth=2.5,
            color=color,
            label=label,
            alpha=0.9,
        )
        axes[1].fill_between(
            steps,
            mean_df["Belief_Std_Dev_LLM"] - std_df["Belief_Std_Dev_LLM"],
            mean_df["Belief_Std_Dev_LLM"] + std_df["Belief_Std_Dev_LLM"],
            color=color,
            alpha=0.14,
        )

    axes[0].set_title("Average Opinion Trajectory (LLM)", fontsize=16, fontweight="bold", pad=15)
    axes[0].set_xlabel("Simulation Step", fontsize=13)
    axes[0].set_ylabel("Average Opinion Score", fontsize=13)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].legend(fontsize=11, frameon=True, shadow=True)
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    axes[1].set_title("Polarization (Opinion Std Dev)", fontsize=16, fontweight="bold", pad=15)
    axes[1].set_xlabel("Simulation Step", fontsize=13)
    axes[1].set_ylabel("Opinion Standard Deviation", fontsize=13)
    axes[1].legend(fontsize=11, frameon=True, shadow=True)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.suptitle(
        "Scenario Comparison: Baseline vs Memory vs Prompt Diversity vs Combined",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "scenario_comparison_belief_polarization.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def plot_vaccination_rate(output_dir: str, scenario_results: dict):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    for scenario_name in SCENARIO_ORDER:
        mean_df, std_df = scenario_results[scenario_name]
        steps = mean_df.index
        color = COLORS[scenario_name]
        label = SCENARIO_LABELS[scenario_name]

        ax.plot(
            steps,
            mean_df["Vaccination_Rate"],
            marker="D",
            markersize=6,
            linewidth=2.5,
            color=color,
            label=label,
            alpha=0.9,
        )
        ax.fill_between(
            steps,
            mean_df["Vaccination_Rate"] - std_df["Vaccination_Rate"],
            mean_df["Vaccination_Rate"] + std_df["Vaccination_Rate"],
            color=color,
            alpha=0.14,
        )

    ax.set_title("Vaccination Rate Comparison Across Scenarios", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Simulation Step", fontsize=13)
    ax.set_ylabel("Vaccination Rate", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "scenario_comparison_vaccination_rate.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def compute_agent_level_convergence(scenario_dir: str):
    """
    Convergence speed = mean over agents of |opinion_t - opinion_{t-1}|,
    then averaged across runs.
    """
    run_dirs = sorted(
        d for d in os.listdir(scenario_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(scenario_dir, d))
    )
    if not run_dirs:
        return None

    per_run_series = []
    for run_dir in run_dirs:
        profile_path = os.path.join(scenario_dir, run_dir, "agent_profiles.json")
        if not os.path.exists(profile_path):
            continue

        with open(profile_path, "r", encoding="utf-8") as file:
            profiles = json.load(file)

        if not profiles:
            continue

        num_steps = len(profiles[0]["belief_history"])
        run_speed = []
        for step in range(1, num_steps):
            step_mean_abs = np.mean([
                abs(agent["belief_history"][step] - agent["belief_history"][step - 1])
                for agent in profiles
            ])
            run_speed.append(float(step_mean_abs))

        per_run_series.append(run_speed)

    if not per_run_series:
        return None

    return np.mean(np.array(per_run_series), axis=0).tolist()


def plot_convergence_speed(output_dir: str, scenario_results: dict, scenario_dirs: dict):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    for scenario_name in SCENARIO_ORDER:
        color = COLORS[scenario_name]
        label = SCENARIO_LABELS[scenario_name]

        # Preferred: agent-level absolute delta (more robust to cancellation)
        change_rate = compute_agent_level_convergence(scenario_dirs[scenario_name])

        # Fallback: absolute delta of scenario-level mean opinion
        if change_rate is None:
            mean_df, _ = scenario_results[scenario_name]
            opinion_series = mean_df["Average_Belief_LLM"].values
            change_rate = np.abs(np.diff(opinion_series)).tolist()

        ax.plot(
            range(1, len(change_rate) + 1),
            change_rate,
            marker="o",
            markersize=5,
            linewidth=2,
            color=color,
            label=label,
            alpha=0.9,
        )

    ax.set_title("Convergence Speed (mean |Δ Opinion| per agent per step)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Simulation Step", fontsize=13)
    ax.set_ylabel("mean |Δ Opinion|", fontsize=13)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "scenario_comparison_convergence_speed.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def save_summary(output_dir: str, scenario_results: dict, scenario_dirs: dict):
    summary_rows = []

    for scenario_name in SCENARIO_ORDER:
        mean_df, std_df = scenario_results[scenario_name]
        final_mean = mean_df.iloc[-1]
        final_std = std_df.iloc[-1]
        summary_rows.append({
            "Scenario": SCENARIO_LABELS[scenario_name],
            "Source Directory": scenario_dirs[scenario_name],
            "Final Avg Opinion (LLM)": f"{final_mean['Average_Belief_LLM']:.3f} ± {final_std['Average_Belief_LLM']:.3f}",
            "Final Polarization (LLM)": f"{final_mean['Belief_Std_Dev_LLM']:.3f} ± {final_std['Belief_Std_Dev_LLM']:.3f}",
            "Final Vax Rate": f"{final_mean['Vaccination_Rate']:.3f} ± {final_std['Vaccination_Rate']:.3f}",
            "Final Avg Opinion (VADER)": f"{final_mean['Average_Belief_VADER']:.3f} ± {final_std['Average_Belief_VADER']:.3f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "scenario_comparison_summary.csv"), index=False)

    with open(os.path.join(output_dir, "scenario_comparison_summary.txt"), "w", encoding="utf-8") as file:
        file.write("=" * 90 + "\n")
        file.write("FOUR-SCENARIO COMPARISON SUMMARY\n")
        file.write("=" * 90 + "\n\n")
        for row in summary_rows:
            file.write(f"Scenario: {row['Scenario']}\n")
            file.write(f"  Source Directory:        {row['Source Directory']}\n")
            file.write(f"  Final Avg Opinion (LLM): {row['Final Avg Opinion (LLM)']}\n")
            file.write(f"  Final Polarization:      {row['Final Polarization (LLM)']}\n")
            file.write(f"  Final Vaccination Rate:  {row['Final Vax Rate']}\n")
            file.write(f"  Final Avg Opinion VADER: {row['Final Avg Opinion (VADER)']}\n")
            file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge scenario results from different simulation folders.")
    parser.add_argument("--baseline-dir", required=True, help="Path to baseline scenario directory")
    parser.add_argument("--memory-dir", required=True, help="Path to memory scenario directory")
    parser.add_argument("--prompt-diversity-dir", required=True, help="Path to prompt_diversity scenario directory")
    parser.add_argument("--combined-dir", required=True, help="Path to combined scenario directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for merged comparison")
    return parser.parse_args()


def main():
    args = parse_args()

    scenario_dirs = {
        "baseline": os.path.abspath(args.baseline_dir),
        "memory": os.path.abspath(args.memory_dir),
        "prompt_diversity": os.path.abspath(args.prompt_diversity_dir),
        "combined": os.path.abspath(args.combined_dir),
    }

    output_dir = create_output_dir(args.output_dir)
    scenario_results = {}

    for scenario_name in SCENARIO_ORDER:
        scenario_results[scenario_name] = load_scenario_result(scenario_name, scenario_dirs[scenario_name])

    plot_opinion_and_polarization(output_dir, scenario_results)
    plot_vaccination_rate(output_dir, scenario_results)
    plot_convergence_speed(output_dir, scenario_results, scenario_dirs)
    save_summary(output_dir, scenario_results, scenario_dirs)

    print("\nMerged four-scenario comparison saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()