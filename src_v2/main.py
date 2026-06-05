# main.py

"""
Main entry point for the LLM-based Multi-round Dialogue Vaccination Simulation.
Supports scenario comparison: baseline vs memory vs prompt_diversity.
"""
import os
import sys
import traceback
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json

from model import VaxSimulationModel
from analysis import run_all_analyses
from visualize_batch_results import (
    visualize_comparative_trends,
    visualize_belief_distributions,
    generate_network_evolution,
    plot_influence_scatter,
    compute_average_beliefs
)
import numpy as np

# 定义配色方案
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'llm': '#2E86AB',
    'vader': '#D62246',
    'baseline': '#2E86AB',
    'memory': '#E8850C',
    'prompt_diversity': '#388E3C',
    'combined': '#7B1FA2',  # Purple for combined scenario
}

# Scenario definitions: name → (memory_enabled, prompt_diversity_enabled)
SCENARIO_DEFS = {
    "baseline":         (False, False),
    "memory":           (True,  False),
    "prompt_diversity":  (False, True),
    "combined":         (True,  True),   # Both Memory + Prompt Diversity enabled
}


def generate_fixed_initial_beliefs(num_agents: int, seed: int = 42):
    """
    生成固定的初始belief，用于所有batch运行
    确保每个agent在所有运行中的初始belief保持一致
    """
    from config import BELIEF_DISTRIBUTION_TYPE, BELIEF_MEANS, BELIEF_STD
    
    np.random.seed(seed)
    random.seed(seed)
    
    mu = BELIEF_MEANS.get(BELIEF_DISTRIBUTION_TYPE, 0.0)
    initial_beliefs = []
    
    for i in range(num_agents):
        belief = float(np.clip(np.random.normal(mu, BELIEF_STD), -1.0, 1.0))
        initial_beliefs.append(belief)
    
    print(f"\n✓ 生成固定初始belief (seed={seed}):")
    print(f"   - 数量: {num_agents}")
    print(f"   - 平均值: {np.mean(initial_beliefs):.3f}")
    print(f"   - 标准差: {np.std(initial_beliefs):.3f}")
    print(f"   - 范围: [{min(initial_beliefs):.3f}, {max(initial_beliefs):.3f}]")
    
    return initial_beliefs


def run_single_scenario(
    scenario_name: str,
    memory_enabled: bool,
    prompt_diversity_enabled: bool,
    output_dir: str,
    population_csv: str,
    network_csv: str,
    fixed_initial_beliefs: list,
    num_agents: int,
):
    """
    Run one scenario (with BATCH_RUNS repetitions) and save results.
    Returns: (mean_df, std_df, all_profiles, network_data)
    """
    from config import (
        MAX_STEPS, AGENT_ALPHA, API_KEY, API_URL, MODEL_NAME, MAX_CONCURRENT_CALLS,
        BATCH_RUNS, BELIEF_DISTRIBUTION_TYPE
    )

    scenario_dir = os.path.join(output_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🔬 Scenario: {scenario_name.upper()}")
    print(f"   Memory: {'ON' if memory_enabled else 'OFF'}  |  Prompt Diversity: {'ON' if prompt_diversity_enabled else 'OFF'}")
    print(f"{'='*60}")

    model_dfs = []
    all_profiles = []
    network_data = None

    for run_idx in range(BATCH_RUNS):
        run_dir = os.path.join(scenario_dir, f"run_{run_idx + 1:02d}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n🔁 [{scenario_name}] Batch Run {run_idx + 1}/{BATCH_RUNS} (belief={BELIEF_DISTRIBUTION_TYPE})")

        model = VaxSimulationModel(
            max_steps=MAX_STEPS,
            agent_alpha=AGENT_ALPHA,
            api_url=API_URL,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            max_concurrent=MAX_CONCURRENT_CALLS,
            use_workplace_data=True,
            population_csv=population_csv,
            network_csv=network_csv,
            fixed_initial_beliefs=fixed_initial_beliefs,
            memory_enabled=memory_enabled,
            prompt_diversity_enabled=prompt_diversity_enabled,
        )

        model.run_model()
        model.export_results(run_dir)

        model_dfs.append(model.datacollector.get_model_vars_dataframe())

        with open(os.path.join(run_dir, "agent_profiles.json"), 'r') as f:
            all_profiles.append(json.load(f))

        if network_data is None:
            with open(os.path.join(run_dir, "network_data.json"), 'r') as f:
                network_data = json.load(f)

    # Aggregate model metrics across runs (mean + std)
    combined = pd.concat(model_dfs, keys=range(BATCH_RUNS))
    mean_df = combined.groupby(level=1).mean()
    std_df = combined.groupby(level=1).std()

    mean_df.to_csv(os.path.join(scenario_dir, "model_data_mean.csv"))
    std_df.to_csv(os.path.join(scenario_dir, "model_data_std.csv"))

    # Generate per-scenario visualizations
    viz_dir = os.path.join(scenario_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    avg_beliefs_llm, avg_beliefs_vader = compute_average_beliefs(all_profiles)
    visualize_comparative_trends(scenario_dir, viz_dir)
    visualize_belief_distributions(all_profiles, viz_dir)
    generate_network_evolution(scenario_dir, network_data, avg_beliefs_llm, viz_dir)
    plot_influence_scatter(scenario_dir, viz_dir)

    print(f"\n✅ Scenario '{scenario_name}' complete.")
    final_mean = mean_df.iloc[-1]
    final_std = std_df.iloc[-1]
    print(f"   Final Avg Belief (LLM): {final_mean['Average_Belief_LLM']:.3f} ± {final_std['Average_Belief_LLM']:.3f}")
    print(f"   Final Polarization (LLM): {final_mean['Belief_Std_Dev_LLM']:.3f} ± {final_std['Belief_Std_Dev_LLM']:.3f}")
    print(f"   Final Vax Rate: {final_mean['Vaccination_Rate']:.3%} ± {final_std['Vaccination_Rate']:.3%}")

    return mean_df, std_df, all_profiles, network_data


def generate_scenario_comparison(output_dir: str, scenario_results: dict):
    """
    Generate comparison charts across scenarios.
    scenario_results: {scenario_name: (mean_df, std_df)}
    """
    cmp_dir = os.path.join(output_dir, "comparison")
    os.makedirs(cmp_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("📊 Generating Scenario Comparison Visualizations")
    print(f"{'='*60}")

    scenario_names = list(scenario_results.keys())
    
    # --- Plot 1: Average Belief Trajectory Comparison (LLM) ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=100)
    
    for name in scenario_names:
        mean_df, std_df = scenario_results[name]
        color = COLORS.get(name, '#666666')
        steps = mean_df.index
        
        # Left: Average Belief
        axes[0].plot(steps, mean_df['Average_Belief_LLM'], marker='o', markersize=6,
                     linewidth=2.5, color=color, label=name.replace('_', ' ').title(), alpha=0.85)
        axes[0].fill_between(steps,
                             mean_df['Average_Belief_LLM'] - std_df['Average_Belief_LLM'],
                             mean_df['Average_Belief_LLM'] + std_df['Average_Belief_LLM'],
                             color=color, alpha=0.15)
        
        # Right: Polarization (Std Dev)
        axes[1].plot(steps, mean_df['Belief_Std_Dev_LLM'], marker='s', markersize=6,
                     linewidth=2.5, color=color, label=name.replace('_', ' ').title(), alpha=0.85)
        axes[1].fill_between(steps,
                             mean_df['Belief_Std_Dev_LLM'] - std_df['Belief_Std_Dev_LLM'],
                             mean_df['Belief_Std_Dev_LLM'] + std_df['Belief_Std_Dev_LLM'],
                             color=color, alpha=0.15)

    axes[0].set_title('Average Opinion Trajectory (LLM)', fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel('Simulation Step', fontsize=13)
    axes[0].set_ylabel('Average Belief Score', fontsize=13)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].legend(fontsize=11, frameon=True, shadow=True)
    axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    axes[1].set_title('Polarization (Belief Std Dev)', fontsize=16, fontweight='bold', pad=15)
    axes[1].set_xlabel('Simulation Step', fontsize=13)
    axes[1].set_ylabel('Belief Standard Deviation', fontsize=13)
    axes[1].legend(fontsize=11, frameon=True, shadow=True)
    axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    fig.suptitle('Scenario Comparison: ' + ' vs '.join(s.replace('_', ' ').title() for s in scenario_names),
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(cmp_dir, "scenario_comparison_belief_polarization.png"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Saved scenario_comparison_belief_polarization.png")

    # --- Plot 2: Vaccination Rate Comparison ---
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    for name in scenario_names:
        mean_df, std_df = scenario_results[name]
        color = COLORS.get(name, '#666666')
        steps = mean_df.index
        ax.plot(steps, mean_df['Vaccination_Rate'], marker='D', markersize=6,
                linewidth=2.5, color=color, label=name.replace('_', ' ').title(), alpha=0.85)
        ax.fill_between(steps,
                         mean_df['Vaccination_Rate'] - std_df['Vaccination_Rate'],
                         mean_df['Vaccination_Rate'] + std_df['Vaccination_Rate'],
                         color=color, alpha=0.15)

    ax.set_title('Vaccination Rate Comparison Across Scenarios', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Simulation Step', fontsize=13)
    ax.set_ylabel('Vaccination Rate', fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(cmp_dir, "scenario_comparison_vaccination_rate.png"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Saved scenario_comparison_vaccination_rate.png")

    # --- Table: Final Step Comparison Summary ---
    summary_rows = []
    for name in scenario_names:
        mean_df, std_df = scenario_results[name]
        final_mean = mean_df.iloc[-1]
        final_std = std_df.iloc[-1]
        summary_rows.append({
            'Scenario': name.replace('_', ' ').title(),
            'Final Avg Belief (LLM)': f"{final_mean['Average_Belief_LLM']:.3f} ± {final_std['Average_Belief_LLM']:.3f}",
            'Final Polarization (LLM)': f"{final_mean['Belief_Std_Dev_LLM']:.3f} ± {final_std['Belief_Std_Dev_LLM']:.3f}",
            'Final Vax Rate': f"{final_mean['Vaccination_Rate']:.3f} ± {final_std['Vaccination_Rate']:.3f}",
            'Final Avg Belief (VADER)': f"{final_mean['Average_Belief_VADER']:.3f} ± {final_std['Average_Belief_VADER']:.3f}",
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(cmp_dir, "scenario_comparison_summary.csv"), index=False)
    
    # Also save as formatted text
    with open(os.path.join(cmp_dir, "scenario_comparison_summary.txt"), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SCENARIO COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        for row in summary_rows:
            f.write(f"Scenario: {row['Scenario']}\n")
            f.write(f"  Final Avg Belief (LLM):   {row['Final Avg Belief (LLM)']}\n")
            f.write(f"  Final Polarization (LLM): {row['Final Polarization (LLM)']}\n")
            f.write(f"  Final Vaccination Rate:   {row['Final Vax Rate']}\n")
            f.write(f"  Final Avg Belief (VADER): {row['Final Avg Belief (VADER)']}\n")
            f.write("\n")
    
    print("  ✓ Saved scenario_comparison_summary.csv and .txt")

    # --- Plot 3: Convergence Speed Comparison (Belief change rate) ---
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    for name in scenario_names:
        mean_df, _ = scenario_results[name]
        color = COLORS.get(name, '#666666')
        belief_series = mean_df['Average_Belief_LLM'].values
        # Convergence speed = absolute change per step
        change_rate = np.abs(np.diff(belief_series))
        ax.plot(range(1, len(change_rate) + 1), change_rate, marker='o', markersize=5,
                linewidth=2, color=color, label=name.replace('_', ' ').title(), alpha=0.85)

    ax.set_title('Convergence Speed (|Δ Belief| per Step)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Simulation Step', fontsize=13)
    ax.set_ylabel('|Δ Average Belief|', fontsize=13)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(cmp_dir, "scenario_comparison_convergence_speed.png"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Saved scenario_comparison_convergence_speed.png")

    print(f"\n📁 All comparison outputs saved to: {cmp_dir}")


def main():
    """Main function to configure and run the simulation."""
    from config import (
        MAX_STEPS, AGENT_ALPHA, API_KEY, API_URL, MODEL_NAME, MAX_CONCURRENT_CALLS,
        BATCH_RUNS, BELIEF_DISTRIBUTION_TYPE, SCENARIO_MODE
    )

    print("\n" + "=" * 60)
    print("🔬 Starting LLM-based Vaccination Simulation")
    print(f"   Scenario Mode: {SCENARIO_MODE}")
    print("=" * 60 + "\n")

    # Define base directory (project root, one level up from src_v2/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create output directory with absolute path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, "data", "output", f"simulation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Define absolute paths to workplace CSV files
    population_csv = os.path.join(base_dir, "data", "input", "workplace_36013030400w1_extended_population.csv")
    network_csv = os.path.join(base_dir, "data", "input", "workplace_36013030400w1_extended_network.csv")

    # Verify files exist
    if not os.path.exists(population_csv):
        raise FileNotFoundError(f"Population CSV not found: {population_csv}")
    if not os.path.exists(network_csv):
        raise FileNotFoundError(f"Network CSV not found: {network_csv}")

    print(f"\n✅ Loading workplace data from CSV files:")
    print(f"   Population: {os.path.basename(population_csv)}")
    print(f"   Network: {os.path.basename(network_csv)}")

    # 读取population数据获取agent数量
    pop_df = pd.read_csv(population_csv)
    num_agents = len(pop_df)

    # 生成固定的初始belief（所有batch运行 + 所有scenario使用相同的初始值）
    fixed_initial_beliefs = generate_fixed_initial_beliefs(num_agents, seed=42)

    # Determine which scenarios to run
    if SCENARIO_MODE == "all":
        scenarios_to_run = list(SCENARIO_DEFS.keys())
    elif ',' in SCENARIO_MODE:
        # Comma-separated list: e.g. "prompt_diversity,combined"
        requested = [s.strip() for s in SCENARIO_MODE.split(',')]
        unknown = [s for s in requested if s not in SCENARIO_DEFS]
        if unknown:
            print(f"⚠️ Unknown scenario(s) in SCENARIO_MODE: {unknown}. Skipping.")
        scenarios_to_run = [s for s in requested if s in SCENARIO_DEFS]
    elif SCENARIO_MODE in SCENARIO_DEFS:
        scenarios_to_run = [SCENARIO_MODE]
    else:
        print(f"⚠️ Unknown SCENARIO_MODE '{SCENARIO_MODE}', running baseline only.")
        scenarios_to_run = ["baseline"]

    print(f"\n📋 Scenarios to run: {scenarios_to_run}")

    # Run each scenario and collect results
    scenario_results = {}  # {name: (mean_df, std_df)}
    
    for scenario_name in scenarios_to_run:
        mem_on, div_on = SCENARIO_DEFS[scenario_name]
        mean_df, std_df, all_profiles, network_data = run_single_scenario(
            scenario_name=scenario_name,
            memory_enabled=mem_on,
            prompt_diversity_enabled=div_on,
            output_dir=output_dir,
            population_csv=population_csv,
            network_csv=network_csv,
            fixed_initial_beliefs=fixed_initial_beliefs,
            num_agents=num_agents,
        )
        scenario_results[scenario_name] = (mean_df, std_df)

    # Generate comparison visualizations (only if multiple scenarios)
    if len(scenario_results) > 1:
        generate_scenario_comparison(output_dir, scenario_results)

    # Print final summary
    print("\n" + "=" * 60)
    print("🏁 FINAL SUMMARY")
    print("=" * 60)
    print(f"   Agents: {num_agents}  |  Steps: {MAX_STEPS}  |  Batch Runs: {BATCH_RUNS}")
    for name, (mean_df, std_df) in scenario_results.items():
        final_mean = mean_df.iloc[-1]
        final_std = std_df.iloc[-1]
        print(f"\n   [{name.upper()}]")
        print(f"     Avg Belief (LLM):   {final_mean['Average_Belief_LLM']:.3f} ± {final_std['Average_Belief_LLM']:.3f}")
        print(f"     Polarization (LLM): {final_mean['Belief_Std_Dev_LLM']:.3f} ± {final_std['Belief_Std_Dev_LLM']:.3f}")
        print(f"     Vaccination Rate:   {final_mean['Vaccination_Rate']:.3%} ± {final_std['Vaccination_Rate']:.3%}")
    
    print(f"\n📁 All outputs: {output_dir}")
    print("✅ Simulation finished!\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)