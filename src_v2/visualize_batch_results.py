"""
批量运行结果可视化脚本
根据批量运行的输出数据生成可视化图表，包括：
1. 使用平均model_data生成comparative belief trends
2. 使用10次运行的平均生成agent层级的belief distribution
3. 生成网络演化可视化（基于平均belief值）
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import networkx as nx
from tqdm import tqdm
from pathlib import Path

# 定义配色方案（与 analysis.py 保持一致）
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'llm': '#2E86AB',
    'vader': '#D62246',
    'neutral': '#7F8C8D',
}


def load_batch_data(simulation_dir, num_runs=10):
    """
    加载所有批次运行的数据
    
    返回:
        all_profiles: 所有运行的agent profiles列表
        network_data: 第一个运行的网络数据（所有运行使用相同网络）
    """
    print(f"\n📂 加载批量运行数据...")
    all_profiles = []
    network_data = None
    
    for run_idx in range(1, num_runs + 1):
        run_dir = os.path.join(simulation_dir, f"run_{run_idx:02d}")
        
        # 加载agent profiles
        profiles_file = os.path.join(run_dir, "agent_profiles.json")
        if os.path.exists(profiles_file):
            with open(profiles_file, 'r') as f:
                profiles = json.load(f)
                all_profiles.append(profiles)
                print(f"   ✓ Run {run_idx:02d}: {len(profiles)} agents")
        
        # 只需要加载一次网络数据（所有运行使用相同网络）
        if network_data is None:
            network_file = os.path.join(run_dir, "network_data.json")
            if os.path.exists(network_file):
                with open(network_file, 'r') as f:
                    network_data = json.load(f)
                    print(f"   ✓ 网络数据: {len(network_data['edges'])} 条边")
    
    return all_profiles, network_data


def compute_average_beliefs(all_profiles):
    """
    计算所有运行中每个agent每个时间步的平均belief
    
    返回:
        avg_beliefs_llm: {agent_id: [belief_t0, belief_t1, ...]}
        avg_beliefs_vader: {agent_id: [belief_t0, belief_t1, ...]}
    """
    print("\n📊 计算平均belief值...")
    
    num_agents = len(all_profiles[0])
    num_steps = len(all_profiles[0][0]['belief_history'])
    num_runs = len(all_profiles)
    
    # 初始化存储结构
    avg_beliefs_llm = {}
    avg_beliefs_vader = {}
    
    # 对每个agent
    for agent_id in range(num_agents):
        belief_llm_all_runs = []
        belief_vader_all_runs = []
        
        # 收集所有运行的belief历史
        for run_profiles in all_profiles:
            agent_profile = run_profiles[agent_id]
            belief_llm_all_runs.append(agent_profile['belief_history'])
            belief_vader_all_runs.append(agent_profile['belief_vader_history'])
        
        # 计算平均值（按时间步）
        avg_beliefs_llm[agent_id] = np.mean(belief_llm_all_runs, axis=0).tolist()
        avg_beliefs_vader[agent_id] = np.mean(belief_vader_all_runs, axis=0).tolist()
    
    print(f"   ✓ 计算完成: {num_agents} agents × {num_steps} steps")
    return avg_beliefs_llm, avg_beliefs_vader


def visualize_comparative_trends(simulation_dir, output_dir):
    """
    生成平均belief趋势对比图（使用已有的model_data_mean.csv）
    """
    print("\n📈 生成平均belief趋势图...")
    
    mean_file = os.path.join(simulation_dir, "model_data_mean.csv")
    if not os.path.exists(mean_file):
        print(f"   ✗ 找不到 {mean_file}")
        return
    
    model_df = pd.read_csv(mean_file, index_col=0)
    model_df.index.name = 'Step'
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    
    # 绘制belief趋势线（左侧Y轴）
    ax.plot(model_df.index, model_df['Average_Belief_LLM'], 
           marker='o', markersize=8, linewidth=2.5, 
           color=COLORS['llm'], label='LLM Self-Score Driven', alpha=0.85)
    ax.plot(model_df.index, model_df['Average_Belief_VADER'], 
           marker='s', markersize=8, linewidth=2.5, 
           color=COLORS['vader'], label='VADER Sentiment Driven', alpha=0.85)
    
    # 添加参考线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # 设置左侧Y轴标签
    ax.set_xlabel('Simulation Step', fontsize=14, fontweight='600')
    ax.set_ylabel('Average Belief Score', fontsize=14, fontweight='600')
    ax.set_ylim(-1.05, 1.05)
    
    # 创建右侧Y轴用于显示vaccination rate
    ax2 = ax.twinx()
    ax2.plot(model_df.index, model_df['Vaccination_Rate'] * 100, 
            marker='^', markersize=7, linewidth=2.0, 
            color='#FF6B6B', label='Vaccination Rate', 
            linestyle='--', alpha=0.8)
    ax2.set_ylabel('Vaccination Rate (%)', fontsize=14, fontweight='600')
    ax2.set_ylim(0, 105)
    
    # 设置标题
    ax.set_title('Comparison of Average Belief Evolution & Vaccination Rate\n(Averaged over 10 runs)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # 美化网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    
    # 合并两个Y轴的图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12, 
             loc='best', frameon=True, shadow=True, fancybox=True)
    ax.get_legend().get_frame().set_facecolor('white')
    ax.get_legend().get_frame().set_alpha(0.9)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "comparative_belief_trends.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ 保存: {os.path.basename(output_file)}")


def visualize_belief_distributions(all_profiles, output_dir):
    """
    生成2x2 belief分布矩阵图（初始vs最终，LLM vs VADER）
    使用10次运行的平均分布
    """
    print("\n📊 生成belief分布矩阵...")
    
    num_agents = len(all_profiles[0])
    num_runs = len(all_profiles)
    num_steps = len(all_profiles[0][0]['belief_history'])
    
    # 收集所有运行的初始和最终belief
    all_initial_llm = []
    all_final_llm = []
    all_initial_vader = []
    all_final_vader = []
    
    for run_profiles in all_profiles:
        for agent in run_profiles:
            all_initial_llm.append(agent['belief_history'][0])
            all_final_llm.append(agent['belief_history'][-1])
            all_initial_vader.append(agent['belief_vader_history'][0])
            all_final_vader.append(agent['belief_vader_history'][-1])
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, dpi=100)
    
    num_bins = max(15, num_agents // 2)
    
    # Plot A: Initial Distribution (LLM method)
    sns.histplot(all_initial_llm, kde=True, bins=num_bins, 
                color='gray', ax=axes[0, 0], alpha=0.6, line_kws={'linewidth': 2})
    axes[0, 0].set_title(f'LLM Self-Score Method (Step 0)', 
                       fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].set_ylabel('Count', fontsize=12, fontweight='600')
    
    # Plot C: Final Distribution (LLM method)
    sns.histplot(all_final_llm, kde=True, bins=num_bins, 
                color=COLORS['llm'], ax=axes[1, 0], alpha=0.7, line_kws={'linewidth': 2})
    axes[1, 0].set_title(f'LLM Self-Score Method (Step {num_steps-1})', 
                       fontsize=14, fontweight='bold', pad=10)
    axes[1, 0].set_xlabel('Belief Score', fontsize=12, fontweight='600')
    axes[1, 0].set_ylabel('Count', fontsize=12, fontweight='600')
    
    # Plot B: Initial Distribution (VADER method)
    sns.histplot(all_initial_vader, kde=True, bins=num_bins, 
                color='gray', ax=axes[0, 1], alpha=0.6, line_kws={'linewidth': 2})
    axes[0, 1].set_title(f'VADER Sentiment Method (Step 0)', 
                       fontsize=14, fontweight='bold', pad=10)
    
    # Plot D: Final Distribution (VADER method)
    sns.histplot(all_final_vader, kde=True, bins=num_bins, 
                color=COLORS['vader'], ax=axes[1, 1], alpha=0.7, line_kws={'linewidth': 2})
    axes[1, 1].set_title(f'VADER Sentiment Method (Step {num_steps-1})', 
                       fontsize=14, fontweight='bold', pad=10)
    axes[1, 1].set_xlabel('Belief Score', fontsize=12, fontweight='600')
    
    # 统一设置所有子图
    for ax in axes.flatten():
        ax.set_xlim(-1.05, 1.05)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
        ax.set_axisbelow(True)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    fig.suptitle(f'Belief Distribution Comparison: LLM Self-Score vs. VADER Sentiment\n(Averaged over {num_runs} runs)', 
                fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    output_file = os.path.join(output_dir, "belief_distribution_matrix.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ 保存: {os.path.basename(output_file)}")


def generate_network_evolution(simulation_dir, network_data, avg_beliefs_llm, output_dir):
    """
    生成网络演化可视化（使用平均belief值）
    参数:
        simulation_dir: 模拟根目录（用于读取model_data_mean.csv）
        network_data: 网络数据字典
        avg_beliefs_llm: 平均belief字典
        output_dir: 输出目录（viz_dir）
    """
    print("\n🎨 生成网络演化可视化...")
    
    # 创建输出目录
    frames_dir = os.path.join(output_dir, "network_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # === 优化网络布局 ===
    network_edges = [tuple(e) for e in network_data['edges']]
    G_layout = nx.Graph()
    G_layout.add_edges_from(network_edges)
    
    # 确保所有节点都在图中
    all_node_ids = [int(k) for k in network_data['layout'].keys()]
    G_layout.add_nodes_from(all_node_ids)
    
    # 使用原始位置作为初始位置
    initial_pos = {int(k): np.array(v) for k, v in network_data['layout'].items()}
    
    # 使用 Kamada-Kawai 布局优化
    network_layout = nx.kamada_kawai_layout(
        G_layout,
        pos=initial_pos,
        scale=3.0,
        weight=None
    )
    
    # 确定步骤数
    num_steps = len(list(avg_beliefs_llm.values())[0])
    
    # 加载model_data_mean用于统计信息（从父目录）
    mean_file = os.path.join(output_dir, "model_data_mean.csv")
    model_data = None
    if os.path.exists(mean_file):
        model_data = pd.read_csv(mean_file, index_col=0)
    
    # 设置颜色映射
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'belief_cmap',
        ['#D32F2F', '#F57C00', '#FDD835', '#9CCC65', '#388E3C']
    )
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    
    print(f"   生成 {num_steps} 帧...")
    
    # 为每个步骤生成图片
    for step in tqdm(range(num_steps), desc="   生成帧"):
        # 收集当前步骤的平均belief
        current_beliefs = {agent_id: avg_beliefs_llm[agent_id][step] 
                          for agent_id in avg_beliefs_llm.keys()}
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(24, 16), dpi=180)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # 映射到颜色
        node_colors = [cmap(norm(current_beliefs[node])) 
                      for node in sorted(current_beliefs.keys())]
        
        # 创建网络图
        G = nx.Graph()
        G.add_nodes_from(current_beliefs.keys())
        G.add_edges_from(network_edges)
        
        # 绘制边
        nx.draw_networkx_edges(
            G, network_layout, ax=ax,
            edge_color='#9E9E9E',
            width=1.4,
            alpha=0.32,
            style='solid'
        )
        
        # 绘制阴影
        shadow_pos = {node: (pos[0] + 0.02, pos[1] - 0.02) 
                     for node, pos in network_layout.items()}
        nx.draw_networkx_nodes(
            G, shadow_pos, ax=ax,
            node_color='black',
            node_size=1000,
            alpha=0.14
        )
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, network_layout, ax=ax,
            node_color=node_colors,
            node_size=1000,
            edgecolors='#212121',
            linewidths=2.8,
            alpha=0.94
        )
        
        # 绘制标签
        labels = {node: f"{node}" for node in current_beliefs.keys()}
        nx.draw_networkx_labels(
            G, network_layout, labels, ax=ax,
            font_size=16,
            font_weight='bold',
            font_color='black',
            font_family='sans-serif'
        )
        
        # 添加标题
        title_text = f'Agent Network - Average Belief Evolution (Step {step})'
        ax.text(
            0.5, 1.02, title_text,
            transform=ax.transAxes,
            fontsize=34,
            fontweight='bold',
            ha='center',
            va='bottom'
        )
        
        ax.axis('off')
        ax.margins(0.1)
        ax.set_aspect('equal')
        
        # 添加颜色条
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.70])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(
            'Belief Score',
            rotation=270,
            labelpad=26,
            fontsize=22,
            fontweight='bold'
        )
        cbar.ax.tick_params(labelsize=15, width=2, length=5)
        cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        cbar.set_ticklabels([
            '-1.0\nStrongly\nOppose',
            '-0.5\nOppose',
            '0.0\nNeutral',
            '+0.5\nSupport',
            '+1.0\nStrongly\nSupport'
        ])
        cbar.outline.set_edgecolor('#424242')
        cbar.outline.set_linewidth(2)
        
        # 添加统计信息
        if model_data is not None and step < len(model_data):
            avg_belief = model_data.loc[step, 'Average_Belief_LLM']
            vacc_rate = model_data.loc[step, 'Vaccination_Rate']
            num_agents = len(current_beliefs)
            num_vaccinated = int(vacc_rate * num_agents)
            
            fig.text(
                0.03, 0.95,
                f'Step: {step} / {num_steps-1}',
                transform=fig.transFigure,
                fontsize=20,
                fontweight='bold',
                ha='left',
                va='top',
                color='#1976D2',
                zorder=1001
            )
            
            fig.text(
                0.03, 0.91,
                f'Avg Belief: {avg_belief:.3f}',
                transform=fig.transFigure,
                fontsize=18,
                ha='left',
                va='top',
                color='#424242',
                family='sans-serif',
                zorder=1001
            )
            
            fig.text(
                0.03, 0.875,
                f'Vaccinated: {num_vaccinated} / {num_agents}  ({vacc_rate*100:.1f}%)',
                transform=fig.transFigure,
                fontsize=18,
                ha='left',
                va='top',
                color='#424242',
                family='sans-serif',
                zorder=1001
            )
        
        # 保存图像
        filename = os.path.join(frames_dir, f"network_step_{step:03d}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
    
    print(f"   ✓ 完成! {num_steps} 帧已保存")


def plot_influence_scatter(simulation_dir, output_dir):
    """
    生成影响力散点图（使用所有10次运行的对话数据）
    分析belief disparity vs belief change的关系
    """
    print("\n📊 生成影响力散点图...")
    
    all_plot_data = []
    total_valid = 0
    total_invalid = 0
    
    # 收集所有运行的对话数据
    for run_idx in range(1, 11):
        run_dir = os.path.join(simulation_dir, f"run_{run_idx:02d}")
        dialogues_file = os.path.join(run_dir, "all_dialogues.json")
        
        if not os.path.exists(dialogues_file):
            continue
        
        with open(dialogues_file, 'r') as f:
            dialogues = json.load(f)
        
        for d in dialogues:
            # 只处理有效的对话
            if not d.get('is_valid', True):
                total_invalid += 1
                continue
            
            if d.get('elicited_self_score') is None:
                total_invalid += 1
                continue
            
            belief_self = d['initial_beliefs']['self']
            belief_neighbor = d['initial_beliefs']['neighbor']
            belief_disparity = belief_neighbor - belief_self
            belief_change = d['elicited_self_score'] - belief_self
            
            all_plot_data.append({
                'belief_disparity': belief_disparity,
                'belief_change': belief_change
            })
            total_valid += 1
    
    if len(all_plot_data) == 0:
        print("   ✗ 没有找到有效的对话数据")
        return
    
    df = pd.DataFrame(all_plot_data)
    print(f"   ✓ 有效对话: {total_valid}, 无效: {total_invalid}")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
    
    # 绘制散点图
    scatter = ax.scatter(df['belief_disparity'], df['belief_change'], 
                       alpha=0.7,
                       s=40,
                       c=df['belief_change'],
                       cmap='RdYlGn',
                       edgecolors='gray',
                       linewidth=0.8,
                       vmin=-1, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Belief Change', rotation=270, labelpad=20, fontsize=12, fontweight='600')
    
    # 添加回归线和置信区间
    sns.regplot(data=df, x='belief_disparity', y='belief_change',
               scatter=False, ax=ax,
               line_kws={'color': COLORS['primary'], 'linewidth': 2.5, 'label': 'Regression Line'},
               ci=95)
    
    # 计算相关系数
    correlation = df['belief_disparity'].corr(df['belief_change'])
    
    # 添加参考线
    ax.axhline(0, color=COLORS['neutral'], linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axvline(0, color=COLORS['neutral'], linestyle='--', linewidth=1.5, alpha=0.6)
    
    # 添加象限标签
    ax.text(0.95, 0.95, 'Positive Influence\n(Higher → Higher)', 
           transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.text(0.05, 0.05, 'Negative Influence\n(Lower → Lower)', 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 设置标题和标签
    ax.set_title('Influence Dynamics: Belief Disparity vs. Belief Change', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Belief Disparity (Neighbor\'s - Self\'s Belief)', 
                 fontsize=14, fontweight='600')
    ax.set_ylabel('Belief Change After Dialogue', 
                 fontsize=14, fontweight='600')
    
    # 添加相关系数注释
    ax.text(0.05, 0.95, f'Pearson r = {correlation:.3f}\nn = {len(all_plot_data)} valid dialogues', 
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor=COLORS['primary'], linewidth=2, alpha=0.9))
    
    # 美化网格
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    
    # 图例
    ax.legend(loc='upper left', frameon=True, shadow=True, fancybox=True)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "influence_scatter.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ 保存: {os.path.basename(output_file)}")


def main():
    parser = argparse.ArgumentParser(
        description="批量运行结果可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visualize_batch_results.py simulation_20260115_120020
  python visualize_batch_results.py data/output/simulation_20260115_120020
        """
    )
    
    parser.add_argument(
        'simulation_dir',
        help='模拟输出目录名称或完整路径'
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        default=10,
        help='批量运行次数（默认: 10）'
    )
    
    args = parser.parse_args()
    
    # 处理目录路径
    if os.path.isabs(args.simulation_dir):
        simulation_dir = args.simulation_dir
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        simulation_dir = os.path.join(base_dir, "data", "output", args.simulation_dir)
    
    if not os.path.exists(simulation_dir):
        print(f"❌ 错误: 找不到目录 {simulation_dir}")
        return
    
    print("\n" + "="*60)
    print("🎨 批量运行结果可视化")
    print("="*60)
    print(f"📁 模拟目录: {simulation_dir}")
    print(f"🔢 运行次数: {args.num_runs}")
    
    # 创建可视化输出目录
    viz_dir = os.path.join(simulation_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. 加载批量数据
    all_profiles, network_data = load_batch_data(simulation_dir, args.num_runs)
    
    if not all_profiles:
        print("❌ 错误: 无法加载agent profiles数据")
        return
    
    if network_data is None:
        print("❌ 错误: 无法加载network数据")
        return
    
    # 2. 计算平均belief
    avg_beliefs_llm, avg_beliefs_vader = compute_average_beliefs(all_profiles)
    
    # 3. 生成可视化
    visualize_comparative_trends(simulation_dir, viz_dir)
    visualize_belief_distributions(all_profiles, viz_dir)
    generate_network_evolution(simulation_dir, network_data, avg_beliefs_llm, viz_dir)
    plot_influence_scatter(simulation_dir, viz_dir)
    
    print("\n" + "="*60)
    print("✅ 所有可视化已完成!")
    print(f"📁 输出目录: {viz_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
