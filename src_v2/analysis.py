# analysis.py

"""
Advanced analysis and visualization module for the LLM Vaccination Simulation.
This module focuses on process analysis and causal exploration.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
import json
import networkx as nx
from typing import Dict, List

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# 自定义配色方案
COLORS = {
    'primary': '#2E86AB',      # 深蓝色
    'secondary': '#A23B72',    # 紫红色
    'accent': '#F18F01',       # 橙色
    'positive': '#06A77D',     # 绿色
    'negative': '#D62246',     # 红色
    'neutral': '#6C757D',      # 灰色
    'palette': ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D62246', '#8338EC']
}


def classify_agent_group(profile_text: str) -> str:
    """A simple classifier to group agents based on keywords in their profile."""
    profile_lower = profile_text.lower()
    if any(keyword in profile_lower for keyword in ['nurse', 'doctor', 'medical', 'health']):
        return 'Medical Field'
    if any(keyword in profile_lower for keyword in ['engineer', 'tech', 'software', 'data']):
        return 'Tech/Engineering'
    if any(keyword in profile_lower for keyword in ['teacher', 'professor', 'education']):
        return 'Education'
    if any(keyword in profile_lower for keyword in ['retired', 'retiree']):
        return 'Retired'
    return 'General/Other'


def plot_comprehensive_trends(output_dir: str):
    """
    综合趋势图：同时展示 LLM belief, VADER belief, 和 Vaccination Rate 的演化
    不包含置信区间，Y轴范围调整为-0.5到1，右侧Y轴与左侧起点对齐
    """
    print("ANALYSIS: Generating comprehensive trends plot (LLM + VADER + Vaccination)...")
    try:
        model_df = pd.read_csv(os.path.join(output_dir, 'model_data.csv'))
        
        # 创建图表 - 使用双 Y 轴
        fig, ax1 = plt.subplots(figsize=(14, 8), dpi=300)
        
        # 左侧 Y 轴：Belief Scores (-0.5 to 1)
        ax1.set_xlabel('Simulation Step', fontsize=14, fontweight='600')
        ax1.set_ylabel('Average Belief Score', fontsize=14, fontweight='600', color='black')
        ax1.set_ylim(-0.5, 1.0)
        ax1.tick_params(axis='y', labelcolor='black')
        
        steps = model_df.index
        
        # === 1. 绘制 LLM Belief（不带置信区间）===
        llm_mean = model_df['Average_Belief_LLM']
        
        ax1.plot(steps, llm_mean, 
                marker='o', markersize=8, linewidth=2.5,
                color=COLORS['primary'], label='LLM Belief (Mean)', 
                alpha=0.9, zorder=3)
        
        # === 2. 绘制 VADER Belief（不带置信区间）===
        vader_mean = model_df['Average_Belief_VADER']
        
        ax1.plot(steps, vader_mean, 
                marker='s', markersize=8, linewidth=2.5,
                color=COLORS['secondary'], label='VADER Belief (Mean)', 
                alpha=0.9, zorder=3)
        
        # 添加参考线（在 belief 轴上）
        ax1.axhline(y=0, color=COLORS['neutral'], linestyle='--', 
                   linewidth=1.5, alpha=0.5, zorder=2)
        ax1.axhline(y=0.5, color=COLORS['positive'], linestyle=':', 
                   linewidth=1, alpha=0.3, zorder=2)
        ax1.axhline(y=-0.5, color=COLORS['negative'], linestyle=':', 
                   linewidth=1, alpha=0.3, zorder=2)
        
        # === 3. 右侧 Y 轴：Vaccination Rate (对齐左轴起点) ===
        ax2 = ax1.twinx()
        ax2.set_ylabel('Vaccination Rate', fontsize=14, fontweight='600', color=COLORS['positive'])
        # 右轴范围设置为与左轴相同，确保起点对齐
        ax2.set_ylim(-0.5, 1.0)
        ax2.tick_params(axis='y', labelcolor=COLORS['positive'])
        
        vacc_rate = model_df['Vaccination_Rate']
        ax2.plot(steps, vacc_rate, 
                marker='^', markersize=8, linewidth=2.5,
                color=COLORS['positive'], label='Vaccination Rate', 
                alpha=0.9, linestyle='--', zorder=3)
        
        # === 4. 设置标题和图例 ===
        ax1.set_title('Comprehensive Trends: Belief Evolution & Vaccination Progress', 
                     fontsize=18, fontweight='bold', pad=20)
        
        # 合并两个轴的图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                  loc='upper left', fontsize=11, framealpha=0.95,
                  edgecolor='gray', fancybox=True)
        
        # === 5. 美化网格和布局 ===
        ax1.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.4, zorder=0)
        ax1.set_xlim(-0.2, steps.max() + 0.2)
        
        plt.tight_layout()
        
        # 保存图表
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        plt.savefig(os.path.join(viz_dir, "comprehensive_trends.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" -> ✓ Saved 'comprehensive_trends.png'")
        print(f"   LLM Belief: {llm_mean.iloc[0]:.3f} → {llm_mean.iloc[-1]:.3f}")
        print(f"   VADER Belief: {vader_mean.iloc[0]:.3f} → {vader_mean.iloc[-1]:.3f}")
        print(f"   Vaccination: {vacc_rate.iloc[0]:.1%} → {vacc_rate.iloc[-1]:.1%}")
        
    except Exception as e:
        print(f" -> ✗ Error: {e}")


def plot_influence_scatter(output_dir: str):
    """
    Creates a scatter plot to analyze the relationship between belief disparity and belief change.
    Enhanced with better visualization and statistical annotations.
    Only includes valid dialogues (with proper summary).
    """
    print("ANALYSIS: Generating influence scatter plot...")
    try:
        with open(os.path.join(output_dir, 'all_dialogues.json'), 'r') as f:
            dialogues = json.load(f)

        plot_data = []
        invalid_count = 0
        
        for d in dialogues:
            # 只处理有效的对话
            if not d.get('is_valid', True):  # 兼容旧数据，默认为 True
                invalid_count += 1
                continue
                
            if d.get('elicited_self_score') is None:
                invalid_count += 1
                continue
            
            belief_self = d['initial_beliefs']['self']
            belief_neighbor = d['initial_beliefs']['neighbor']

            belief_disparity = belief_neighbor - belief_self
            belief_change = d['elicited_self_score'] - belief_self

            plot_data.append({
                'belief_disparity': belief_disparity,
                'belief_change': belief_change
            })

        if len(plot_data) == 0:
            print(" -> ⚠ No valid dialogues found for scatter plot")
            return

        df = pd.DataFrame(plot_data)
        
        # 打印统计信息
        print(f"   Valid dialogues: {len(plot_data)}, Invalid: {invalid_count}")

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
        
        # 创建密度散点图
        from scipy.stats import gaussian_kde
        
        # 绘制散点 - 增强颜色饱和度
        scatter = ax.scatter(df['belief_disparity'], df['belief_change'], 
                           alpha=0.7,  # 从 0.4 提高到 0.7，增强不透明度
                           s=80,  # 从 60 提高到 80，稍微增大点的大小
                           c=df['belief_change'],
                           cmap='RdYlGn',  # 红-黄-绿渐变（红=负，绿=正）
                           edgecolors='gray',  # 从 white 改为 gray，边框更明显
                           linewidth=0.8,  # 从 0.5 提高到 0.8
                           vmin=-1, vmax=1)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Belief Change', rotation=270, labelpad=20, fontsize=12)
        
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
        ax.text(0.05, 0.95, f'Pearson r = {correlation:.3f}\nn = {len(plot_data)} valid dialogues', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor=COLORS['primary'], linewidth=2, alpha=0.9))
        
        # 美化网格
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
        ax.set_axisbelow(True)
        
        # 图例
        ax.legend(loc='upper left', frameon=True, shadow=True, fancybox=True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        viz_dir = os.path.join(output_dir, "visualizations")
        plt.savefig(os.path.join(viz_dir, "influence_scatter_plot.png"), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(" -> ✓ Saved 'influence_scatter_plot.png'")

    except Exception as e:
        print(f" -> ✗ Failed to generate influence scatter plot: {e}")
        import traceback
        traceback.print_exc()


def generate_impactful_dialogues_report(output_dir: str, top_n: int = 3):
    """
    Finds the most persuasive dialogues and saves them to a text report.
    Enhanced with better formatting and statistics.
    Only includes valid dialogues with proper summaries.
    """
    print("ANALYSIS: Generating impactful dialogues report...")
    try:
        with open(os.path.join(output_dir, 'all_dialogues.json'), 'r') as f:
            dialogues = json.load(f)

        # 只保留有效对话
        valid_dialogues = []
        invalid_count = 0
        
        for d in dialogues:
            if d.get('is_valid', True) and d.get('elicited_self_score') is not None:
                d['belief_change'] = d['elicited_self_score'] - d['initial_beliefs']['self']
                valid_dialogues.append(d)
            else:
                invalid_count += 1

        if len(valid_dialogues) == 0:
            print(" -> ⚠ No valid dialogues found for report")
            return

        # Sort by the absolute magnitude of change to find the most impactful
        sorted_dialogues = sorted(valid_dialogues, key=lambda x: abs(x['belief_change']), reverse=True)

        # 计算统计信息
        belief_changes = [d['belief_change'] for d in valid_dialogues]
        avg_change = np.mean(belief_changes)
        std_change = np.std(belief_changes)
        max_change = np.max(belief_changes)
        min_change = np.min(belief_changes)

        report_path = os.path.join(output_dir, "most_impactful_dialogues_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("           MOST IMPACTFUL DIALOGUES ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # 添加统计摘要
            f.write("📊 STATISTICAL SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Dialogues: {len(dialogues)}\n")
            f.write(f"Valid Dialogues Analyzed: {len(valid_dialogues)}\n")
            f.write(f"Invalid Dialogues (No Summary): {invalid_count}\n")
            f.write(f"Average Belief Change: {avg_change:.4f}\n")
            f.write(f"Std Dev of Change: {std_change:.4f}\n")
            f.write(f"Maximum Positive Change: {max_change:.4f}\n")
            f.write(f"Maximum Negative Change: {min_change:.4f}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            f.write(f"🎯 TOP {top_n} MOST IMPACTFUL DIALOGUES (by magnitude of belief change)\n")
            f.write("=" * 80 + "\n\n")

            for i, d in enumerate(sorted_dialogues[:top_n], 1):
                self_id = d['interlocutors'][0]
                neighbor_id = d['interlocutors'][1]
                
                # 判断影响方向
                direction = "↑ Positive" if d['belief_change'] > 0 else "↓ Negative"
                magnitude = "Strong" if abs(d['belief_change']) > 0.5 else "Moderate"

                f.write(f"┌{'─' * 78}┐\n")
                f.write(f"│ DIALOGUE #{i} - {magnitude} {direction} Influence │\n")
                f.write(f"└{'─' * 78}┘\n\n")
                
                f.write(f"⏱  Simulation Tick: {d['tick']}\n")
                f.write(f"👥 Participants:\n")
                f.write(f"   • Agent {self_id} (Listener/Influenced)\n")
                f.write(f"   • Agent {neighbor_id} (Speaker/Influencer)\n\n")
                
                f.write(f"📈 Belief Metrics:\n")
                f.write(f"   • Initial Belief (Self):     {d['initial_beliefs']['self']:>6.3f}\n")
                f.write(f"   • Initial Belief (Neighbor): {d['initial_beliefs']['neighbor']:>6.3f}\n")
                f.write(f"   • Final Belief (Self):       {d['elicited_self_score']:>6.3f}\n")
                f.write(f"   • Belief Change:             {d['belief_change']:>+6.3f} ({direction})\n")
                f.write(f"   • Change Magnitude:          {abs(d['belief_change']):>6.3f}\n\n")

                f.write("💬 DIALOGUE TRANSCRIPT\n")
                f.write("-" * 80 + "\n")
                for j, exchange in enumerate(d['exchanges'], 1):
                    speaker = f"Agent {exchange['speaker_id']}"
                    role = "(Listener)" if exchange['speaker_id'] == self_id else "(Speaker)"
                    f.write(f"[Turn {j}] {speaker} {role}:\n")
                    f.write(f"{exchange['message']}\n\n")

                f.write("💭 POST-DIALOGUE REFLECTION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Agent {self_id}'s Summary: \"{d['elicited_summary']}\"\n")
                f.write(f"Self-Reported Score: {d['elicited_self_score']:.3f}\n\n")
                
                f.write("=" * 80 + "\n\n")

        print(f" -> ✓ Report saved to '{report_path}' ({len(valid_dialogues)} valid dialogues)")

    except Exception as e:
        print(f" -> ✗ Failed to generate impactful dialogues report: {e}")
        import traceback
        traceback.print_exc()


def run_all_analyses(output_dir: str, num_agents: int):
    """A wrapper function to run all advanced analyses."""
    print("\n🔬 Running Advanced Analyses...")
    plot_comprehensive_trends(output_dir)  # 新的综合趋势图
    plot_influence_scatter(output_dir)
    generate_impactful_dialogues_report(output_dir, top_n=3)


def visualize_network_evolution(output_dir: str, network_layout: Dict, network_edges: List, agent_profiles: List[Dict]):
    """
    生成逐帧网络可视化，展示 belief_LLM 随时间的变化。
    使用红色（反对）到白色（中立）到绿色（支持）的渐变色表示信念。
    
    Args:
        output_dir: 输出目录
        network_layout: 节点布局位置字典 {agent_id: (x, y)}
        network_edges: 边列表 [(agent_i, agent_j), ...]
        agent_profiles: 包含每个 agent 的 belief_history 的档案数据
    """
    print("\n🌐 Generating Network Evolution Visualization...")
    
    try:
        viz_dir = os.path.join(output_dir, "visualizations", "network_frames")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 提取所有 agent 的 belief 历史
        belief_histories = {
            agent['agent_id']: agent['belief_history'] 
            for agent in agent_profiles
        }
        
        # 确定总步数
        max_steps = max(len(hist) for hist in belief_histories.values())
        print(f"   Total steps to visualize: {max_steps}")
        
        # 创建自定义颜色映射：深红色(-1) → 白色(0) → 深绿色(+1)
        colors_list = ['#C62828', '#EF5350', '#FFCDD2', '#FFFFFF', '#C8E6C9', '#66BB6A', '#2E7D32']
        n_bins = 256
        cmap = mcolors.LinearSegmentedColormap.from_list('belief_cmap', colors_list, N=n_bins)
        
        # 为每个步骤生成一张图
        for step in range(max_steps):
            fig = plt.figure(figsize=(18, 12), dpi=150)
            
            # 创建主绘图区域（留出右侧空间给颜色条）
            ax = fig.add_axes([0.05, 0.05, 0.80, 0.90])
            
            # 获取当前步骤所有 agent 的 belief
            current_beliefs = {}
            for agent_id, hist in belief_histories.items():
                if step < len(hist):
                    current_beliefs[agent_id] = hist[step]
                else:
                    current_beliefs[agent_id] = hist[-1]  # 使用最后一个值
            
            # 准备节点颜色（根据 belief 映射到颜色）
            node_colors = [cmap((current_beliefs[node] + 1) / 2) for node in sorted(current_beliefs.keys())]
            
            # 创建 NetworkX 图（用于绘制）
            G = nx.Graph()
            G.add_nodes_from(current_beliefs.keys())
            G.add_edges_from(network_edges)
            
            # 绘制边（浅灰色，细线，带透明度）
            nx.draw_networkx_edges(
                G, network_layout, ax=ax,
                edge_color='#9E9E9E',
                width=2.0,
                alpha=0.35,
                style='solid'
            )
            
            # 绘制节点（带阴影效果）
            # 先绘制阴影层
            nx.draw_networkx_nodes(
                G, network_layout, ax=ax,
                node_color='black',
                node_size=850,
                alpha=0.15
            )
            
            # 再绘制主节点
            nx.draw_networkx_nodes(
                G, network_layout, ax=ax,
                node_color=node_colors,
                node_size=900,
                edgecolors='#212121',
                linewidths=2.5,
                alpha=0.95
            )
            
            # 绘制节点标签（Agent ID）
            labels = {node: f"{node}" for node in current_beliefs.keys()}
            nx.draw_networkx_labels(
                G, network_layout, labels, ax=ax,
                font_size=18,
                font_weight='bold',
                font_color='black',
                font_family='sans-serif'
            )
            
            # 添加标题
            title_text = f'Agent Network - Belief Evolution (Step {step})'
            ax.text(
                0.5, 1.02, title_text,
                transform=ax.transAxes,
                fontsize=36,
                fontweight='bold',
                ha='center',
                va='bottom'
            )
            
            # 移除坐标轴
            ax.axis('off')
            # 自动调整坐标轴范围以包含所有节点
            ax.margins(0.1)  # 添加10%的边距
            ax.set_aspect('equal')
            
            # === 创建颜色条图例（右侧） ===
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.70])
            
            norm = mcolors.Normalize(vmin=-1, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
            cbar.set_label(
                'Belief Score',
                rotation=270,
                labelpad=28,
                fontsize=24,
                fontweight='bold'
            )
            cbar.ax.tick_params(labelsize=18, width=2, length=6)
            
            # 设置颜色条刻度
            cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            cbar.set_ticklabels([
                '-1.0\nStrongly\nOppose',
                '-0.5\nOppose',
                '0.0\nNeutral',
                '+0.5\nSupport',
                '+1.0\nStrongly\nSupport'
            ])
            
            # 添加颜色条边框
            cbar.outline.set_edgecolor('#424242')
            cbar.outline.set_linewidth(2)
            
            # === 添加统计信息框（左上角）===
            avg_belief = np.mean(list(current_beliefs.values()))
            std_belief = np.std(list(current_beliefs.values()))
            min_belief = np.min(list(current_beliefs.values()))
            max_belief = np.max(list(current_beliefs.values()))
            vaccinated_count = sum(1 for b in current_beliefs.values() if b >= 0.99)
            
            info_text = (
                f"Step: {step:2d} / {max_steps-1}\n"
                f"━━━━━━━━━━━━━━━\n"
                f"Avg Belief:  {avg_belief:+.3f}\n"
                f"Std Dev:     {std_belief:6.3f}\n"
                f"Range:       [{min_belief:+.2f}, {max_belief:+.2f}]\n"
                f"━━━━━━━━━━━━━━━\n"
                f"Vaccinated:  {vaccinated_count:2d} / {len(current_beliefs)}\n"
                f"Rate:        {vaccinated_count/len(current_beliefs)*100:5.1f}%"
            )
            
            # === 添加标题（在信息框上方）===
            fig.text(
                0.12, 0.62, 'Network Statistics',  # 从 0.97 降到 0.62（垂直居中）
                fontsize=19,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='center',
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor='#E3F2FD',
                    edgecolor='#1976D2',
                    linewidth=2,
                    alpha=0.95
                ),
                zorder=11
            )
            
            # 创建信息框背景（调整位置和大小）
            info_box = FancyBboxPatch(
                (0.015, 0.38), 0.21, 0.22,  # 从 (0.015, 0.73) 降到 (0.015, 0.38)
                boxstyle="round,pad=0.015",
                transform=fig.transFigure,
                facecolor='white',
                edgecolor='#1976D2',
                linewidth=3,
                alpha=0.97,
                zorder=10
            )
            fig.patches.append(info_box)
            
            # 添加信息文本（调整位置）
            fig.text(
                0.12, 0.58, info_text,  # 从 0.93 降到 0.58
                fontsize=16,
                verticalalignment='top',
                horizontalalignment='center',
                fontfamily='monospace',
                fontweight='500',
                bbox=dict(facecolor='none', edgecolor='none'),
                zorder=11
            )
            
            # 保存图像
            filename = os.path.join(viz_dir, f"network_step_{step:03d}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            # 进度提示（每隔几步或最后一步）
            if step == 0 or step % max(1, max_steps // 5) == 0 or step == max_steps - 1:
                print(f"   -> Frame {step:2d}/{max_steps-1} saved (Avg belief: {avg_belief:+.3f}, Vaccinated: {vaccinated_count}/{len(current_beliefs)})")
        
        print(f"\n ✓ All {max_steps} network frames saved to:")
        print(f"   {viz_dir}")
        print(f"\n💡 To create an animation video (requires ffmpeg):")
        print(f"   cd {viz_dir}")
        print(f"   ffmpeg -framerate 2 -i network_step_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 18 network_evolution.mp4")
        print(f"\n💡 Or create a GIF (requires ImageMagick):")
        print(f"   cd {viz_dir}")
        print(f"   convert -delay 50 -loop 0 network_step_*.png network_evolution.gif")
        
    except Exception as e:
        print(f" -> ✗ Error generating network visualization: {e}")
        import traceback
        traceback.print_exc()