# model.py

"""
Defines the main VaxSimulationModel for the agent-based simulation.
This version uses a manual data collection method to bypass async/sync issues,
guaranteeing correct multi-step agent data logging.
"""

import mesa
import asyncio
import numpy as np
import os
import json
import aiohttp
import random
import pandas as pd
import networkx as nx
from tqdm import tqdm

from agent import VaxAgent
from tools import generate_profiles, create_network, convert_numpy_types, load_workplace_profiles
from config import MAX_DIALOGS_PER_MICROSTEP


class VaxSimulationModel(mesa.Model):
    """A model for simulating vaccination attitude changes through multi-round dialogues."""

    def __init__(
        self,
        max_steps: int = 3,
        agent_alpha: float = 0.7,
        api_url: str = "http://10.13.12.164:7890/v1/chat/completions",
        api_key: str = "abc123",
        model_name: str = "Qwen/Qwen3-8B",
        max_concurrent: int = 3,
        use_workplace_data: bool = True,
        population_csv: str = None,
        network_csv: str = None,
        fixed_initial_beliefs: list = None,  # 固定的初始belief列表
        memory_enabled: bool = False,         # Memory System 开关
        prompt_diversity_enabled: bool = False  # Prompt Diversity 开关
    ):
        super().__init__()
        self.max_steps = max_steps
        self.agent_alpha = agent_alpha
        self.running = True
        self.use_workplace_data = use_workplace_data
        self.population_csv = population_csv
        self.network_csv = network_csv
        self.fixed_initial_beliefs = fixed_initial_beliefs  # 保存固定初始belief
        self.memory_enabled = memory_enabled                 # Memory System 开关
        self.prompt_diversity_enabled = prompt_diversity_enabled  # Prompt Diversity 开关

        self.schedule = mesa.time.BaseScheduler(self)
        self.network = None
        self.network_layout = None  # 保存网络节点布局位置
        self.network_edges = None   # 保存网络边列表
        
        # 初始化 API 调用进度条
        self.api_call_counter = None

        print("Initializing simulation model...")
        print(f"  Memory System: {'ON' if self.memory_enabled else 'OFF'}")
        print(f"  Prompt Diversity: {'ON' if self.prompt_diversity_enabled else 'OFF'}")
        asyncio.run(self.async_init())

        # ** FINAL FIX: Use DataCollector ONLY for model-level reporters. **
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Vaccination_Rate": lambda m: np.mean([a.is_vaccinated for a in m.schedule.agents]),
                "Average_Belief_LLM": lambda m: np.mean([a.belief for a in m.schedule.agents]),
                "Average_Belief_VADER": lambda m: np.mean([a.belief_vader for a in m.schedule.agents]),
                "Belief_Std_Dev_LLM": lambda m: np.std([a.belief for a in m.schedule.agents]),
                "Belief_Std_Dev_VADER": lambda m: np.std([a.belief_vader for a in m.schedule.agents]),  # 新增
            }
        )
        # ** FINAL FIX: Create a simple list to manually store agent data. **
        self.agent_data_log = []

        print("Model initialization complete.")

    async def async_init(self):
        """Asynchronous initialization - loads workplace data from CSV files."""
        
        if not (self.use_workplace_data and self.population_csv and self.network_csv):
            raise ValueError("Workplace data is required. Please provide population_csv and network_csv paths.")
        
        # Load profiles and network from CSV files
        print("Loading workplace data from CSV files...")
        profiles, network, reindex_mapping = load_workplace_profiles(
            self.population_csv, 
            self.network_csv
        )
        
        # Set n_agents to match the actual number of workplace members
        self.n_agents = len(profiles)
        print(f"✓ Using {self.n_agents} agents from workplace data")
        
        # Create agents
        for i in range(self.n_agents):
            self.schedule.add(VaxAgent(i, self, profiles[i], alpha=self.agent_alpha))
        
        # Use the loaded network directly
        self.network = network
        
        # 设置 5% 的智能体为高信念的医护人员/工作人员（但未接种）
        # 使用固定seed确保所有scenario和batch run选中相同的agent，保证对比公平性
        n_high_belief = max(1, int(self.n_agents * 0.05))
        high_belief_rng = random.Random(42)
        high_belief_ids = high_belief_rng.sample(range(self.n_agents), n_high_belief)
        
        for agent_id in high_belief_ids:
            agent = self.schedule.agents[agent_id]
            agent.belief = 1.0  # 强烈支持疫苗
            agent.belief_vader = 1.0
            # 注意：不设置 is_vaccinated = True，让他们在后续步骤自然接种
            print(f"   High-belief Agent {agent_id} (Healthcare worker, not yet vaccinated)")
        
        print(f"   Initialized {n_high_belief} high-belief agents ({n_high_belief/self.n_agents*100:.1f}%)")
        
        # 保存网络布局和边列表，供可视化使用
        self.network_layout = nx.spring_layout(self.network, seed=42, k=0.5, iterations=50)
        self.network_edges = list(self.network.edges())
        print(f"   Network layout computed and saved for visualization.")

    async def async_actions(self):
        """
        Micro-step based dialog scheduler with strict locking.
        Implements exclusive dyadic dialogs with bounded concurrency.
        """
        async with aiohttp.ClientSession() as session:
            # 生成所有可能的对话边（有向）
            # 已接种的agent不发起对话，但可以作为neighbor被对话
            all_edges = []
            for agent in self.schedule.agents:
                # 已接种的agent不发起对话
                if agent.is_vaccinated:
                    continue
                    
                # 未接种的agent可以和所有neighbors对话（无论neighbor是否接种）
                neighbors = agent.get_neighbors()
                for neighbor in neighbors:
                    all_edges.append((agent.unique_id, neighbor.unique_id))
            
            total_edges = len(all_edges)
            
            # Initialize progress tracking
            total_estimated_api_calls = total_edges * 5  # Upper bound estimate
            self.api_call_counter = tqdm(
                total=total_estimated_api_calls,
                desc=f"  📡 API Calls (Step {self.schedule.steps + 1})",
                unit="call",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            
            # Track which edges have been processed in this step
            processed_edges = set()
            micro_step = 0
            
            print(f"\n   Starting micro-step scheduling for {total_edges} possible dialogs...")
            
            # Keep running micro-steps until all edges are processed
            while len(processed_edges) < total_edges:
                micro_step += 1
                
                # STEP 1: Get unprocessed edges
                remaining_edges = [e for e in all_edges if e not in processed_edges]
                
                if not remaining_edges:
                    break
                
                # STEP 2: Randomly shuffle (for reproducibility, use model's RNG)
                random.shuffle(remaining_edges)
                
                # STEP 3: Select up to MAX_DIALOGS_PER_MICROSTEP non-conflicting dialogs
                locked_agents = set()
                selected_dialogs = []
                
                for i, j in remaining_edges:
                    # Check if both agents are available
                    agent_i = self.schedule.agents[i]
                    agent_j = self.schedule.agents[j]
                    
                    if i not in locked_agents and j not in locked_agents:
                        # Lock both agents
                        agent_i.is_locked = True
                        agent_j.is_locked = True
                        locked_agents.add(i)
                        locked_agents.add(j)
                        
                        selected_dialogs.append((i, j, agent_i, agent_j))
                        processed_edges.add((i, j))
                        
                        # Stop if we have MAX_DIALOGS_PER_MICROSTEP concurrent dialogs
                        if len(selected_dialogs) >= MAX_DIALOGS_PER_MICROSTEP:
                            break
                
                if not selected_dialogs:
                    # No valid dialogs possible in this micro-step
                    break
                
                print(f"     Micro-step {micro_step}: {len(selected_dialogs)} dialogs scheduled")
                
                # STEP 4: Execute selected dialogs in parallel
                dialog_tasks = [
                    self._execute_exclusive_dialog(session, agent_i, agent_j)
                    for (i, j, agent_i, agent_j) in selected_dialogs
                ]
                
                dialog_results = await asyncio.gather(*dialog_tasks, return_exceptions=True)
                
                # STEP 5: Commit state updates and release locks
                for idx, ((i, j, agent_i, agent_j), result) in enumerate(zip(selected_dialogs, dialog_results)):
                    if isinstance(result, Exception):
                        print(f"       ⚠️  Dialog ({i}, {j}) failed: {result}")
                    else:
                        dialogue_record, belief_change_i, belief_change_j = result
                        
                        # IMMEDIATE UPDATE: Apply belief changes (only for non-vaccinated agents)
                        if not agent_i.is_vaccinated:
                            agent_i.belief = np.clip(agent_i.belief + self.agent_alpha * belief_change_i, -1.0, 1.0)
                        if not agent_j.is_vaccinated:
                            agent_j.belief = np.clip(agent_j.belief + self.agent_alpha * belief_change_j, -1.0, 1.0)
                        
                        # Update dialog memory
                        if j not in agent_i.dialog_memory:
                            agent_i.dialog_memory[j] = []
                        if i not in agent_j.dialog_memory:
                            agent_j.dialog_memory[i] = []
                        
                        agent_i.dialog_memory[j].append(dialogue_record)
                        agent_j.dialog_memory[i].append(dialogue_record)
                        
                        # Log dialogue
                        agent_i.dialogue_history.append(dialogue_record)
                    
                    # RELEASE LOCKS
                    agent_i.is_locked = False
                    agent_j.is_locked = False
            
            print(f"   Completed {micro_step} micro-steps, processed {len(processed_edges)}/{total_edges} edges")
            
            # Close progress bar
            self.api_call_counter.close()
            self.api_call_counter = None
            
        # Run vaccination decision
        for agent in self.schedule.agents:
            agent.step()
        
        # Apply vaccination belief updates through advance()
        for agent in self.schedule.agents:
            agent.advance()
    
    async def _execute_exclusive_dialog(self, session: aiohttp.ClientSession, agent_i: VaxAgent, agent_j: VaxAgent):
        """
        Execute a single exclusive dyadic dialog.
        Returns: (dialogue_record, belief_change_i, belief_change_j)
        """
        # Call the agent's dialog method (we'll modify it to be simpler)
        dialogue_record = await agent_i.conduct_dialogue_with_neighbor(session, agent_j)
        
        # Calculate belief changes from this dialog
        if dialogue_record.get('is_valid', False):
            belief_change_i = dialogue_record['elicited_self_score'] - agent_i.belief
            # For agent_j, we need to infer their change (or they could also reflect)
            # For simplicity, assume symmetric influence
            belief_change_j = dialogue_record['elicited_sentiment_score'] - agent_j.belief
        else:
            belief_change_i = 0.0
            belief_change_j = 0.0
        
        return dialogue_record, belief_change_i, belief_change_j

    def collect_agent_data(self):
        """
        ** FINAL FIX: A dedicated manual function to log agent data. **
        This function is guaranteed to be called at the right time.
        """
        for agent in self.schedule.agents:
            self.agent_data_log.append({
                "Step": self.schedule.steps,
                "AgentID": agent.unique_id,
                "Belief_LLM": agent.belief,
                "Belief_VADER": agent.belief_vader,
                "Is_Vaccinated": agent.is_vaccinated
            })

    def step(self):
        """Executes one step of the simulation."""
        print(f"\n--- Running Step {self.schedule.steps + 1}/{self.max_steps} ---")

        asyncio.run(self.async_actions())

        # Collect model-level data using the standard datacollector
        self.datacollector.collect(self)
        # Manually collect agent-level data
        self.collect_agent_data()

        self.schedule.steps += 1  # Manually advance step counter

        print(f"Step {self.schedule.steps} Summary:")
        print(f"  Avg Belief (LLM): {self.datacollector.model_vars['Average_Belief_LLM'][-1]:.3f}")
        print(f"  Avg Belief (VADER): {self.datacollector.model_vars['Average_Belief_VADER'][-1]:.3f}")

    def run_model(self):
        """Run the simulation for the configured number of steps."""
        print(f"\n🚀 Starting simulation for {self.max_steps} steps...")

        # Collect initial state (Step 0) for both model and agents
        self.datacollector.collect(self)
        self.collect_agent_data()

        for i in range(self.max_steps):
            self.step()
        print("\n✅ Simulation finished!")

    def export_results(self, output_dir: str):
        """Export all simulation data to the specified output directory."""
        print(f"\n💾 Exporting results to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        # Export model data as before
        self.datacollector.get_model_vars_dataframe().to_csv(os.path.join(output_dir, "model_data.csv"))

        # ** FINAL FIX: Convert our manual log into a DataFrame and save it. **
        agent_df = pd.DataFrame(self.agent_data_log)
        agent_df.to_csv(os.path.join(output_dir, "agent_data.csv"), index=False)

        # The rest of the exports remain the same
        all_dialogues = sorted(
            [dialogue for agent in self.schedule.agents for dialogue in agent.dialogue_history],
            key=lambda d: (d.get('tick', 0), tuple(d.get('interlocutors', (0, 0))))
        )
        with open(os.path.join(output_dir, "all_dialogues.json"), 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(all_dialogues), f, indent=2, ensure_ascii=False)

        profiles = [{
            'agent_id': agent.unique_id, 'profile': agent.profile, 'age': agent.age,
            'belief_history': agent.belief_history, 'belief_vader_history': agent.belief_vader_history,
            'is_vaccinated': agent.is_vaccinated, 'tick_vaccinated': agent.tick_vaccinated
        } for agent in self.schedule.agents]
        with open(os.path.join(output_dir, "agent_profiles.json"), 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
        
        # ** 新增：保存真实的网络数据 **
        network_data = {
            'layout': {str(k): [float(v[0]), float(v[1])] for k, v in self.network_layout.items()},
            'edges': [[int(e[0]), int(e[1])] for e in self.network_edges],
            'network_type': 'workplace_extended',
            'num_agents': self.n_agents
        }
        with open(os.path.join(output_dir, "network_data.json"), 'w', encoding='utf-8') as f:
            json.dump(network_data, f, indent=2, ensure_ascii=False)
        
        # ** 新增：保存场景配置元数据 **
        scenario_metadata = {
            'memory_enabled': self.memory_enabled,
            'prompt_diversity_enabled': self.prompt_diversity_enabled,
            'max_steps': self.max_steps,
            'agent_alpha': self.agent_alpha,
            'n_agents': self.n_agents,
        }
        with open(os.path.join(output_dir, "scenario_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(scenario_metadata, f, indent=2, ensure_ascii=False)
        
        # ** 新增：保存 Memory System 数据（如果启用）**
        if self.memory_enabled:
            memory_data = {}
            for agent in self.schedule.agents:
                memory_data[str(agent.unique_id)] = {
                    'episodic_memory': agent.episodic_memory,
                    'n_episodes': len(agent.episodic_memory),
                }
            with open(os.path.join(output_dir, "memory_data.json"), 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_types(memory_data), f, indent=2, ensure_ascii=False)
        
        print(" -> Export complete (including network topology).")