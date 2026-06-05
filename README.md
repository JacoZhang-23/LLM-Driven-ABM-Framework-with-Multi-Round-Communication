# LLM-Driven ABM for Vaccine Opinion Dynamics

## Overview
This project implements an LLM-driven agent-based simulation of vaccination opinion dynamics on a
real-world workplace social network. Agents hold demographic personas, engage in structured
multi-round dialogues, and update beliefs through a post-dialogue reflection step. The framework is
designed to study how conversational content aggregates into macro-level outcomes such as mean
opinion, polarization, and vaccination uptake, and to compare baseline vs memory-augmented agents.

## Framework Summary (Paper-Aligned)
- Network: three relation layers (household hh, workplace wk, social media sm) with different
      profile visibility rules.
- Personas: age, occupation, education, income, and urban/rural attributes are injected into prompts.
- Beliefs: initialized from a truncated normal distribution N(0, 0.3) on [-1, 1], fixed across runs.
- Dialogue: 4 alternating rounds between two agents, then a reflection step for opinion elicitation.
- Update: the elicited score becomes the next-step belief; agents vaccinate when belief > 0.5.
- Memory scenario: agents store episodic records (up to 50) with neighbor stance, opinion shift, and
      a key argument, which are injected into future prompts.

## Inputs
Required data files in the repo root data/input/:
- workplace_36013030400w1_extended_population.csv
- workplace_36013030400w1_extended_network.csv

## Configuration
Edit config.py:
- MAX_STEPS, BATCH_RUNS, MAX_DIALOGS_PER_MICROSTEP
- BELIEF_DISTRIBUTION_TYPE, BELIEF_MEANS, BELIEF_STD
- API_URL, API_KEY, MODEL_NAME
- SCENARIO_MODE (baseline, memory, prompt_diversity, combined, or all)

## Multi-Round Dialogue Protocol
Each conversation uses 4 rounds followed by a reflection step:
1. Person B initiates (2-3 sentences)
2. Person A responds (2-3 sentences)
3. Person B continues (2-3 sentences)
4. Person A continues (2-3 sentences)
5. Person A reflects and reports updated opinion score

## Scenarios
- Baseline: no memory, no prompt diversity.
- Memory: agents inject episodic conversation records into prompts.
- Prompt Diversity: a subset of agents receive persuasion-style constraints.
- Combined: memory + prompt diversity.

## Run
Run from the repo root so paths resolve to data/input:
```bash
cd src_v2
pip install -r requirements.txt
python main.py
```

## Outputs
Output directory: data/output/simulation_<timestamp>/
- model_data.csv, agent_data.csv, agent_profiles.json
- all_dialogues.json, network_data.json
- most_impactful_dialogues_report.txt

## Optional Visualization
Visualization scripts are in visualize_batch_results.py and analysis.py.
If you do not need charts, you can skip or remove those scripts and their calls in main.py.

## Project Structure
```
data/
└── input/
    ├── workplace_36013030400w1_extended_network.csv
    └── workplace_36013030400w1_extended_population.csv
src_v2/
├── main.py
├── model.py
├── agent.py
├── tools.py
├── config.py
├── analysis.py
├── visualize_batch_results.py
└── requirements.txt
```

## Requirements
- Python 3.8+
- See requirements.txt for dependencies
