# LLM-Driven ABM Framework with Multi-Round Communication

## Overview
This repository contains an agent-based simulation where LLM-powered agents conduct multi-round
dialogues about vaccination attitudes over a social network. Agents exchange opinions, update
beliefs, and the framework supports batch runs with scenario comparisons.

## Core Features
- Multi-round dialogue protocol with post-dialogue opinion elicitation.
- Scenario comparison: baseline, memory, prompt diversity, and combined.
- Batch runs with mean and standard deviation metrics.
- Social-network-based interaction with profile visibility rules.

## Inputs
Required data files in src_v2/data/input/:
- workplace_36013030400w1_extended_population.csv
- workplace_36013030400w1_extended_network.csv

## Configuration
Edit src_v2/config.py:
- MAX_STEPS, BATCH_RUNS, MAX_DIALOGS_PER_MICROSTEP
- BELIEF_DISTRIBUTION_TYPE, BELIEF_MEANS, BELIEF_STD
- API_URL, API_KEY, MODEL_NAME
- SCENARIO_MODE

## Run
From the repo root:
```bash
cd src_v2
pip install -r requirements.txt
python main.py
```

## Outputs
Outputs are written under src_v2/data/output/simulation_<timestamp>/. Each run contains:
- model_data.csv, agent_data.csv, agent_profiles.json
- all_dialogues.json, network_data.json
- most_impactful_dialogues_report.txt

## Optional Visualization
Visualization scripts are in src_v2/visualize_batch_results.py and src_v2/analysis.py.
If you do not need charts, you can skip or remove those scripts and their calls in src_v2/main.py.

## Project Structure
```
src_v2/
├── main.py
├── model.py
├── agent.py
├── tools.py
├── config.py
├── analysis.py
├── visualize_batch_results.py
├── requirements.txt
└── data/
	└── input/
```

## Requirements
- Python 3.8+
- See src_v2/requirements.txt for dependencies

