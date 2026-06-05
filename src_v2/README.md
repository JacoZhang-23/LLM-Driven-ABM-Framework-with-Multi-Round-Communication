# Multi-Round LLM for Opinion Dynamics

## Overview
This project implements a multi-agent simulation where LLM-powered agents engage in multi-round
dialogues about vaccination attitudes. Agents interact through a social network, exchange opinions,
and update beliefs across simulation steps. The system supports batch runs with aggregated
statistics and scenario comparisons.

## Inputs
Required data files in data/input/:
- workplace_36013030400w1_extended_population.csv
- workplace_36013030400w1_extended_network.csv

## Configuration
Edit config.py:
- MAX_STEPS, BATCH_RUNS, MAX_DIALOGS_PER_MICROSTEP
- BELIEF_DISTRIBUTION_TYPE, BELIEF_MEANS, BELIEF_STD
- API_URL, API_KEY, MODEL_NAME
- SCENARIO_MODE

## Multi-Round Dialogue Protocol
Each conversation uses 4 rounds followed by a reflection step:
1. Person B initiates (2-3 sentences)
2. Person A responds (2-3 sentences)
3. Person B continues (2-3 sentences)
4. Person A continues (2-3 sentences)
5. Person A reflects and reports updated opinion score

## Run
```bash
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
- See requirements.txt for dependencies
