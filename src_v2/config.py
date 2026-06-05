import os
from Melodie import Config

# API Settings
API_KEY = "abc123"

# API URL Options (uncomment one):
# Option 1: Remote server (original)
# API_URL = "http://10.13.12.164:7899/v1"
# Option 2: Local server (localhost)
API_URL = "http://localhost:7899/v1"
# Option 3: Local server (0.0.0.0)
# API_URL = "http://0.0.0.0:7899/v1"

CHAT_ENDPOINT = f"{API_URL}/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B"

# Concurrency Settings
MAX_DIALOGS_PER_MICROSTEP = 18  # Maximum concurrent dialogs (as per requirements)
MAX_CONCURRENT_CALLS = 50  # Number of concurrent API calls for other operations

# Simulation Parameters
# Note: Using extended workplace network with 95 agents (30 workplace + 65 external)
MAX_STEPS = 10
AGENT_ALPHA = 0.5
BATCH_RUNS = 10  # Number of times to run the entire simulation for statistical robustness

# Belief Distribution Settings (Normal)
# Options: "neutral", "pro", "resist"
BELIEF_DISTRIBUTION_TYPE = "neutral"
BELIEF_MEANS = {
    "neutral": 0.0,
    "pro": 0.4,
    "resist": -0.4,
}
BELIEF_STD = 0.3

# ========== Scenario Configuration ==========
# SCENARIO_MODE controls which scenarios to run:
#   "baseline"                    - Original behavior (no memory, no prompt diversity)
#   "memory"                      - Memory System enabled
#   "prompt_diversity"             - Prompt Diversity enabled
#   "combined"                    - Both Memory + Prompt Diversity enabled
#   "all"                         - Run all four scenarios for comparison
#   "prompt_diversity,combined"   - Run only scenarios 3 & 4 (comma-separated list)
SCENARIO_MODE = "prompt_diversity,combined"

# These flags are set dynamically by the scenario runner; do NOT set manually.
MEMORY_ENABLED = False
PROMPT_DIVERSITY_ENABLED = False

# Memory System Settings
MEMORY_MAX_EPISODES = 50        # Max episodic memory entries per agent
MEMORY_RECENT_CONTEXT = 3       # Number of recent episodes to inject into prompt

# Persuasion Strategy Definitions (injected into agent prompts when prompt diversity is on)
# Only 3 core strategies. Most agents receive None (natural/unconstrained baseline behaviour).
# Strategy is only assigned to demographically distinctive agents; everyone else speaks freely.
PERSUASION_STRATEGIES = {
    "evidence_based": (
        "You prefer to use scientific evidence, statistics, and research findings "
        "when discussing vaccination. Cite data and studies to support your points."
    ),
    "anecdotal": (
        "You prefer to share personal stories and experiences from people you know "
        "when discussing vaccination. Real-life examples are most convincing to you."
    ),
    "emotional": (
        "You tend to share personal feelings, fears, and hopes when discussing vaccination. "
        "You connect with others through emotional experiences and empathy."
    ),
}

# Profile Visibility by Network Layer (relation type → visibility level)
# hh = household/family, wk = workplace, sm = social media
PROFILE_VISIBILITY = {
    "hh": "full",       # Family: full profile (all fields)
    "wk": "medium",     # Workplace: occupation, education, age, income, employment, urban
    "sm": "limited",    # Social media: occupation, age only
}

# Relation type priority (higher = closer relationship, more info shared)
RELATION_PRIORITY = {"hh": 3, "wk": 2, "sm": 1}

config = Config(
    project_name="LLMIP_basic",
    project_root=os.path.dirname(__file__),
    input_folder="data/input",
    output_folder="data/output",
)
