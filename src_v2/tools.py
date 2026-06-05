# tools.py

"""
Utility module for the LLM Vaccination Simulation.

Contains:
- Helper functions for text and data processing.
- Network generation logic.
- Asynchronous functions for generating agent profiles via LLM.
"""

import numpy as np
import re
import random
import json
import asyncio
import aiohttp
import networkx as nx
import pandas as pd
from typing import Optional, Dict, List
import nltk
nltk.download('vader_lexicon')


# --- Global Configuration ---
API_KEY = "abc123"
API_URL = "http://10.13.12.164:7890/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B"  # Model name that matches the server configuration
# Import settings from config
from config import MAX_CONCURRENT_CALLS, RELATION_PRIORITY, PROFILE_VISIBILITY, PERSUASION_STRATEGIES

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    sentiment_analyzer = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False
    sentiment_analyzer = None
    print("Warning: NLTK VADER not found. Sentiment analysis will be disabled.")

# ==================================
# SECTION 1: HELPER FUNCTIONS
# ==================================

def get_attitude_from_belief(belief: float) -> str:
    """Convert numerical belief value to a descriptive attitude string."""
    if belief >= 0.8: return "strongly supports vaccination"
    if belief >= 0.5: return "supports vaccination"
    if belief >= 0.2: return "is leaning towards vaccination"
    if belief > -0.2: return "is uncertain about vaccination"
    if belief > -0.5: return "is leaning against vaccination"
    if belief > -0.8: return "is against vaccination"
    return "is strongly against vaccination"

def get_sentiment_score(text: str) -> float:
    """Get a sentiment score for a given text using VADER. The compound score is already in [-1, 1]."""
    if not VADER_AVAILABLE or not sentiment_analyzer:
        return 0.0
    return sentiment_analyzer.polarity_scores(text)['compound']


def get_visible_profile(profile_data: Dict, relation_type: str) -> str:
    """
    Return a filtered profile string based on the network-layer relation type.
    
    Visibility levels (from config.PROFILE_VISIBILITY):
      - "full"    (hh/family):  all profile fields
      - "medium"  (wk/work):    occupation, education, age, income, employment, urban
      - "limited" (sm/social):  occupation, age only
    
    Args:
        profile_data: Full profile dict with keys like 'profile', 'age', 'occupation', etc.
        relation_type: One of 'hh', 'wk', 'sm'
    
    Returns:
        A profile description string with appropriate detail level.
    """
    visibility = PROFILE_VISIBILITY.get(relation_type, "limited")
    
    age = profile_data.get('age', 'unknown')
    occupation = profile_data.get('occupation', 'unknown')
    
    if visibility == "full":
        # Family/household: share everything
        return profile_data.get('profile', f"A {age}-year-old working in {occupation}.")
    
    elif visibility == "medium":
        # Workplace: occupation, education, age, income, employment status, urban/rural
        education = profile_data.get('education', 'unknown')
        income = profile_data.get('personal_income', None)
        urban = "urban area" if profile_data.get('urban', False) else "rural area"
        base = (
            f"A {age}-year-old working in {occupation}. "
            f"Has {education} level education. "
            f"Lives in an {urban}."
        )
        if isinstance(income, (int, float)):
            base += f" Annual income around ${income:,}."
        return base
    
    else:  # "limited" — social media
        # Only occupation and age
        return f"A {age}-year-old working in {occupation}."


def assign_persuasion_strategy(profile_data: Dict, belief: float):
    """
    Assign a persuasion strategy to an agent based on demographic profile ONLY.

    Design intent: only demographically distinctive agents receive a style instruction;
    the majority (middle-aged, lower/some-college educated) return None and speak freely,
    preserving natural social-influence dynamics and keeping influence_scatter r unbiased.

    Three strategies (checked in order, first match wins):
      1. education ≥ Bachelor/Master/PhD  → evidence_based
      2. age > 55                         → anecdotal  (personal stories)
      3. age < 30                         → emotional   (feelings & empathy)
      4. default                          → None  (no constraint; baseline behaviour)

    Returns:
        The persuasion strategy instruction string, or None if no strategy applies.
    """
    education = profile_data.get('education', '')
    age = profile_data.get('age', 40)

    # Rule 1: Highly-educated agents prefer evidence-based reasoning
    high_edu_keywords = ["Bachelor", "Master", "Doctorate", "PhD", "higher"]
    if any(kw.lower() in education.lower() for kw in high_edu_keywords):
        return PERSUASION_STRATEGIES["evidence_based"]

    # Rule 2: Older agents (>55) rely on personal anecdotes
    if age > 55:
        return PERSUASION_STRATEGIES["anecdotal"]

    # Rule 3: Young adults (<30) lean on emotional appeals
    if age < 30:
        return PERSUASION_STRATEGIES["emotional"]

    # Default: no strategy — agent speaks naturally (same as baseline)
    return None

def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract and parse a JSON object from an LLM response string.
    Tries multiple extraction strategies to handle various response formats.
    """
    if not response or not isinstance(response, str):
        return None
    
    # 增强的预处理：清除 Qwen 模型的 <think> 标签（三步清洗法）
    # 1. 清除完整的 <think>...</think> 对
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    # 2a. 清除不完整标签（到 JSON 之前）
    response = re.sub(r'<think>.*?(?=\{)', '', response, flags=re.DOTALL).strip()
    # 2b. 清除残留的 <think> 及其后所有内容
    response = re.sub(r'<think>.*$', '', response, flags=re.DOTALL).strip()
    # 3. 清除任何残留的单独标签
    response = re.sub(r'</?think>', '', response).strip()
    
    if not response:
        return None
    
    # 策略1: 尝试解析整个响应为 JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # 策略2: 提取 markdown JSON 代码块
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 策略3: 提取普通代码块
    json_match = re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 策略4: 查找第一个完整的 JSON 对象
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # 策略5: 尝试提取关键字段（作为最后手段）
    try:
        # 查找 summary_sentence
        summary_match = re.search(r'"summary_sentence"\s*:\s*"([^"]*)"', response)
        # 查找 belief_score
        score_match = re.search(r'"belief_score"\s*:\s*([-+]?\d*\.?\d+)', response)
        
        if summary_match and score_match:
            return {
                "summary_sentence": summary_match.group(1),
                "belief_score": float(score_match.group(1))
            }
    except (ValueError, AttributeError):
        pass
    
    # 所有策略失败
    print(f"⚠️  JSON Extraction Failed (length: {len(response)})")
    if '<think>' in response.lower():
        print(f"   ❌ Response still contains <think> tag!")
    print(f"   Preview: {response[:200]}...")
    return None

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    return obj

# ==================================
# SECTION 2: GENERATOR FUNCTIONS
# ==================================

def create_network(agent_ids: List[int], connection_prob: float, network_type: str) -> nx.Graph:
    """Create a weighted social network graph for the agents."""
    n_agents = len(agent_ids)
    print(f"Creating '{network_type}' network for {n_agents} agents...")

    if network_type == "small_world":
        k = max(2, int(connection_prob * n_agents * 2))
        k = min(k, n_agents - 1)
        if k % 2 != 0: k -= 1
        G = nx.watts_strogatz_graph(n_agents, k, p=0.3)
    elif network_type == "scale_free":
        m = max(1, int(connection_prob * n_agents * 0.5))
        G = nx.barabasi_albert_graph(n_agents, m)
    else:
        G = nx.erdos_renyi_graph(n_agents, connection_prob)

    mapping = {i: agent_ids[i] for i in range(n_agents)}
    nx.relabel_nodes(G, mapping, copy=False)

    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(0.1, 1.0)

    print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

async def _generate_single_profile_async(session: aiohttp.ClientSession) -> Dict:
    """Generate a single agent profile using the LLM."""
    prompt = """Generate a profile for a person in a simulation about vaccination attitudes. Provide a brief, realistic background story (2 sentences).
Return a JSON object with fields: "profile", "age" (18-80), "urban" (boolean), "occupation", "education", "personal_income".
Example:
{
    "profile": "John is a retired history teacher who enjoys reading and gardening. He generally trusts his family doctor but is wary of government mandates.",
    "age": 68,
    "urban": false,
    "occupation": "Retired Teacher",
    "education": "Master's Degree",
    "personal_income": 45000
}"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.9}

    try:
        async with session.post(API_URL, headers=headers, json=payload) as response:
            if response.status == 200:
                # FIX: First, parse the entire response from the API.
                api_response_data = await response.json()
                # FIX: Second, extract the 'content' string which contains the JSON we want.
                content_string = api_response_data['choices'][0]['message']['content']
                # FIX: Third, parse the JSON from the extracted content string.
                profile_data = extract_json_from_response(content_string)
                if profile_data:
                    return profile_data
    except Exception as e:
        print(f"Profile generation API error: {e}")

    # Fallback profile on failure
    return {
        'profile': "A person with average characteristics.", 'age': random.randint(18, 80),
        'urban': random.choice([True, False]), 'occupation': 'Worker',
        'education': 'High School', 'personal_income': random.randint(30000, 90000)
    }

async def generate_profiles(n_agents: int) -> List[Dict]:
    """Generate a specified number of agent profiles concurrently."""
    print(f"Generating {n_agents} agent profiles via LLM...")
    async with aiohttp.ClientSession() as session:
        tasks = [_generate_single_profile_async(session) for _ in range(n_agents)]
        profiles = await asyncio.gather(*tasks)
    print(f"Successfully generated {len(profiles)} profiles.")
    return profiles


def load_workplace_profiles(population_csv: str, network_csv: str) -> tuple[List[Dict], nx.Graph, Dict[int, int]]:
    """
    Load agent profiles and network from workplace CSV files.
    
    Args:
        population_csv: Path to the extended population CSV file
        network_csv: Path to the extended network CSV file (includes wk, hh, sm relations)
    
    Returns:
        - List of profile dictionaries
        - NetworkX graph
        - Mapping from reindex to agent_id (0-based)
    """
    print(f"Loading workplace data from CSV files...")
    
    # Load population data
    pop_df = pd.read_csv(population_csv)
    
    # Filter only workplace members
    workplace_members = pop_df[pop_df['is_workplace_member'] == True].copy()
    print(f"Found {len(workplace_members)} workplace members")
    
    # Create profiles from population data
    profiles = []
    reindex_to_agent_id = {}
    
    for idx, row in workplace_members.iterrows():
        # Map reindex to sequential agent_id (0-based)
        agent_id = len(profiles)
        reindex_to_agent_id[row['reindex']] = agent_id
        
        # Create profile description
        gender = "male" if row['gender'] == 'm' else "female"
        age = int(row['age'])
        
        # Determine education level
        edu_level = row['education']
        if edu_level >= 21:
            education = "Master's Degree or higher"
        elif edu_level >= 20:
            education = "Bachelor's Degree"
        elif edu_level >= 16:
            education = "High School Graduate"
        else:
            education = "Some High School"
        
        # Determine occupation category
        occ = int(row['occupation']) if pd.notna(row['occupation']) else 1
        occupation_map = {
            1: "Management",
            2: "Business/Finance",
            3: "Professional",
            4: "Service",
            5: "Sales",
            6: "Office/Administrative",
            7: "Healthcare"
        }
        occupation = occupation_map.get(occ, "Service")
        
        # Create profile narrative
        profile_text = f"A {age}-year-old {gender} working in {occupation}. "
        profile_text += f"Has {education} level education. "
        
        if row['health_insurance'] == 1:
            profile_text += "Has health insurance coverage. "
        
        income = int(row['personal_income']) if pd.notna(row['personal_income']) else 35000
        family_size = int(row['family_size']) if pd.notna(row['family_size']) else 1
        
        if family_size > 1:
            profile_text += f"Lives with {family_size-1} family member(s). "
        
        profile_data = {
            'profile': profile_text,
            'age': age,
            'urban': row['urban'],
            'occupation': occupation,
            'education': education,
            'personal_income': income,
            'reindex': int(row['reindex'])
        }
        
        profiles.append(profile_data)
    
    print(f"Created {len(profiles)} workplace member profiles")
    
    # Load network data - include ALL relations (wk, hh, sm)
    network_df = pd.read_csv(network_csv)
    print(f"Loading network connections from {len(network_df)} total edges...")
    
    # First pass: identify all nodes in the network that connect to workplace members
    all_network_nodes = set()
    for _, row in network_df.iterrows():
        source_reindex = int(row['source_reindex'])
        target_reindex = int(row['target_reindex'])
        all_network_nodes.add(source_reindex)
        all_network_nodes.add(target_reindex)
    
    # Identify external nodes (not workplace members but connected to them)
    workplace_reindexes = set(reindex_to_agent_id.keys())
    external_nodes = all_network_nodes - workplace_reindexes
    
    print(f"Found {len(external_nodes)} external nodes connected to workplace members")
    
    # Create profiles for external nodes (if they exist in population data)
    external_added = 0
    for ext_reindex in external_nodes:
        # Try to find in population data
        ext_row = pop_df[pop_df['reindex'] == ext_reindex]
        
        if len(ext_row) > 0:
            ext_row = ext_row.iloc[0]
            agent_id = len(profiles)
            reindex_to_agent_id[ext_reindex] = agent_id
            
            gender = "male" if ext_row['gender'] == 'm' else "female"
            age = int(ext_row['age'])
            
            edu_level = ext_row['education']
            if edu_level >= 21:
                education = "Master's Degree or higher"
            elif edu_level >= 20:
                education = "Bachelor's Degree"
            elif edu_level >= 16:
                education = "High School Graduate"
            else:
                education = "Some High School"
            
            occ = int(ext_row['occupation']) if pd.notna(ext_row['occupation']) else 1
            occupation_map = {
                1: "Management", 2: "Business/Finance", 3: "Professional",
                4: "Service", 5: "Sales", 6: "Office/Administrative", 7: "Healthcare"
            }
            occupation = occupation_map.get(occ, "Service")
            
            profile_text = f"A {age}-year-old {gender} working in {occupation}. "
            profile_text += f"Has {education} level education. "
            
            if ext_row['health_insurance'] == 1:
                profile_text += "Has health insurance coverage. "
            
            income = int(ext_row['personal_income']) if pd.notna(ext_row['personal_income']) else 35000
            family_size = int(ext_row['family_size']) if pd.notna(ext_row['family_size']) else 1
            
            if family_size > 1:
                profile_text += f"Lives with {family_size-1} family member(s). "
            
            profile_text += "(External contact)"
            
            profile_data = {
                'profile': profile_text,
                'age': age,
                'urban': ext_row['urban'],
                'occupation': occupation,
                'education': education,
                'personal_income': income,
                'reindex': int(ext_reindex)
            }
            
            profiles.append(profile_data)
            external_added += 1
    
    print(f"Added {external_added} external contacts with profiles")
    print(f"Total agents: {len(profiles)} (30 workplace + {external_added} external)")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add all agents as nodes
    for agent_id in range(len(profiles)):
        G.add_node(agent_id)
    
    # Add edges based on network CSV (all relation types)
    edges_added = 0
    edges_by_type = {'wk': 0, 'hh': 0, 'sm': 0}
    for _, row in network_df.iterrows():
        source_reindex = int(row['source_reindex'])
        target_reindex = int(row['target_reindex'])
        relation = row['Relation']
        
        # Map reindex to agent_id (both must be in our agent list)
        if source_reindex in reindex_to_agent_id and target_reindex in reindex_to_agent_id:
            source_id = reindex_to_agent_id[source_reindex]
            target_id = reindex_to_agent_id[target_reindex]
            
            # Store relation type on edge; if edge already exists, keep the strongest relation
            if G.has_edge(source_id, target_id):
                existing_relation = G[source_id][target_id].get('relation', 'sm')
                if RELATION_PRIORITY.get(relation, 0) > RELATION_PRIORITY.get(existing_relation, 0):
                    G[source_id][target_id]['relation'] = relation
            else:
                G.add_edge(source_id, target_id, weight=random.uniform(0.5, 1.0), relation=relation)
            
            edges_added += 1
            if relation in edges_by_type:
                edges_by_type[relation] += 1
    
    print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"  - Workplace (wk): {edges_by_type['wk']} edges")
    print(f"  - Household (hh): {edges_by_type['hh']} edges")
    print(f"  - Social Media (sm): {edges_by_type['sm']} edges")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    return profiles, G, reindex_to_agent_id