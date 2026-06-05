# agent.py

"""
Defines the VaxAgent class for the simulation.
This is the final, definitive version with robust dialogue and elicitation logic.
"""

import mesa
import numpy as np
import random
import aiohttp
import re
from typing import List, Dict
from openai import AsyncOpenAI
import logging

from tools import get_attitude_from_belief, get_sentiment_score, extract_json_from_response, get_visible_profile, assign_persuasion_strategy

# 配置日志，禁用 OpenAI 的详细 INFO 日志
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Import API configuration from config
from config import API_URL, API_KEY, MODEL_NAME, BELIEF_DISTRIBUTION_TYPE, BELIEF_MEANS, BELIEF_STD, MEMORY_MAX_EPISODES, MEMORY_RECENT_CONTEXT

# Initialize OpenAI client with configuration from config.py
client = AsyncOpenAI(
    base_url=API_URL,
    api_key=API_KEY,
)


REFUSAL_PATTERNS = [
    "i can't", "i cannot", "i'm sorry", "i am sorry", "unable to", "i won't",
    "i will not", "can't assist", "cannot assist", "not able to", "i must decline",
    "refuse", "policy", "not appropriate", "can't help with that"
]


def is_refusal(text: str) -> bool:
    if not text:
        return True
    lowered = text.strip().lower()
    return any(p in lowered for p in REFUSAL_PATTERNS)


class VaxAgent(mesa.Agent):
    """An agent with parallel belief states for comparative analysis."""

    def __init__(self, unique_id: int, model, profile_data: Dict, alpha: float):
        # Initialize Mesa Agent with unique_id and model
        super().__init__(unique_id, model)
        # Set additional attributes
        self.profile = profile_data['profile']
        self.profile_data = profile_data  # Store full profile dict for visibility filtering
        self.age = profile_data['age']
        self.alpha = alpha

        # 使用固定的初始belief（如果提供）
        if hasattr(model, 'fixed_initial_beliefs') and model.fixed_initial_beliefs is not None:
            initial_belief = model.fixed_initial_beliefs[unique_id]
        else:
            # 否则随机生成
            mu = BELIEF_MEANS.get(BELIEF_DISTRIBUTION_TYPE, 0.0)
            initial_belief = float(np.clip(np.random.normal(mu, BELIEF_STD), -1.0, 1.0))
        
        self.belief = initial_belief
        self.belief_vader = initial_belief

        self.is_vaccinated = False
        self.tick_vaccinated = -1

        self.dialogue_history = []
        self.belief_history = [self.belief]
        self.belief_vader_history = [self.belief_vader]
        self.tick_belief = self.belief
        
        # Scheduling and locking
        self.is_locked = False  # Lock status for exclusive dialog participation
        self.dialog_memory = {}  # Per-neighbor dialog memory: {neighbor_id: [messages]}
        
        # ===== Memory System (enabled via model.memory_enabled) =====
        # Episodic Memory: stores structured records of past dialogues
        self.episodic_memory = []  # List of {tick, neighbor_id, neighbor_stance, opinion_shift, key_argument}
        
        # ===== Prompt Diversity (enabled via model.prompt_diversity_enabled) =====
        # Persuasion strategy is assigned once at init and stays stable
        self.persuasion_strategy = assign_persuasion_strategy(profile_data, initial_belief)

    def get_neighbors(self) -> List['VaxAgent']:
        """Get list of neighbors from the network graph."""
        neighbor_ids = list(self.model.network.neighbors(self.unique_id))
        return [self.model.schedule.agents[nid] for nid in neighbor_ids]

    # ===== Memory System Helper Methods =====

    def _build_memory_context(self, neighbor_id: int) -> str:
        """
        Build a memory context string from episodic memory to inject into prompts.
        Includes: history with this specific neighbor + global stats + enriched recent episode summaries.
        """
        if not self.episodic_memory:
            return ""
        
        context_parts = []
        
        # 1. History with this specific neighbor (if any)
        with_neighbor = [ep for ep in self.episodic_memory if ep['neighbor_id'] == neighbor_id]
        if with_neighbor:
            last_ep = with_neighbor[-1]
            n_times = len(with_neighbor)
            context_parts.append(
                f"You have spoken with this person {n_times} time(s) before. "
                f"Last time (step {last_ep['tick']}), they {last_ep['neighbor_stance']}. "
                f"After that conversation, your opinion shifted by {last_ep['opinion_shift']:+.2f}."
            )
            # If they made a memorable argument, recall it
            if last_ep.get('key_argument'):
                context_parts.append(
                    f"Their most memorable point was: \"{last_ep['key_argument']}\""
                )
        
        # 2. Global statistics
        n_total = len(self.episodic_memory)
        avg_shift = np.mean([ep['opinion_shift'] for ep in self.episodic_memory])
        context_parts.append(
            f"You have had {n_total} conversation(s) about vaccination so far, "
            f"with an average opinion shift of {avg_shift:+.3f} per conversation."
        )
        
        # 3. Enriched recent conversation summaries (with key arguments inline)
        recent = self.episodic_memory[-MEMORY_RECENT_CONTEXT:]
        # Exclude current neighbor to avoid duplication with section 1
        recent_others = [ep for ep in recent if ep['neighbor_id'] != neighbor_id]
        if recent_others:
            summaries = []
            for ep in recent_others:
                summary = (
                    f"step {ep['tick']}: spoke with someone who {ep['neighbor_stance']} "
                    f"(your shift: {ep['opinion_shift']:+.2f})"
                )
                # Include the key argument from that conversation for richer context
                if ep.get('key_argument'):
                    summary += f" — they argued: \"{ep['key_argument']}\""
                summaries.append(summary)
            context_parts.append("Recent conversations: " + "; ".join(summaries))
        
        return "\n[Your conversation memory] " + " ".join(context_parts)

    @staticmethod
    def _extract_key_argument(exchanges: List[Dict], neighbor_id: int) -> str:
        """
        Extract the most argumentative statement from the neighbor's exchanges.
        
        Strategy (multi-pass):
          1. Score each neighbor statement by argumentative keyword density
          2. Prefer middle rounds (turns 2-3) over opening/closing
          3. Prefer longer substantive statements
          4. Fallback: longest neighbor statement
        """
        # Collect neighbor's statements with turn index
        neighbor_msgs = []
        for idx, ex in enumerate(exchanges):
            if ex.get('speaker_id') == neighbor_id:
                msg = ex.get('message', '').strip()
                if len(msg) > 15:  # Skip very short filler responses
                    neighbor_msgs.append((idx, msg))
        
        if not neighbor_msgs:
            return ""
        
        # Argumentative keywords that signal a substantive point
        ARG_KEYWORDS = [
            'because', 'research', 'study', 'evidence', 'data', 'risk',
            'safety', 'effective', 'side effect', 'experience', 'personally',
            'believe', 'think', 'important', 'concern', 'trust', 'recommend',
            'statistics', 'proven', 'natural', 'immune', 'health', 'protect',
            'family', 'children', 'community', 'doctor', 'expert', 'government',
            'however', 'although', 'but', 'actually', 'in fact', 'consider',
        ]
        
        def score_message(idx: int, msg: str) -> float:
            msg_lower = msg.lower()
            # Keyword density score
            keyword_hits = sum(1 for kw in ARG_KEYWORDS if kw in msg_lower)
            # Length bonus (longer = more substantive, capped)
            length_score = min(len(msg) / 200.0, 1.0)
            # Position bonus: middle turns (index 1-2 in a 4-exchange dialogue) are usually argumentative
            total_turns = len(exchanges)
            if total_turns > 2:
                middle_ratio = 1.0 - abs(idx / total_turns - 0.5) * 2  # Peak at middle
            else:
                middle_ratio = 0.5
            return keyword_hits * 2.0 + length_score + middle_ratio * 1.5
        
        # Score and rank
        scored = [(score_message(idx, msg), msg) for idx, msg in neighbor_msgs]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        best_msg = scored[0][1]
        # Truncate to keep memory concise
        if len(best_msg) > 150:
            # Try to cut at a sentence boundary
            cut_point = best_msg[:150].rfind('.')
            if cut_point > 80:  # Only if we keep a reasonable portion
                best_msg = best_msg[:cut_point + 1]
            else:
                best_msg = best_msg[:150] + "..."
        
        return best_msg

    def _store_episode(self, tick: int, neighbor_id: int, neighbor_belief: float,
                       belief_before: float, belief_after: float, exchanges: List[Dict]):
        """
        Store an episodic memory record after a dialogue completes.
        Also extracts and stores the key argument from the neighbor's statements.
        """
        opinion_shift = belief_after - belief_before
        neighbor_stance = get_attitude_from_belief(neighbor_belief)
        
        # Extract key argument using multi-strategy scorer
        key_argument = self._extract_key_argument(exchanges, neighbor_id)
        
        # Store episodic memory
        episode = {
            'tick': tick,
            'neighbor_id': neighbor_id,
            'neighbor_stance': neighbor_stance,
            'opinion_shift': round(opinion_shift, 4),
            'key_argument': key_argument,
        }
        self.episodic_memory.append(episode)
        
        # Trim if exceeds max
        if len(self.episodic_memory) > MEMORY_MAX_EPISODES:
            self.episodic_memory = self.episodic_memory[-MEMORY_MAX_EPISODES:]

    # ===== Profile Visibility Helper =====

    def _get_profile_for_neighbor(self, neighbor: 'VaxAgent') -> str:
        """
        Get the profile string to show to a neighbor, filtered by network layer.
        Uses the relation type stored on the edge between self and the neighbor.
        Only applies when prompt_diversity_enabled is True on the model.
        """
        if not getattr(self.model, 'prompt_diversity_enabled', False):
            return self.profile  # Return full profile when diversity is off
        
        # Get relation type from the network edge
        if self.model.network.has_edge(self.unique_id, neighbor.unique_id):
            relation = self.model.network[self.unique_id][neighbor.unique_id].get('relation', 'sm')
        else:
            relation = 'sm'  # Default to most limited
        
        return get_visible_profile(self.profile_data, relation)

    async def conduct_dialogue_with_neighbor(self, session: aiohttp.ClientSession, neighbor: 'VaxAgent') -> Dict:
        """Conducts a multi-turn dialogue with robust reflection."""
        agent_a, agent_b = self, neighbor
        dialogue_record = {
            'tick': self.model.schedule.steps, 'interlocutors': [agent_a.unique_id, agent_b.unique_id],
            'initial_beliefs': {'self': agent_a.tick_belief, 'neighbor': agent_b.belief},
            'exchanges': [],
        }
        messages = []

        # ===== Determine feature flags from model =====
        memory_on = getattr(self.model, 'memory_enabled', False)
        diversity_on = getattr(self.model, 'prompt_diversity_enabled', False)

        # ===== Build profile strings (filtered by network layer if diversity is on) =====
        profile_a_for_b = agent_a._get_profile_for_neighbor(agent_b)  # What B sees of A
        profile_b_for_a = agent_b._get_profile_for_neighbor(agent_a)  # What A sees of B
        # Each agent's own self-view is always the full profile
        profile_a_self = agent_a.profile
        profile_b_self = agent_b.profile

        # ===== Build memory context (if enabled) =====
        memory_ctx_a = agent_a._build_memory_context(agent_b.unique_id) if memory_on else ""
        memory_ctx_b = agent_b._build_memory_context(agent_a.unique_id) if memory_on else ""

        # ===== Build persuasion strategy instruction (if enabled and strategy assigned) =====
        persuasion_a = (
            f"\n[Your communication style] {agent_a.persuasion_strategy}"
            if diversity_on and agent_a.persuasion_strategy is not None else ""
        )
        persuasion_b = (
            f"\n[Your communication style] {agent_b.persuasion_strategy}"
            if diversity_on and agent_b.persuasion_strategy is not None else ""
        )

        # ========== 第一次 API 调用：初始对话 ==========
        prompt = f"""You are Person B having a conversation about vaccination. 
Your background: {profile_b_self}
What you know about Person A: {profile_a_for_b}
Your current attitude: You {get_attitude_from_belief(agent_b.belief)}{persuasion_b}{memory_ctx_b}

Start a brief conversation with Person A about vaccination (2-3 sentences)."""
        
        api_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant in a fictional social simulation. "
                    "Provide brief, natural responses (2-3 sentences). "
                    "Avoid medical advice; speak in general terms. Do not refuse to answer."
                )
            },
            {"role": "user", "content": prompt}
        ]
        
        try:
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=api_messages,
                temperature=0.7
                # 不限制 max_tokens，完整接收响应后再处理
            )
            # 提取内容并立即清洗（三步增强清洗）
            content = completion.choices[0].message.content
            
            # 调试：打印原始内容长度
            if '<think>' in content.lower():
                print(f"🔍 Agent {self.unique_id}: Raw content contains <think> tag (length: {len(content)})")
            
            # 三步清洗法
            response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            response = re.sub(r'<think>.*', '', response, flags=re.DOTALL).strip()  # 清除残留的不完整开始标签
            response = re.sub(r'</?think>', '', response).strip()
            
            # 调试：验证清洗后的结果
            if '<think>' in response.lower():
                print(f"❌ Agent {self.unique_id}: Cleaning FAILED! Still contains <think> (length: {len(response)})")
            else:
                if len(content) != len(response):
                    print(f"✅ Agent {self.unique_id}: Cleaned {len(content) - len(response)} chars")
            
            if is_refusal(response):
                response = "I can share general thoughts without giving medical advice. It's a complex topic, and people weigh safety, effectiveness, and trust differently. I'm open to discussing it in general terms."

            if hasattr(self.model, 'api_call_counter'):
                self.model.api_call_counter.update(1)
        except Exception as e:
            print(f"LLM API Exception (Agent {self.unique_id}): {e}")
            response = "I understand your perspective on this matter."
        
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})
        dialogue_record['exchanges'].append({'speaker_id': agent_b.unique_id, 'message': response})

        # ========== 对话循环：3轮交互 ==========
        for turn in range(1, 4):
            is_self_turn = turn % 2 != 0
            current_speaker = agent_a if is_self_turn else agent_b

            # 构建提示 — inject filtered profile, memory, and persuasion strategy
            if is_self_turn:
                prompt = f"""You are Person A responding.
Your background: {profile_a_self}
What you know about Person B: {profile_b_for_a}
Your current view: You {get_attitude_from_belief(current_speaker.tick_belief)}{persuasion_a}{memory_ctx_a}

Respond naturally to what Person B said (2-3 sentences)."""
            else:
                prompt = f"""You are Person B continuing the conversation.
Your background: {profile_b_self}
What you know about Person A: {profile_a_for_b}
Your current view: You {get_attitude_from_belief(current_speaker.belief)}{persuasion_b}{memory_ctx_b}

Continue the conversation naturally (2-3 sentences)."""

            messages.append({"role": "user", "content": prompt})
            
            # API 调用
            api_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant in a fictional social simulation. "
                        "Provide brief, natural responses (2-3 sentences). "
                        "Avoid medical advice; speak in general terms. Do not refuse to answer."
                    )
                }
            ] + messages
            
            try:
                completion = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=api_messages,
                    temperature=0.7
                    # 不限制 max_tokens，完整接收响应后再处理
                )
                # 提取内容并立即清洗（三步增强清洗）
                content = completion.choices[0].message.content
                
                # 三步清洗法：处理完整和不完整的 <think> 标签
                response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                response = re.sub(r'<think>.*', '', response, flags=re.DOTALL).strip()  # 清除残留的不完整开始标签
                response = re.sub(r'</?think>', '', response).strip()
                
                if is_refusal(response):
                    response = "I can share general thoughts without giving medical advice. It's a nuanced topic, and people balance different concerns. I'm open to discussing it respectfully."

                if hasattr(self.model, 'api_call_counter'):
                    self.model.api_call_counter.update(1)
            except Exception as e:
                print(f"LLM API Exception (Agent {self.unique_id}): {e}")
                response = "I understand your perspective on this matter."
            
            messages.append({"role": "assistant", "content": response})
            dialogue_record['exchanges'].append({'speaker_id': current_speaker.unique_id, 'message': response})

        # ========== 最后一次 API 调用：开放式自评（更自然的反思）==========
        # Add memory reminder for reflection (if enabled)
        memory_reflection = ""
        if memory_on and agent_a.episodic_memory:
            n_past = len(agent_a.episodic_memory)
            memory_reflection = f"\nNote: You have had {n_past} previous conversation(s) about this topic. Consider how your accumulated experience shapes your current view."
        
        elicitation_instruction = f"""After this conversation, please reflect on your current view about vaccination.

First, in 2-3 sentences, describe how this conversation affected your thinking (if at all).{memory_reflection}

Then, provide a JSON object with your updated view:
{{
  "summary_sentence": "One sentence describing your current view after the conversation",
  "belief_score": <a number between -1.0 and 1.0 representing your current stance>
}}

Guidelines for belief_score:
- Think about where you stand NOW on the spectrum from strongly against (-1.0) to strongly support (+1.0)
- Be honest about your genuine view, not what you think is "correct"
- Consider: +1.0=strongly support, +0.5=support, 0=uncertain, -0.5=against, -1.0=strongly against

Your view BEFORE the conversation: You {get_attitude_from_belief(agent_a.tick_belief)} (score: {agent_a.tick_belief:.2f})

Please share your reflection and provide the JSON:"""

        messages.append({"role": "user", "content": elicitation_instruction})
        
        # JSON 请求的 API 调用
        api_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant in a fictional social simulation. "
                    "Provide responses in valid JSON format only. Do not include any text outside the JSON object. "
                    "Avoid medical advice; speak in general terms. Do not refuse to answer."
                )
            }
        ] + messages
        
        try:
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=api_messages,
                temperature=0.7
                # 不限制 max_tokens，完整接收响应后再处理
            )
            # 提取内容并立即清洗
            content = completion.choices[0].message.content
            
            # 调试：JSON 请求的清洗验证（关键！）
            if '<think>' in content.lower():
                print(f"🔍 JSON Request - Agent {self.unique_id}: Raw content contains <think> (length: {len(content)})")
                print(f"   Preview: {content[:150]}...")
                # 检查是否有结束标签
                if '</think>' not in content.lower():
                    print(f"   ⚠️  WARNING: No closing </think> tag found!")
            
            # 增强的清洗：处理完整和不完整的 <think> 标签
            # 1. 先清除完整的 <think>...</think> 对
            final_response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            # 2. 清除残留的开始标签：两种策略
            #    a) 如果后面有 JSON（{），只删除到 { 之前
            final_response = re.sub(r'<think>.*?(?=\{)', '', final_response, flags=re.DOTALL).strip()
            #    b) 如果没有 JSON，删除整个残留的 <think> 标签及其后所有内容
            final_response = re.sub(r'<think>.*$', '', final_response, flags=re.DOTALL).strip()
            # 3. 最后清除任何残留的单独 <think> 或 </think> 标签
            final_response = re.sub(r'</?think>', '', final_response).strip()
            
            # 调试：验证清洗后的结果
            if '<think>' in final_response.lower():
                print(f"❌ JSON Request - Agent {self.unique_id}: Cleaning FAILED!")
                print(f"   After cleaning: {final_response[:150]}...")
            else:
                if len(content) != len(final_response):
                    print(f"✅ JSON Request - Agent {self.unique_id}: Cleaned {len(content) - len(final_response)} chars")
            
            if is_refusal(final_response):
                final_response = (
                    "{\"summary_sentence\": \"I feel uncertain and prefer to think about this more in general terms.\", "
                    f"\"belief_score\": {agent_a.tick_belief:.2f}}}"
                )

            if hasattr(self.model, 'api_call_counter'):
                self.model.api_call_counter.update(1)
        except Exception as e:
            print(f"LLM API Exception (Agent {self.unique_id}): {e}")
            final_response = (
                "{\"summary_sentence\": \"I feel uncertain and prefer to think about this more in general terms.\", "
                f"\"belief_score\": {agent_a.tick_belief:.2f}}}"
            )

        # 解析 JSON
        elicited_data = extract_json_from_response(final_response)

        # Verification: retry once if JSON invalid or refusal detected
        if not elicited_data or 'summary_sentence' not in elicited_data or 'belief_score' not in elicited_data:
            verification_prompt = (
                "Return ONLY a JSON object with keys: summary_sentence (string) and belief_score (number between -1 and 1). "
                "No extra text. Keep it brief and neutral."
            )
            verification_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant in a fictional social simulation. "
                        "Provide responses in valid JSON format only. Do not include any text outside the JSON object."
                    )
                },
                {"role": "user", "content": verification_prompt}
            ]
            try:
                verification_completion = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=verification_messages,
                    temperature=0.2
                )
                verification_content = verification_completion.choices[0].message.content
                verification_clean = re.sub(r'<think>.*?</think>', '', verification_content, flags=re.DOTALL).strip()
                verification_clean = re.sub(r'<think>.*', '', verification_clean, flags=re.DOTALL).strip()
                verification_clean = re.sub(r'</?think>', '', verification_clean).strip()

                if is_refusal(verification_clean):
                    verification_clean = (
                        "{\"summary_sentence\": \"I feel uncertain and prefer to think about this more in general terms.\", "
                        f"\"belief_score\": {agent_a.tick_belief:.2f}}}"
                    )

                if hasattr(self.model, 'api_call_counter'):
                    self.model.api_call_counter.update(1)

                elicited_data = extract_json_from_response(verification_clean)
            except Exception as e:
                print(f"LLM Verification Exception (Agent {self.unique_id}): {e}")

        if elicited_data and 'summary_sentence' in elicited_data and 'belief_score' in elicited_data:
            elicited_score = np.clip(float(elicited_data['belief_score']), -1.0, 1.0)
            dialogue_record.update({
                'elicited_summary': elicited_data['summary_sentence'],
                'elicited_self_score': elicited_score,
                'elicited_sentiment_score': get_sentiment_score(elicited_data['summary_sentence']),
                'is_valid': True  # 标记为有效对话
            })
            
            # ===== Store episodic memory (if memory system is enabled) =====
            if memory_on:
                agent_a._store_episode(
                    tick=self.model.schedule.steps,
                    neighbor_id=agent_b.unique_id,
                    neighbor_belief=agent_b.belief,
                    belief_before=agent_a.tick_belief,
                    belief_after=elicited_score,
                    exchanges=dialogue_record['exchanges']
                )
        else:
            # 没有有效的 summary，设置为无效对话
            dialogue_record.update({
                'elicited_summary': None,
                'elicited_self_score': None,
                'elicited_sentiment_score': None,
                'is_valid': False  # 标记为无效对话
            })
        return dialogue_record

    # ... The rest of agent.py (update_belief_from_dialogues, step, advance) remains unchanged ...
    async def update_belief_from_dialogues(self, session: aiohttp.ClientSession):
        """
        Conduct dialogues ONLY with network neighbors and calculate pending belief changes.
        Invalid dialogues (without proper summary) are excluded from belief updates.
        """
        # 清理旧的pending beliefs
        self.pending_belief = None
        self.pending_belief_vader = None
        
        # 已接种的agent不再更新belief（但仍可作为neighbor参与别人的对话）
        if self.is_vaccinated:
            self.pending_belief = self.belief  # 保持当前belief不变
            self.pending_belief_vader = self.belief_vader
            return

        # 保存当前状态
        self.tick_belief = self.belief
        tick_belief_vader = self.belief_vader
        
        # 只与网络中的邻居进行对话
        neighbors = self.get_neighbors()
        if not neighbors:
            self.pending_belief = self.belief
            self.pending_belief_vader = self.belief_vader
            return

        belief_changes_llm, belief_changes_vader, weights = [], [], []

        for neighbor in neighbors:
            dialogue_record = await self.conduct_dialogue_with_neighbor(session, neighbor)
            self.dialogue_history.append(dialogue_record)

            # 只处理有效的对话（有 summary 的对话）
            if dialogue_record.get('is_valid', False):
                final_elicited_belief_llm = dialogue_record['elicited_self_score']
                change_llm = final_elicited_belief_llm - self.tick_belief
                belief_changes_llm.append(change_llm)

                final_elicited_belief_vader = dialogue_record['elicited_sentiment_score']
                change_vader = final_elicited_belief_vader - tick_belief_vader
                belief_changes_vader.append(change_vader)

                # 从网络边获取权重
                edge_weight = self.model.network[self.unique_id][neighbor.unique_id].get('weight', 0.5)
                weights.append(edge_weight)

        # 如果有有效的对话，计算加权平均变化
        if weights and len(belief_changes_llm) > 0:
            weighted_mean_change_llm = np.average(belief_changes_llm, weights=weights)
            self.pending_belief = np.clip(self.tick_belief + self.alpha * weighted_mean_change_llm, -1.0, 1.0)

            weighted_mean_change_vader = np.average(belief_changes_vader, weights=weights)
            self.pending_belief_vader = np.clip(tick_belief_vader + self.alpha * weighted_mean_change_vader, -1.0, 1.0)
        else:
            # 没有有效对话，保持原信念不变
            self.pending_belief = self.belief
            self.pending_belief_vader = self.belief_vader

    def step(self):
        if not self.is_vaccinated:
            vaccination_prob = max(0, self.belief)
            if random.random() < vaccination_prob:
                self.is_vaccinated = True
                self.tick_vaccinated = self.model.schedule.steps
                # 接种后设置pending_belief为1.0，将在advance()中更新
                self.pending_belief = 1.0
                self.pending_belief_vader = 1.0

    def advance(self):
        # Update belief from pending values if they exist
        if hasattr(self, 'pending_belief') and self.pending_belief is not None:
            self.belief = self.pending_belief
        if hasattr(self, 'pending_belief_vader') and self.pending_belief_vader is not None:
            self.belief_vader = self.pending_belief_vader
        
        # Always record current belief to history
        self.belief_history.append(self.belief)
        self.belief_vader_history.append(self.belief_vader)