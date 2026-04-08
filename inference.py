"""
Inference Script — Autonomous Traffic Control OpenEnv
======================================================
Hackathon-compliant inference script following the required STDOUT format.

Usage:
  export API_BASE_URL=https://router.huggingface.co/v1
  export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
  export HF_TOKEN=your_hf_token
  export TRAFFIC_TASK=basic_flow   # or peak_hour / full_control
  python inference.py
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("TRAFFIC_TASK", "basic_flow")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "traffic_control"

TEMPERATURE  = 0.3
MAX_TOKENS   = 200
MAX_STEPS    = {
    "basic_flow":   50,
    "peak_hour":    75,
    "full_control": 100,
}.get(TASK_NAME, 50)

SUCCESS_SCORE_THRESHOLD = {
    "basic_flow":   0.5,
    "peak_hour":    0.6,
    "full_control": 0.7,
}.get(TASK_NAME, 0.5)

# ─── STDOUT Logging ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI traffic signal controller for a 4-way intersection.
You receive the current state of the intersection and must output a single JSON action.

Signal phases:
  0 = NS Through (North-South straight traffic)
  1 = NS Left-turn (North-South turning traffic)
  2 = EW Through (East-West straight traffic)
  3 = EW Left-turn (East-West turning traffic)
  4 = Pedestrian NS crossing
  5 = Pedestrian EW crossing
  6 = All-Red (clearance)
  7 = Emergency Corridor (used ONLY when emergency_active=true)

Strategy:
  - Prioritise the direction with the longest queue
  - If emergency_active=true AND emergency_distance<100, set emergency_override=true and phase=7
  - If pedestrian_waiting=true and no emergency, occasionally use phase 4 or 5
  - Avoid switching phases too frequently (duration >= 20 seconds preferred)
  - Choose duration 30-45 for high queues, 15-20 for balanced queues

Output ONLY a JSON object with these exact keys, nothing else:
{"phase": <int 0-7>, "duration": <int 5-60>, "emergency_override": <bool>}
""").strip()

# ─── Agent Logic ──────────────────────────────────────────────────────────────

def build_user_prompt(obs: Dict[str, Any], step: int, history: List[str]) -> str:
    queues = {
        "N1": obs.get("queue_N1", 0), "N2": obs.get("queue_N2", 0),
        "S1": obs.get("queue_S1", 0), "S2": obs.get("queue_S2", 0),
        "E1": obs.get("queue_E1", 0), "E2": obs.get("queue_E2", 0),
        "W1": obs.get("queue_W1", 0), "W2": obs.get("queue_W2", 0),
    }
    total_queue = sum(queues.values())
    dominant = max(queues, key=queues.get)

    history_text = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(f"""
    Step {step} | Task: {TASK_NAME}
    
    Queue lengths: {json.dumps(queues)}
    Total vehicles waiting: {total_queue}
    Busiest lane: {dominant} ({queues[dominant]} vehicles)
    
    Current phase: {obs.get('current_phase')} | Time remaining: {obs.get('phase_time_remaining')}s
    
    Emergency: active={obs.get('emergency_active')} | direction={obs.get('emergency_direction') or 'none'} | distance={obs.get('emergency_distance'):.0f}m | type={obs.get('emergency_type') or 'none'}
    Incident active: {obs.get('incident_active')}
    Pedestrians waiting: {obs.get('pedestrian_waiting')}
    
    Throughput so far: {obs.get('throughput_last_phase')}
    Avg wait time: {obs.get('avg_wait_time'):.1f}s
    
    Recent actions:
    {history_text}
    
    Output your action JSON now:
    """).strip()


def get_action(client: OpenAI, obs: Dict[str, Any], step: int, history: List[str]) -> Dict[str, Any]:
    """Call LLM to get next action. Falls back to heuristic on failure."""
    user_prompt = build_user_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if "```" in text:
            text = text.split("```")[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        action = json.loads(text)
        # Validate keys
        assert "phase" in action and "duration" in action and "emergency_override" in action
        return action
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}, using heuristic fallback", flush=True)
        return _heuristic_action(obs)


def _heuristic_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback heuristic: serve longest queue, honour emergency."""
    if obs.get("emergency_active") and obs.get("emergency_distance", 999) < 100:
        return {"phase": 7, "duration": 20, "emergency_override": True}

    queues = {
        0: obs.get("queue_N1", 0) + obs.get("queue_S1", 0),  # NS Through
        1: obs.get("queue_N2", 0) + obs.get("queue_S2", 0),  # NS Left
        2: obs.get("queue_E1", 0) + obs.get("queue_W1", 0),  # EW Through
        3: obs.get("queue_E2", 0) + obs.get("queue_W2", 0),  # EW Left
    }
    best_phase = max(queues, key=queues.get)
    return {"phase": best_phase, "duration": 30, "emergency_override": False}

# ─── Environment Client ───────────────────────────────────────────────────────

async def env_reset(task: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{ENV_BASE_URL}/reset", json={"task": task})
        r.raise_for_status()
        return r.json()

async def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{ENV_BASE_URL}/step", json={"action": action})
        r.raise_for_status()
        return r.json()

async def env_grade() -> float:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{ENV_BASE_URL}/grade")
        r.raise_for_status()
        return r.json()["score"]

# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_data = await env_reset(TASK_NAME)
        obs = reset_data["observation"]

        for step in range(1, MAX_STEPS + 1):
            action = get_action(client, obs, step, history)

            try:
                result = await env_step(action)
            except Exception as e:
                log_step(step=step, action=str(action), reward=0.0, done=True, error=str(e))
                break

            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            obs = result.get("observation", obs)
            info = result.get("info", {})
            error = None

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(action, separators=(",", ":"))
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: phase={action['phase']} dur={action['duration']} "
                f"override={action['emergency_override']} → reward={reward:+.2f}"
            )

            if done:
                break

        # Get final score from environment
        try:
            score = await env_grade()
        except Exception:
            # Fallback scoring if server unreachable
            total_reward = sum(rewards)
            max_possible = MAX_STEPS * 5.0
            score = min(max(total_reward / max(max_possible, 1), 0.0), 1.0)

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
