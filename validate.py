"""
Pre-Submission Validation Script
==================================
Run this before submitting to verify your environment passes all checks.
Usage: python validate.py
"""

import asyncio
import json
import sys

# ─── Imports ─────────────────────────────────────────────────────────────────

try:
    from traffic_env import (
        TrafficControlEnv,
        TrafficAction,
        TASK_CONFIGS,
        grade_task,
    )
    print("[OK] traffic_env imports successfully")
except ImportError as e:
    print(f"[FAIL] Cannot import traffic_env: {e}")
    sys.exit(1)

try:
    import yaml
    with open("openenv.yaml") as f:
        meta = yaml.safe_load(f)
    assert "name" in meta
    assert "tasks" in meta
    assert "action_space" in meta
    assert "observation_space" in meta
    print("[OK] openenv.yaml is valid")
except Exception as e:
    print(f"[FAIL] openenv.yaml: {e}")
    sys.exit(1)


# ─── Tests ────────────────────────────────────────────────────────────────────

async def test_reset():
    """reset() returns a valid StepResult."""
    for task in TASK_CONFIGS:
        env = TrafficControlEnv(task=task, seed=42)
        result = await env.reset()
        assert result.observation is not None, f"[{task}] observation is None"
        assert result.done == False, f"[{task}] done=True after reset"
        assert result.reward == 0.0, f"[{task}] non-zero reward on reset"
        obs = result.observation
        # Check all queue fields exist
        for lane in ["N1","N2","S1","S2","E1","E2","W1","W2"]:
            val = getattr(obs, f"queue_{lane}")
            assert val >= 0, f"[{task}] queue_{lane} negative"
    print("[OK] reset() — all tasks pass")


async def test_step():
    """step() returns correct types and reward in expected range."""
    for task in TASK_CONFIGS:
        env = TrafficControlEnv(task=task, seed=0)
        await env.reset()
        for phase in range(8):
            action = TrafficAction(phase=phase, duration=20, emergency_override=False)
            result = await env.step(action)
            assert isinstance(result.reward, float), f"reward not float in {task}"
            assert isinstance(result.done, bool), f"done not bool in {task}"
            assert result.observation is not None
        # Emergency override
        action = TrafficAction(phase=7, duration=15, emergency_override=True)
        result = await env.step(action)
        assert result.observation is not None
    print("[OK] step() — all phases and emergency override pass")


async def test_state():
    """state() returns a dict with expected keys."""
    env = TrafficControlEnv(task="basic_flow", seed=1)
    await env.reset()
    await env.step(TrafficAction(phase=0, duration=30, emergency_override=False))
    state = await env.state()
    assert "step" in state
    assert "observation" in state
    assert "cumulative_reward" in state
    print("[OK] state() — returns expected keys")


async def test_grade():
    """grade() returns float in [0, 1] for all tasks."""
    for task in TASK_CONFIGS:
        env = TrafficControlEnv(task=task, seed=99)
        await env.reset()
        max_steps = TASK_CONFIGS[task]["max_steps"]
        for i in range(min(max_steps, 10)):
            action = TrafficAction(phase=i % 6, duration=25, emergency_override=False)
            result = await env.step(action)
            if result.done:
                break
        score = env.grade()
        assert 0.0 <= score <= 1.0, f"score {score} out of [0,1] for {task}"
    print("[OK] grade() — scores in [0, 1] for all tasks")


async def test_three_tasks():
    """Verify exactly 3 tasks are defined and each has required fields."""
    assert len(TASK_CONFIGS) >= 3, f"Only {len(TASK_CONFIGS)} tasks, need at least 3"
    for name, cfg in TASK_CONFIGS.items():
        assert "difficulty" in cfg, f"Task {name} missing 'difficulty'"
        assert "max_steps" in cfg, f"Task {name} missing 'max_steps'"
        assert "success_threshold" in cfg, f"Task {name} missing 'success_threshold'"
        assert cfg["max_steps"] > 0
    print(f"[OK] {len(TASK_CONFIGS)} tasks defined with required fields")


async def test_difficulty_range():
    """Easy < medium < hard in terms of difficulty."""
    difficulties = ["easy", "medium", "hard"]
    found = {cfg["difficulty"] for cfg in TASK_CONFIGS.values()}
    for d in difficulties:
        assert d in found, f"Missing difficulty level: {d}"
    print("[OK] Difficulty range: easy → medium → hard")


async def test_reproducibility():
    """Same seed produces same sequence of rewards."""
    rewards_a = []
    rewards_b = []
    for seed_run in range(2):
        env = TrafficControlEnv(task="basic_flow", seed=7)
        await env.reset()
        for i in range(5):
            action = TrafficAction(phase=i % 4, duration=20, emergency_override=False)
            r = await env.step(action)
            if seed_run == 0:
                rewards_a.append(r.reward)
            else:
                rewards_b.append(r.reward)
    assert rewards_a == rewards_b, f"Non-reproducible: {rewards_a} vs {rewards_b}"
    print("[OK] Reproducibility — same seed gives same rewards")


async def test_grade_task_helper():
    """grade_task() helper function works correctly."""
    actions = [{"phase": i % 4, "duration": 25, "emergency_override": False} for i in range(50)]
    score = await grade_task("basic_flow", actions, seed=0)
    assert 0.0 <= score <= 1.0
    print(f"[OK] grade_task() helper — score={score:.4f}")


# ─── Run all ──────────────────────────────────────────────────────────────────

async def main():
    print("\n" + "="*55)
    print("  Autonomous Traffic Control — Pre-Submission Validator")
    print("="*55 + "\n")

    tests = [
        test_reset,
        test_step,
        test_state,
        test_grade,
        test_three_tasks,
        test_difficulty_range,
        test_reproducibility,
        test_grade_task_helper,
    ]

    failed = 0
    for test_fn in tests:
        try:
            await test_fn()
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print("\n" + "="*55)
    if failed == 0:
        print("  ✅ ALL CHECKS PASSED — Ready to submit!")
    else:
        print(f"  ❌ {failed} check(s) FAILED — Fix before submitting.")
    print("="*55 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
