"""
Autonomous Traffic Control Environment
=======================================
A real-world OpenEnv environment simulating a 4-way intersection
with multi-phase signal control and emergency vehicle prioritization.

Action Space:
    phase (int): 0-7 — which signal phase to activate
    duration (int): 5-60 — how long to hold the phase (seconds)
    emergency_override (bool): force emergency corridor if active

Observation Space:
    - Queue lengths per lane (8 lanes: N/S/E/W × 2 lanes each)
    - Current signal phase
    - Time remaining in current phase
    - Emergency vehicle presence (per direction, distance)
    - Ambulance/fire/police flags
    - Step number, time of day
    - Cumulative wait time

Tasks:
    1. basic_flow    (easy)   — minimize queue lengths, no emergency vehicles
    2. peak_hour     (medium) — rush-hour surge, dynamic demand, partial emergency
    3. full_control  (hard)   — emergency vehicles, incidents, pedestrian phases
"""

import asyncio
import copy
import json
import random
import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ─── Signal Phase Definitions ────────────────────────────────────────────────

class Phase(IntEnum):
    """
    Standard 8-phase NEMA signal plan:
      0: N-S through
      1: N-S left-turn protected
      2: E-W through
      3: E-W left-turn protected
      4: N pedestrian + S pedestrian
      5: E pedestrian + W pedestrian
      6: All-red / clearance
      7: Emergency corridor (clears requested direction)
    """
    NS_THROUGH    = 0
    NS_LEFT       = 1
    EW_THROUGH    = 2
    EW_LEFT       = 3
    PED_NS        = 4
    PED_EW        = 5
    ALL_RED       = 6
    EMERGENCY     = 7


# ─── Pydantic Models (OpenEnv spec) ──────────────────────────────────────────

class TrafficAction(BaseModel):
    phase: int = Field(..., ge=0, le=7, description="Signal phase index 0-7")
    duration: int = Field(default=30, ge=5, le=60, description="Phase duration in seconds")
    emergency_override: bool = Field(default=False, description="Force emergency corridor")

    class Config:
        json_schema_extra = {
            "example": {"phase": 0, "duration": 30, "emergency_override": False}
        }


class EmergencyVehicle(BaseModel):
    direction: str          # "N", "S", "E", "W"
    distance: float         # metres to intersection (0 = at intersection)
    vehicle_type: str       # "ambulance", "fire", "police"
    active: bool = True


class TrafficObservation(BaseModel):
    # Queue lengths per lane (vehicles waiting)
    queue_N1: int = 0; queue_N2: int = 0
    queue_S1: int = 0; queue_S2: int = 0
    queue_E1: int = 0; queue_E2: int = 0
    queue_W1: int = 0; queue_W2: int = 0

    # Signal state
    current_phase: int = 0
    phase_time_remaining: int = 30

    # Emergency vehicle info
    emergency_active: bool = False
    emergency_direction: str = ""
    emergency_distance: float = 999.0
    emergency_type: str = ""

    # Context
    step_number: int = 0
    time_of_day: float = 0.0      # 0.0-24.0 hours
    cumulative_wait: float = 0.0  # total vehicle-seconds waited
    incident_active: bool = False
    pedestrian_waiting: bool = False

    # Performance metrics
    throughput_last_phase: int = 0
    avg_wait_time: float = 0.0


class TrafficReward(BaseModel):
    value: float
    breakdown: Dict[str, float] = {}


class StepResult(BaseModel):
    observation: TrafficObservation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


# ─── Task Configs ─────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "basic_flow": {
        "description": "Manage a 4-way intersection under moderate steady traffic. No emergency vehicles.",
        "difficulty": "easy",
        "max_steps": 50,
        "base_arrival_rate": 0.3,    # vehicles/second per lane
        "emergency_prob": 0.0,
        "incident_prob": 0.0,
        "pedestrian_prob": 0.0,
        "time_of_day": 10.0,         # mid-morning
        "success_threshold": 0.5,
        "grader": "traffic_env:grade_task",
    },
    "peak_hour": {
        "description": "Rush-hour surge traffic (4-6 PM). Occasional emergency vehicles and pedestrian crossings.",
        "difficulty": "medium",
        "max_steps": 75,
        "base_arrival_rate": 0.7,
        "emergency_prob": 0.08,
        "incident_prob": 0.05,
        "pedestrian_prob": 0.15,
        "time_of_day": 17.0,
        "success_threshold": 0.6,
        "grader": "traffic_env:grade_task",
    },
    "full_control": {
        "description": "High-density traffic with frequent emergency vehicles, incidents, and pedestrian phases. Frontier-model challenge.",
        "difficulty": "hard",
        "max_steps": 100,
        "base_arrival_rate": 1.0,
        "emergency_prob": 0.18,
        "incident_prob": 0.12,
        "pedestrian_prob": 0.25,
        "time_of_day": 8.0,
        "success_threshold": 0.7,
        "grader": "traffic_env:grade_task",
    },
}


# ─── Reward Calculator ────────────────────────────────────────────────────────

def compute_reward(
    obs: TrafficObservation,
    action: TrafficAction,
    prev_obs: TrafficObservation,
    task_cfg: dict,
    emergency_vehicles: List[EmergencyVehicle],
    phase_throughput: int,
    emergency_delay_penalty: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Shaped reward function providing signal across full trajectory.

    Components:
      +queue_reduction:       reward for reducing total queue
      +throughput:            reward for vehicles passing through
      -wait_penalty:          penalty for long queues
      -emergency_penalty:     penalty if emergency vehicle delayed without override
      +emergency_bonus:       bonus for correctly activating emergency corridor
      -phase_oscillation:     penalty for rapid unnecessary phase switching
      -pedestrian_neglect:    penalty for ignoring pedestrian phases too long
    """
    breakdown: Dict[str, float] = {}

    # Total queue before and after
    def total_queue(o: TrafficObservation) -> int:
        return (o.queue_N1 + o.queue_N2 + o.queue_S1 + o.queue_S2 +
                o.queue_E1 + o.queue_E2 + o.queue_W1 + o.queue_W2)

    prev_q = total_queue(prev_obs)
    curr_q = total_queue(obs)
    delta_q = prev_q - curr_q

    # Queue reduction reward (positive if queues shrink)
    queue_reward = delta_q * 0.5
    breakdown["queue_reduction"] = round(queue_reward, 3)

    # Throughput reward
    throughput_reward = phase_throughput * 0.3
    breakdown["throughput"] = round(throughput_reward, 3)

    # Wait penalty (exponential to punish starvation)
    max_q = max(
        obs.queue_N1, obs.queue_N2, obs.queue_S1, obs.queue_S2,
        obs.queue_E1, obs.queue_E2, obs.queue_W1, obs.queue_W2
    )
    wait_penalty = -(curr_q * 0.1) - (max_q * 0.2 if max_q > 10 else 0)
    breakdown["wait_penalty"] = round(wait_penalty, 3)

    # Emergency vehicle handling
    emergency_reward = 0.0
    if emergency_vehicles:
        ev = emergency_vehicles[0]
        if action.emergency_override and ev.active:
            emergency_reward = 5.0   # strong positive for correct response
        elif not action.emergency_override and ev.distance < 100:
            emergency_reward = -emergency_delay_penalty  # penalise ignoring close EV
        breakdown["emergency"] = round(emergency_reward, 3)

    # Phase oscillation penalty (switching too fast is bad)
    phase_switch = 0.0
    if obs.current_phase != prev_obs.current_phase and action.duration < 10:
        phase_switch = -1.0
    breakdown["phase_oscillation"] = round(phase_switch, 3)

    # Pedestrian neglect penalty
    ped_penalty = 0.0
    if obs.pedestrian_waiting and action.phase not in (Phase.PED_NS, Phase.PED_EW):
        ped_penalty = -0.5
    breakdown["pedestrian_neglect"] = round(ped_penalty, 3)

    total = queue_reward + throughput_reward + wait_penalty + emergency_reward + phase_switch + ped_penalty

    # Clamp to prevent extreme values
    total = max(-20.0, min(20.0, total))
    return total, breakdown


# ─── Core Environment ─────────────────────────────────────────────────────────

class TrafficControlEnv:
    """
    OpenEnv-compliant environment for Autonomous Traffic Control.
    Manages a 4-way signalised intersection.
    """

    DIRECTIONS = ["N", "S", "E", "W"]

    def __init__(self, task: str = "basic_flow", seed: Optional[int] = None):
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_CONFIGS)}")
        self.task = task
        self.cfg = TASK_CONFIGS[task]
        self.seed = seed
        self._rng = random.Random(seed)
        self._step = 0
        self._obs: TrafficObservation = TrafficObservation()
        self._prev_obs: TrafficObservation = TrafficObservation()
        self._emergency_vehicles: List[EmergencyVehicle] = []
        self._phase_change_step = 0
        self._total_throughput = 0
        self._cumulative_reward = 0.0
        self._pedestrian_wait_steps = 0
        self._incident_lane: Optional[str] = None
        self._incident_remaining = 0

    # ── OpenEnv Interface ─────────────────────────────────────────────────────

    async def reset(self) -> StepResult:
        """Reset environment to initial state. Returns first observation."""
        self._rng = random.Random(self.seed)
        self._step = 0
        self._emergency_vehicles = []
        self._phase_change_step = 0
        self._total_throughput = 0
        self._cumulative_reward = 0.0
        self._pedestrian_wait_steps = 0
        self._incident_lane = None
        self._incident_remaining = 0

        obs = self._make_initial_obs()
        self._obs = obs
        self._prev_obs = copy.deepcopy(obs)

        return StepResult(observation=obs, reward=0.0, done=False, info={"reset": True})

    async def step(self, action: TrafficAction) -> StepResult:
        """Execute action, advance simulation one step, return result."""
        self._prev_obs = copy.deepcopy(self._obs)
        self._step += 1

        # Validate action
        if action.emergency_override and not self._obs.emergency_active:
            action = TrafficAction(
                phase=action.phase,
                duration=action.duration,
                emergency_override=False
            )

        # 1. Apply signal phase
        self._apply_phase(action)

        # 2. Simulate traffic arrivals
        self._simulate_arrivals()

        # 3. Simulate vehicles clearing on green
        throughput = self._simulate_departures(action)
        self._total_throughput += throughput

        # 4. Move emergency vehicles
        ev_delay = self._update_emergency_vehicles(action)

        # 5. Update incidents
        self._update_incident()

        # 6. Update pedestrians
        self._update_pedestrians(action)

        # 7. Build new observation
        obs = self._build_obs(action)
        self._obs = obs

        # 8. Compute reward
        reward, breakdown = compute_reward(
            obs=obs,
            action=action,
            prev_obs=self._prev_obs,
            task_cfg=self.cfg,
            emergency_vehicles=self._emergency_vehicles,
            phase_throughput=throughput,
            emergency_delay_penalty=ev_delay,
        )

        self._cumulative_reward += reward

        # 9. Check termination
        done = self._step >= self.cfg["max_steps"]

        info = {
            "step": self._step,
            "reward_breakdown": breakdown,
            "total_throughput": self._total_throughput,
            "cumulative_reward": round(self._cumulative_reward, 3),
            "task": self.task,
        }

        return StepResult(observation=obs, reward=round(reward, 4), done=done, info=info)

    async def state(self) -> Dict[str, Any]:
        """Return full internal state snapshot (for debugging/logging)."""
        return {
            "step": self._step,
            "task": self.task,
            "observation": self._obs.model_dump(),
            "emergency_vehicles": [ev.model_dump() for ev in self._emergency_vehicles],
            "cumulative_reward": self._cumulative_reward,
            "total_throughput": self._total_throughput,
            "incident_lane": self._incident_lane,
        }

    def grade(self) -> float:
        """
        Score the episode: normalised to [0, 1].
        Based on throughput efficiency, wait minimisation, and emergency handling.
        """
        max_steps = self.cfg["max_steps"]
        if max_steps == 0:
            return 0.0

        # Throughput score: compare to theoretical max (assume 5 vehicles/step max)
        max_possible_throughput = max_steps * 5
        throughput_score = min(self._total_throughput / max(max_possible_throughput, 1), 1.0)

        # Queue score: invert final queue (lower is better)
        final_q = (self._obs.queue_N1 + self._obs.queue_N2 +
                   self._obs.queue_S1 + self._obs.queue_S2 +
                   self._obs.queue_E1 + self._obs.queue_E2 +
                   self._obs.queue_W1 + self._obs.queue_W2)
        max_queue = 80  # 10 vehicles per lane × 8 lanes
        queue_score = max(0.0, 1.0 - final_q / max_queue)

        # Reward score: normalise cumulative reward
        max_reward = max_steps * 5.0  # rough upper bound
        reward_score = min(max(self._cumulative_reward / max(max_reward, 1), 0.0), 1.0)

        # Weighted combination
        score = 0.4 * throughput_score + 0.4 * queue_score + 0.2 * reward_score
        # Clamp to open interval (0, 1) — validator rejects exactly 0.0 and 1.0
        score = max(0.001, min(0.999, score))
        return round(score, 4)

    # ── Internal Simulation ───────────────────────────────────────────────────

    def _make_initial_obs(self) -> TrafficObservation:
        rate = self.cfg["base_arrival_rate"]
        return TrafficObservation(
            queue_N1=int(rate * 5), queue_N2=int(rate * 3),
            queue_S1=int(rate * 4), queue_S2=int(rate * 4),
            queue_E1=int(rate * 6), queue_E2=int(rate * 2),
            queue_W1=int(rate * 5), queue_W2=int(rate * 3),
            current_phase=Phase.NS_THROUGH,
            phase_time_remaining=30,
            step_number=0,
            time_of_day=self.cfg["time_of_day"],
        )

    def _apply_phase(self, action: TrafficAction):
        if action.emergency_override:
            self._obs.current_phase = Phase.EMERGENCY
        else:
            self._obs.current_phase = action.phase
        self._obs.phase_time_remaining = action.duration
        self._phase_change_step = self._step

    def _arrival_count(self, base_rate: float) -> int:
        """Poisson-like arrival with time-of-day modulation."""
        tod = self.cfg["time_of_day"]
        # Rush hour multiplier
        if 7 <= tod <= 9 or 16 <= tod <= 19:
            multiplier = 1.8
        elif 22 <= tod or tod <= 6:
            multiplier = 0.3
        else:
            multiplier = 1.0
        lam = base_rate * multiplier
        # Approximate Poisson with binomial
        return sum(1 for _ in range(10) if self._rng.random() < lam / 10)

    def _simulate_arrivals(self):
        rate = self.cfg["base_arrival_rate"]
        cap = 20  # max queue per lane
        incident_mult = 0.3 if self._incident_lane else 1.0

        lanes = [
            ("queue_N1", rate), ("queue_N2", rate * 0.7),
            ("queue_S1", rate), ("queue_S2", rate * 0.7),
            ("queue_E1", rate * incident_mult), ("queue_E2", rate * 0.6 * incident_mult),
            ("queue_W1", rate), ("queue_W2", rate * 0.7),
        ]
        for attr, r in lanes:
            arrived = self._arrival_count(r)
            current = getattr(self._obs, attr)
            setattr(self._obs, attr, min(current + arrived, cap))

        # Random emergency vehicle spawn
        if self._rng.random() < self.cfg["emergency_prob"] and not self._emergency_vehicles:
            ev_dir = self._rng.choice(self.DIRECTIONS)
            ev_type = self._rng.choice(["ambulance", "fire", "police"])
            self._emergency_vehicles.append(EmergencyVehicle(
                direction=ev_dir, distance=250.0, vehicle_type=ev_type
            ))

        # Random incident
        if self._rng.random() < self.cfg["incident_prob"] and not self._incident_lane:
            self._incident_lane = self._rng.choice(["queue_E1", "queue_W1"])
            self._incident_remaining = self._rng.randint(5, 15)

        # Pedestrian demand
        if self._rng.random() < self.cfg["pedestrian_prob"]:
            self._pedestrian_wait_steps += 1

    def _simulate_departures(self, action: TrafficAction) -> int:
        """Vehicles depart on green. Returns throughput count."""
        phase = action.phase if not action.emergency_override else Phase.EMERGENCY
        throughput = 0

        def depart(attr: str, max_depart: int) -> int:
            current = getattr(self._obs, attr)
            leaving = min(current, max_depart)
            setattr(self._obs, attr, current - leaving)
            return leaving

        # Each phase clears specific lanes
        if phase == Phase.NS_THROUGH:
            throughput += depart("queue_N1", 3) + depart("queue_S1", 3)
        elif phase == Phase.NS_LEFT:
            throughput += depart("queue_N2", 2) + depart("queue_S2", 2)
        elif phase == Phase.EW_THROUGH:
            throughput += depart("queue_E1", 3) + depart("queue_W1", 3)
        elif phase == Phase.EW_LEFT:
            throughput += depart("queue_E2", 2) + depart("queue_W2", 2)
        elif phase == Phase.PED_NS:
            # Pedestrians cross, no vehicles
            if self._pedestrian_wait_steps > 0:
                self._pedestrian_wait_steps = max(0, self._pedestrian_wait_steps - 3)
        elif phase == Phase.PED_EW:
            if self._pedestrian_wait_steps > 0:
                self._pedestrian_wait_steps = max(0, self._pedestrian_wait_steps - 3)
        elif phase == Phase.ALL_RED:
            pass  # clearance, no departures
        elif phase == Phase.EMERGENCY:
            # Clear the emergency direction
            if self._emergency_vehicles:
                ev_dir = self._emergency_vehicles[0].direction
                if ev_dir == "N":
                    throughput += depart("queue_N1", 5) + depart("queue_N2", 5)
                elif ev_dir == "S":
                    throughput += depart("queue_S1", 5) + depart("queue_S2", 5)
                elif ev_dir == "E":
                    throughput += depart("queue_E1", 5) + depart("queue_E2", 5)
                elif ev_dir == "W":
                    throughput += depart("queue_W1", 5) + depart("queue_W2", 5)

        return throughput

    def _update_emergency_vehicles(self, action: TrafficAction) -> float:
        """Move EVs closer; return delay penalty if ignored."""
        penalty = 0.0
        updated = []
        for ev in self._emergency_vehicles:
            speed = 15.0 if action.emergency_override else 5.0
            ev.distance = max(0.0, ev.distance - speed)
            if ev.distance <= 0:
                ev.active = False  # passed through
            else:
                updated.append(ev)
                # Accumulate penalty for every step ignored when close
                if ev.distance < 100 and not action.emergency_override:
                    penalty += 2.0
        self._emergency_vehicles = [ev for ev in updated if ev.active]
        return penalty

    def _update_incident(self):
        if self._incident_remaining > 0:
            self._incident_remaining -= 1
        else:
            self._incident_lane = None

    def _update_pedestrians(self, action: TrafficAction):
        if action.phase in (Phase.PED_NS, Phase.PED_EW):
            self._pedestrian_wait_steps = max(0, self._pedestrian_wait_steps - 2)

    def _build_obs(self, action: TrafficAction) -> TrafficObservation:
        ev = self._emergency_vehicles[0] if self._emergency_vehicles else None

        # Compute cumulative wait as sum of all queues over all steps
        total_q = (self._obs.queue_N1 + self._obs.queue_N2 +
                   self._obs.queue_S1 + self._obs.queue_S2 +
                   self._obs.queue_E1 + self._obs.queue_E2 +
                   self._obs.queue_W1 + self._obs.queue_W2)

        return TrafficObservation(
            queue_N1=self._obs.queue_N1, queue_N2=self._obs.queue_N2,
            queue_S1=self._obs.queue_S1, queue_S2=self._obs.queue_S2,
            queue_E1=self._obs.queue_E1, queue_E2=self._obs.queue_E2,
            queue_W1=self._obs.queue_W1, queue_W2=self._obs.queue_W2,
            current_phase=action.phase if not action.emergency_override else Phase.EMERGENCY,
            phase_time_remaining=action.duration,
            emergency_active=bool(self._emergency_vehicles),
            emergency_direction=ev.direction if ev else "",
            emergency_distance=ev.distance if ev else 999.0,
            emergency_type=ev.vehicle_type if ev else "",
            step_number=self._step,
            time_of_day=self.cfg["time_of_day"],
            cumulative_wait=float(self._obs.cumulative_wait + total_q),
            incident_active=self._incident_lane is not None,
            pedestrian_waiting=self._pedestrian_wait_steps > 0,
            throughput_last_phase=self._total_throughput,
            avg_wait_time=round(self._obs.cumulative_wait / max(self._step, 1), 2),
        )


# ─── Grader Functions ─────────────────────────────────────────────────────────

async def grade_task(task: str, agent_actions: List[Dict], seed: int = 42) -> float:
    """
    Run a full episode using provided actions and return score in [0, 1].
    agent_actions: list of dicts with keys matching TrafficAction fields.
    """
    env = TrafficControlEnv(task=task, seed=seed)
    await env.reset()
    max_steps = TASK_CONFIGS[task]["max_steps"]

    for i in range(max_steps):
        if i < len(agent_actions):
            raw = agent_actions[i]
        else:
            raw = {"phase": 0, "duration": 30, "emergency_override": False}
        action = TrafficAction(**raw)
        result = await env.step(action)
        if result.done:
            break

    return env.grade()


# ─── Quick smoke test ─────────────────────────────────────────────────────────

async def _smoke_test():
    for task in TASK_CONFIGS:
        env = TrafficControlEnv(task=task, seed=0)
        result = await env.reset()
        assert result.observation is not None
        for i in range(5):
            phase = i % 8
            action = TrafficAction(phase=phase, duration=20, emergency_override=False)
            result = await env.step(action)
            assert 0.0 <= result.reward  # reward could be negative, just check it runs
        score = env.grade()
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for task {task}"
        print(f"  [OK] task={task} score={score:.4f}")


if __name__ == "__main__":
    print("Running smoke tests...")
    asyncio.run(_smoke_test())
    print("All smoke tests passed.")
