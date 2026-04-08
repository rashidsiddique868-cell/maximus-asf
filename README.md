---
title: Autonomous Traffic Control OpenEnv
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
---
# 🚦 Autonomous Traffic Control — OpenEnv

> **Meta PyTorch OpenEnv Hackathon Submission**  
> A real-world RL environment for 4-way intersection signal control with emergency vehicle prioritization.

---

## Overview

This environment simulates a **signalised 4-way intersection** where an AI agent controls traffic signals across 8 lanes to:

- Minimise vehicle queue lengths and waiting time
- Respond correctly to approaching emergency vehicles (ambulance, fire, police)
- Handle pedestrian crossing phases
- Manage road incidents that block lanes
- Adapt to time-of-day traffic demand patterns (rush hour, night, etc.)

Unlike toy environments, this models real traffic engineering problems that cities face daily.

---

## Tasks

| Task | Difficulty | Max Steps | Success Threshold | Description |
|------|------------|-----------|-------------------|-------------|
| `basic_flow` | Easy | 50 | 0.50 | Steady moderate traffic, no emergencies |
| `peak_hour` | Medium | 75 | 0.60 | Rush-hour surge, occasional emergency vehicles |
| `full_control` | Hard | 100 | 0.70 | High density + frequent emergencies + incidents |

---

## Action Space

```json
{
  "phase": 0,               // int [0-7] — signal phase (see below)
  "duration": 30,           // int [5-60] — seconds to hold this phase
  "emergency_override": false  // bool — force emergency corridor
}
```

### Signal Phases (NEMA 8-phase)

| Phase | Name | Description |
|-------|------|-------------|
| 0 | NS Through | North-South straight traffic |
| 1 | NS Left | North-South left-turn protected |
| 2 | EW Through | East-West straight traffic |
| 3 | EW Left | East-West left-turn protected |
| 4 | Pedestrian NS | Pedestrians cross N-S |
| 5 | Pedestrian EW | Pedestrians cross E-W |
| 6 | All-Red | Full clearance interval |
| 7 | Emergency Corridor | Clears path for approaching EV |

---

## Observation Space

```json
{
  "queue_N1": 4,           "queue_N2": 2,
  "queue_S1": 3,           "queue_S2": 1,
  "queue_E1": 7,           "queue_E2": 0,
  "queue_W1": 5,           "queue_W2": 2,
  "current_phase": 0,
  "phase_time_remaining": 25,
  "emergency_active": true,
  "emergency_direction": "E",
  "emergency_distance": 87.5,
  "emergency_type": "ambulance",
  "step_number": 12,
  "time_of_day": 17.0,
  "cumulative_wait": 432.0,
  "incident_active": false,
  "pedestrian_waiting": true,
  "throughput_last_phase": 38,
  "avg_wait_time": 36.0
}
```

---

## Reward Function

Reward is provided **every step** (dense, shaped — not sparse):

| Component | Value | Trigger |
|-----------|-------|---------|
| `queue_reduction` | `+0.5 × delta_queue` | Queues shrink |
| `throughput` | `+0.3 × vehicles_cleared` | Vehicles pass through |
| `wait_penalty` | `−0.1 × total_queue` | Queues exist |
| `emergency_bonus` | `+5.0` | Correct emergency override |
| `emergency_penalty` | `−2.0/step` | EV ignored within 100m |
| `phase_oscillation` | `−1.0` | Rapid unnecessary switching |
| `pedestrian_neglect` | `−0.5` | Pedestrians ignored |

Range: `[−20, +20]` per step.

---

## Quick Start

### Local (Python)

```bash
pip install -r requirements.txt

# Run validation
python validate.py

# Start server
uvicorn server:app --host 0.0.0.0 --port 7860

# Run inference (separate terminal)
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export TRAFFIC_TASK=basic_flow
python inference.py
```

### Docker

```bash
docker build -t traffic-control-env .
docker run -p 7860:7860 -e TRAFFIC_TASK=basic_flow traffic-control-env
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/reset` | Reset environment |
| POST | `/step` | Execute action |
| GET | `/state` | Full internal state |
| GET | `/grade` | Current episode score |
| GET | `/tasks` | List all tasks |
| GET | `/openenv.yaml` | Environment metadata |

### Example Usage

```python
import httpx

# Reset
r = httpx.post("http://localhost:7860/reset", json={"task": "basic_flow"})
obs = r.json()["observation"]

# Step
r = httpx.post("http://localhost:7860/step", json={
    "action": {"phase": 2, "duration": 30, "emergency_override": False}
})
print(r.json())  # {"observation": {...}, "reward": 2.1, "done": false, "info": {...}}

# Grade
r = httpx.get("http://localhost:7860/grade")
print(r.json())  # {"score": 0.72}
```

---

## Baseline Scores

Tested with `Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Score | Success |
|------|-------|---------|
| `basic_flow` | ~0.61 | ✅ |
| `peak_hour` | ~0.58 | ~borderline |
| `full_control` | ~0.49 | ❌ (intentionally hard) |

---

## Project Structure

```
traffic_control_env/
├── traffic_env.py      # Core environment (models, simulation, grader)
├── server.py           # FastAPI HTTP server (OpenEnv REST API)
├── inference.py        # Hackathon inference script
├── validate.py         # Pre-submission validation
├── openenv.yaml        # Environment metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Design Decisions

**Why traffic control?** It's a high-stakes real-world problem. Cities worldwide use adaptive signal control — but current systems are rule-based. An RL environment here directly enables research into smarter urban mobility.

**Why shaped rewards?** Sparse rewards (only at episode end) make learning extremely hard for 50-100 step episodes. Each component of the reward function maps to a measurable traffic KPI (throughput, delay, safety).

**Why 8 phases?** NEMA 8-phase is the real-world standard used in North American traffic controllers. This makes the environment directly applicable to real deployments.

**Emergency vehicle design:** The 100m threshold and approach speed mechanics model actual preemption systems (OPTICOM/GPS-based) used by fire departments.
