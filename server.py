"""
Traffic Control OpenEnv — FastAPI Server
=========================================
Serves the environment over HTTP following the OpenEnv REST spec.

Endpoints:
  POST /reset          — reset environment, returns first observation
  POST /step           — execute action, returns (obs, reward, done, info)
  GET  /state          — returns full internal state
  GET  /health         — health check (returns 200)
  GET  /openenv.yaml   — environment metadata

Run with:
  uvicorn server:app --host 0.0.0.0 --port 7860
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from traffic_env import (
    TrafficControlEnv,
    TrafficAction,
    TrafficObservation,
    StepResult,
    TASK_CONFIGS,
)

# ─── State ────────────────────────────────────────────────────────────────────

_env: Optional[TrafficControlEnv] = None
_current_task: str = os.getenv("TRAFFIC_TASK", "basic_flow")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    _env = TrafficControlEnv(task=_current_task)
    await _env.reset()
    yield


app = FastAPI(
    title="Autonomous Traffic Control OpenEnv",
    description="4-way intersection RL environment with emergency vehicle prioritization",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Request/Response Models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any] = {}


class StepRequest(BaseModel):
    action: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any] = {}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "task": _current_task}


@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest = ResetRequest()):
    global _env, _current_task

    task = req.task or _current_task
    if task not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'. Valid: {list(TASK_CONFIGS)}")

    _current_task = task
    _env = TrafficControlEnv(task=task, seed=req.seed)
    result = await _env.reset()

    return ResetResponse(
        observation=result.observation.model_dump(),
        info=result.info,
    )


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    try:
        action = TrafficAction(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    result = await _env.step(action)
    return StepResponse(
        observation=result.observation.model_dump(),
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state")
async def state():
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised.")
    return await _env.state()


@app.get("/grade")
async def grade():
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised.")
    score = _env.grade()
    return {"score": score, "task": _current_task}


@app.get("/tasks")
async def list_tasks():
    tasks_list = [
        {
            "id": name,
            "name": name,
            "difficulty": cfg["difficulty"],
            "description": cfg["description"],
            "grader": cfg.get("grader", "traffic_env:grade_task"),
        }
        for name, cfg in TASK_CONFIGS.items()
    ]
    return {"tasks": tasks_list}


@app.get("/openenv.yaml", response_class=PlainTextResponse)
async def openenv_yaml():
    with open("openenv.yaml", "r") as f:
        return f.read()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
