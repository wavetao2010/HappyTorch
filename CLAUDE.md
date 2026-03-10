# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup (first time)
conda create -n torchcode python=3.11 -y
conda activate torchcode
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jupyterlab numpy
pip install -e .

# Install web dependencies (for web mode)
pip install fastapi uvicorn python-multipart

# Prepare notebooks (for Jupyter mode)
python prepare_notebooks.py

# Start Web Mode (LeetCode-like interface)
python start_web.py
# Open http://localhost:8000

# Start Jupyter Mode
python start_jupyter.py

# Start Web Plugin (standalone, for other TorchCode projects)
cd web-plugin && pip install -r requirements.txt && python start.py

# Docker
make run    # docker compose up (supports podman and docker)
make stop   # docker compose down
make clean  # full cleanup including progress.json
```

Python 3.10+ required. PyTorch CPU-only, no GPU needed.

## Architecture

HappyTorch is a PyTorch coding practice platform (LeetCode for tensors). Based on [TorchCode](https://github.com/duoan/TorchCode) with additional problems for LLM, Diffusion, PEFT, and RLHF topics.

Two interfaces: a LeetCode-like web UI (FastAPI + Monaco Editor SPA) and Jupyter notebooks. Both share the same `torch_judge` auto-grading engine.

### Core Components

**torch_judge/** — Auto-grading engine (installed via `pip install -e .`)
- `__init__.py` — Public API: `check()`, `hint()`, `status()`, `reset_progress()`
- `engine.py` — Extracts user function from IPython namespace via `get_ipython().user_ns`, runs tests via `exec()` with `{fn}` placeholder substitution
- `progress.py` — Tracks solved/attempted status in `data/progress.json` (path configurable via `PROGRESS_PATH` env var)
- `tasks/_registry.py` — Auto-discovers all TASK dicts from sibling modules using `pkgutil.iter_modules()`
- `tasks/*.py` — ~37 task definition files, each exporting a `TASK` dictionary

**web/app.py** — FastAPI backend serving the SPA and REST API endpoints:
- `GET /api/tasks` — List all tasks
- `GET /api/tasks/{task_id}` — Task details (includes template, signature, example, has_solution)
- `POST /api/submit` — Submit code for testing
- `GET /api/tasks/{task_id}/solution` — Get solution markdown and code
- `GET /api/progress` — User progress
- `POST /api/reset` — Reset progress
- `GET /api/random` — Random unsolved task

**web-plugin/** — Standalone drop-in web interface that auto-discovers tasks from the parent project's `torch_judge` module. No configuration needed when adding new problems.

### Task Definition Format

Every task file in `torch_judge/tasks/` exports a `TASK` dict:

```python
TASK = {
    "title": "Implement X",
    "difficulty": "Easy",  # Easy/Medium/Hard
    "function_name": "my_function",
    "hint": "Think about...",
    "tests": [
        {"name": "Basic test", "code": "assert torch.allclose(...)"},
    ]
}
```

Tasks are auto-registered — just add a new `.py` file to `tasks/` with a `TASK` dict and it will be discovered automatically.

### Workflow

1. User implements a function (e.g., `relu(x)`) in a notebook or the web editor
2. `check("relu")` extracts the function from the user namespace and runs test assertions
3. Results shown with pass/fail per test case; progress saved to JSON

### Key Directories

- `templates/` — Blank practice notebooks
- `solutions/` — Reference implementations
- `notebooks/` — User workspace (created at runtime)
- `data/` — Progress tracking (auto-created)
