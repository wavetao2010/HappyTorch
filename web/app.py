"""FastAPI backend for HappyTorch web interface."""

from __future__ import annotations

import json
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_judge.tasks import get_task, list_tasks
from torch_judge.progress import _load, mark_solved, mark_attempted

app = FastAPI(title="HappyTorch", description="PyTorch Interview Practice Platform")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


# ==================== Models ====================

class SubmitRequest(BaseModel):
    task_id: str
    code: str


class SubmitResponse(BaseModel):
    success: bool
    passed: int
    total: int
    total_time: float
    results: list[dict[str, Any]]
    output: str


class TaskResponse(BaseModel):
    id: str
    title: str
    difficulty: str
    hint: str
    function_name: str
    description: str


class TaskDetailResponse(BaseModel):
    id: str
    title: str
    difficulty: str
    hint: str
    function_name: str
    description: str
    template: str
    signature: str
    example: str
    tests_count: int


class ProgressResponse(BaseModel):
    solved: int
    total: int
    tasks: list[dict[str, Any]]


# ==================== Helper Functions ====================

def _find_template_path(task_id: str) -> Path | None:
    """Find template notebook path by task_id.
    
    Template files are named like '01_relu.ipynb', task_id is 'relu'.
    """
    templates_dir = Path(__file__).parent.parent / "templates"
    
    # Try exact match first
    exact = templates_dir / f"{task_id}.ipynb"
    if exact.exists():
        return exact
    
    # Try with number prefix (e.g., 01_relu.ipynb)
    for f in templates_dir.glob("*.ipynb"):
        # Match patterns like "01_relu.ipynb" where task_id is "relu"
        name = f.stem  # "01_relu"
        if name.endswith(f"_{task_id}") or name == task_id:
            return f
        # Also try matching the task_id directly in the name
        parts = name.split("_", 1)
        if len(parts) == 2 and parts[1] == task_id:
            return f
    
    return None


def _get_task_description(task_id: str) -> str:
    """Get task description from template notebook or generate from task.
    
    Note: Signature and Example sections are removed since they're shown separately.
    """
    template_path = _find_template_path(task_id)
    
    # Try to get from template notebook markdown
    if template_path and template_path.exists():
        try:
            with open(template_path, encoding="utf-8") as f:
                nb = json.load(f)
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "markdown":
                    source = "".join(cell.get("source", []))
                    if task_id in source.lower() or "implement" in source.lower():
                        # Remove Signature and Example sections since they're shown separately
                        return _clean_description(source.strip())
        except Exception:
            pass
    
    # Fallback to task info
    task = get_task(task_id)
    if task:
        return f"Implement `{task['function_name']}` - {task['title']}"
    return ""


def _extract_signature_from_markdown(markdown: str) -> str:
    """Extract function signature from markdown."""
    import re
    # Look for signature in code block after "### Signature"
    pattern = r'###\s*Signature\s*```python\s*(.*?)```'
    match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _extract_example_from_markdown(markdown: str) -> str:
    """Extract example from markdown."""
    import re
    # Look for example after "### Example"
    pattern = r'###\s*Example\s*```\s*(.*?)```'
    match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _clean_description(markdown: str) -> str:
    """Remove Signature and Example sections from description since they're shown separately."""
    import re
    # Remove Signature section (### Signature ... ```python ... ```)
    markdown = re.sub(
        r'###\s*Signature\s*```python\s*.*?```',
        '',
        markdown,
        flags=re.DOTALL | re.IGNORECASE
    )
    # Remove Example section (### Example ... ``` ... ```)
    markdown = re.sub(
        r'###\s*Example\s*```\s*.*?```',
        '',
        markdown,
        flags=re.DOTALL | re.IGNORECASE
    )
    # Clean up extra blank lines
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    return markdown.strip()


def _get_template_code(task_id: str) -> tuple[str, str, str]:
    """Extract template code, signature, and example from notebook.
    
    Returns: (template_code, signature, example)
    """
    template_path = _find_template_path(task_id)
    
    signature = ""
    example = ""
    template_code = ""
    markdown_content = ""
    
    if template_path and template_path.exists():
        try:
            with open(template_path, encoding="utf-8") as f:
                nb = json.load(f)
            
            # Extract markdown content
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "markdown":
                    source = "".join(cell.get("source", []))
                    markdown_content += source + "\n\n"
            
            # Extract signature and example from markdown
            signature = _extract_signature_from_markdown(markdown_content)
            example = _extract_example_from_markdown(markdown_content)
            
            # Extract template code
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = "".join(cell.get("source", []))
                    if "TODO" in source or "def " in source or "class " in source:
                        # Skip import-only cells
                        if source.strip().startswith("import") and "\n" not in source.strip():
                            continue
                        template_code = source.strip()
                        break
        except Exception:
            pass
    
    # Generate template if not found
    if not template_code:
        task = get_task(task_id)
        if task:
            fn = task["function_name"]
            if fn[0].isupper():
                template_code = f"class {fn}:\n    def __init__(self, ...):\n        # TODO: implement\n        pass\n"
                if not signature:
                    signature = f"class {fn}:\n    def __init__(self, ...)"
            else:
                template_code = f"def {fn}(...):\n    # TODO: implement\n    pass\n"
                if not signature:
                    signature = f"def {fn}(...)"
    
    return template_code, signature, example


def _run_tests(task_id: str, code: str) -> tuple[int, int, float, list[dict], str]:
    """Execute user code and run tests. Returns (passed, total, time, results, output)."""
    task = get_task(task_id)
    if not task:
        return 0, 0, 0.0, [], "Task not found"
    
    fn_name = task["function_name"]
    tests = task["tests"]
    
    # Capture stdout/stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    results = []
    passed = 0
    total = len(tests)
    total_time = 0.0
    
    # Prepare namespace with torch
    namespace: dict[str, Any] = {
        "torch": torch,
        "__builtins__": __builtins__,
    }
    
    # Execute user code
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compile(code, "<user_code>", "exec"), namespace)
    except SyntaxError as e:
        return 0, total, 0.0, [], f"Syntax Error: {e}"
    except Exception as e:
        return 0, total, 0.0, [], f"Code execution error: {type(e).__name__}: {e}"
    
    # Check if function/class exists
    if fn_name not in namespace:
        return 0, total, 0.0, [], f"Function/class '{fn_name}' not found in your code."
    
    user_fn = namespace[fn_name]
    test_namespace = {**namespace, fn_name: user_fn}
    
    # Run each test
    for i, test in enumerate(tests, 1):
        test_code = test["code"].replace("{fn}", fn_name)
        t0 = time.perf_counter()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compile(test_code, f"<test:{test['name']}>", "exec"), test_namespace)
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            passed += 1
            results.append({
                "name": test["name"],
                "passed": True,
                "time": elapsed,
                "error": None
            })
        except AssertionError as e:
            elapsed = time.perf_counter() - t0
            results.append({
                "name": test["name"],
                "passed": False,
                "time": elapsed,
                "error": str(e) or "Assertion failed"
            })
        except Exception as e:
            elapsed = time.perf_counter() - t0
            tb = traceback.format_exc()
            results.append({
                "name": test["name"],
                "passed": False,
                "time": elapsed,
                "error": f"{type(e).__name__}: {e}\n{tb}"
            })
    
    output = stdout_capture.getvalue()
    if stderr_capture.getvalue():
        output += "\n" + stderr_capture.getvalue()
    
    return passed, total, total_time, results, output.strip()


# ==================== API Routes ====================

@app.get("/")
async def root():
    """Serve the main page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>HappyTorch</h1><p>Static files not found</p>")


@app.get("/api/tasks")
async def get_tasks():
    """Get all available tasks."""
    tasks = []
    for task_id, task in list_tasks():
        tasks.append({
            "id": task_id,
            "title": task["title"],
            "difficulty": task["difficulty"],
            "function_name": task["function_name"],
        })
    return {"tasks": tasks}


@app.get("/api/tasks/{task_id}")
async def get_task_detail(task_id: str):
    """Get details for a specific task."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    template, signature, example = _get_template_code(task_id)
    
    return {
        "id": task_id,
        "title": task["title"],
        "difficulty": task["difficulty"],
        "hint": task["hint"],
        "function_name": task["function_name"],
        "description": _get_task_description(task_id),
        "template": template,
        "signature": signature,
        "example": example,
        "tests_count": len(task["tests"]),
    }


@app.get("/api/random")
async def get_random_task():
    """Get a random unsolved task, or random task if all solved."""
    import random
    
    progress = _load()
    tasks = list_tasks()
    
    # Find unsolved tasks
    unsolved = [(tid, t) for tid, t in tasks if progress.get(tid, {}).get("status") != "solved"]
    
    if unsolved:
        task_id, task = random.choice(unsolved)
    else:
        task_id, task = random.choice(tasks)
    
    return {
        "id": task_id,
        "title": task["title"],
        "difficulty": task["difficulty"],
        "function_name": task["function_name"],
    }


@app.get("/api/progress")
async def get_progress():
    """Get user progress."""
    progress = _load()
    tasks = list_tasks()
    
    task_progress = []
    for task_id, task in tasks:
        entry = progress.get(task_id, {})
        task_progress.append({
            "id": task_id,
            "title": task["title"],
            "difficulty": task["difficulty"],
            "status": entry.get("status", "todo"),
            "attempts": entry.get("attempts", 0),
            "best_time": entry.get("best_time"),
        })
    
    solved = sum(1 for t in task_progress if t["status"] == "solved")
    
    return {
        "solved": solved,
        "total": len(tasks),
        "tasks": task_progress,
    }


@app.post("/api/submit", response_model=SubmitResponse)
async def submit_code(request: SubmitRequest):
    """Submit code and run tests."""
    task = get_task(request.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    passed, total, total_time, results, output = _run_tests(request.task_id, request.code)
    
    # Update progress
    if passed == total:
        mark_solved(request.task_id, total_time)
    else:
        mark_attempted(request.task_id)
    
    return SubmitResponse(
        success=(passed == total),
        passed=passed,
        total=total,
        total_time=total_time,
        results=results,
        output=output,
    )


@app.post("/api/reset")
async def reset_progress():
    """Reset all progress."""
    from torch_judge.progress import reset_progress as _reset
    _reset()
    return {"success": True}


# Mount static files at the end
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
