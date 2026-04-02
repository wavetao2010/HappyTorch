"""Judge microservice for HappyTorch code execution."""

from __future__ import annotations

import os
import signal
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any

import math
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="HappyTorch Judge", description="Code execution and test runner")

# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

EXECUTION_TIMEOUT = int(os.environ.get("JUDGE_TIMEOUT", "30"))


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise _TimeoutError("Execution timed out")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class TestCase(BaseModel):
    name: str
    code: str


class ExecuteRequest(BaseModel):
    code: str
    function_name: str
    tests: list[TestCase]


class TestResult(BaseModel):
    name: str
    passed: bool
    time: float
    error: str | None = None


class ExecuteResponse(BaseModel):
    success: bool
    passed: int
    total: int
    total_time: float
    results: list[TestResult]
    output: str


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_ALLOWED_MODULES = frozenset({
    "torch", "torch.nn", "torch.nn.functional", "torch.linalg",
    "torch.optim", "torch.autograd", "torch.distributions",
    "numpy", "math", "collections", "functools", "itertools", "typing",
})

_real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Allow importing only whitelisted modules."""
    top = name.split(".")[0]
    if top not in ("torch", "numpy", "math", "collections", "functools", "itertools", "typing", "time", "random", "copy"):
        raise ImportError(f"Module '{name}' is not allowed")
    return _real_import(name, *args, **kwargs)


_ALLOWED_BUILTINS = {
    k: v
    for k, v in (__builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)).items()
    if k not in ("open", "exec", "eval", "compile", "breakpoint")
}
_ALLOWED_BUILTINS["__import__"] = _safe_import


def _build_namespace() -> dict[str, Any]:
    """Build an isolated namespace for user code execution."""
    return {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "np": np,
        "numpy": np,
        "math": math,
        "__builtins__": _ALLOWED_BUILTINS,
    }


# ---------------------------------------------------------------------------
# Core execution logic
# ---------------------------------------------------------------------------


def _execute(code: str, function_name: str, tests: list[TestCase]) -> ExecuteResponse:
    """Execute user code and run tests in an isolated namespace."""
    total = len(tests)

    stdout_capture = StringIO()
    stderr_capture = StringIO()

    # Note: memory limits enforced by Docker container, not process-level setrlimit
    # (setrlimit on RLIMIT_AS breaks torch tensor allocation)

    # Set timeout
    prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(EXECUTION_TIMEOUT)

    try:
        return _execute_inner(code, function_name, tests, stdout_capture, stderr_capture)
    except _TimeoutError:
        return ExecuteResponse(
            success=False,
            passed=0,
            total=total,
            total_time=float(EXECUTION_TIMEOUT),
            results=[],
            output=f"Execution timed out after {EXECUTION_TIMEOUT} seconds",
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


def _execute_inner(
    code: str,
    function_name: str,
    tests: list[TestCase],
    stdout_capture: StringIO,
    stderr_capture: StringIO,
) -> ExecuteResponse:
    total = len(tests)
    namespace = _build_namespace()

    # Execute user code
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compile(code, "<user_code>", "exec"), namespace)
    except SyntaxError as e:
        return ExecuteResponse(
            success=False, passed=0, total=total, total_time=0.0, results=[], output=f"Syntax Error: {e}"
        )
    except Exception as e:
        return ExecuteResponse(
            success=False,
            passed=0,
            total=total,
            total_time=0.0,
            results=[],
            output=f"Code execution error: {type(e).__name__}: {e}",
        )

    # Check if function/class exists
    if function_name not in namespace:
        return ExecuteResponse(
            success=False,
            passed=0,
            total=total,
            total_time=0.0,
            results=[],
            output=f"Function/class '{function_name}' not found in your code.",
        )

    # Build test namespace
    test_namespace = {**namespace, function_name: namespace[function_name]}

    # Run each test
    results: list[TestResult] = []
    passed = 0
    total_time = 0.0

    for test in tests:
        test_code = test.code.replace("{fn}", function_name)
        t0 = time.perf_counter()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compile(test_code, f"<test:{test.name}>", "exec"), test_namespace)
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            passed += 1
            results.append(TestResult(name=test.name, passed=True, time=elapsed, error=None))
        except AssertionError as e:
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            results.append(TestResult(name=test.name, passed=False, time=elapsed, error=str(e) or "Assertion failed"))
        except Exception as e:
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            tb = traceback.format_exc()
            results.append(TestResult(name=test.name, passed=False, time=elapsed, error=f"{type(e).__name__}: {e}\n{tb}"))

    output = stdout_capture.getvalue()
    if stderr_capture.getvalue():
        output += "\n" + stderr_capture.getvalue()

    return ExecuteResponse(
        success=(passed == total),
        passed=passed,
        total=total,
        total_time=total_time,
        results=results,
        output=output.strip(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest) -> ExecuteResponse:
    return _execute(request.code, request.function_name, request.tests)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
