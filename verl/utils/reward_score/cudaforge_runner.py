from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch


# ---------------------------
# Error kinds (stable strings)
# ---------------------------
KIND_BAD_PAYLOAD = "bad_payload"
KIND_CODE_INVALID = "code_invalid"
KIND_REF_INVALID = "reference_invalid"
KIND_COMPILE = "compile_error"
KIND_RUNTIME = "runtime_error"
KIND_CORRECTNESS = "correctness_error"


class CompilationError(RuntimeError):
    """Raised when dynamic import / nvcc build fails. args[0] contains full build log."""


def _json_print(obj: Dict[str, Any]) -> None:
    # Ensure single-line JSON for robust parsing
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.flush()


def _fail(kind: str, message: str, *, log: Optional[str] = None, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "kind": kind,
        "message": message,
    }
    if log:
        out["log"] = log
    if detail:
        out["detail"] = detail
    return out


# ---------------------------
# Dynamic import with full log capture
# ---------------------------
def _capture_import(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    mod_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = module
    assert spec.loader is not None

    py_buf = io.StringIO()
    with tempfile.TemporaryFile(mode="w+") as fd_buf, \
         contextlib.redirect_stdout(py_buf), \
         contextlib.redirect_stderr(py_buf):

        old1, old2 = os.dup(1), os.dup(2)
        try:
            os.dup2(fd_buf.fileno(), 1)
            os.dup2(fd_buf.fileno(), 2)

            spec.loader.exec_module(module)  # type: ignore[attr-defined]

            fd_buf.flush()
            fd_buf.seek(0)
            sub = fd_buf.read()

        except Exception as exc:
            fd_buf.flush()
            fd_buf.seek(0)
            sub = fd_buf.read()
            full_log = (py_buf.getvalue() + sub + "\n" + str(exc)).strip()
            raise CompilationError(full_log) from None

        finally:
            os.dup2(old1, 1)
            os.dup2(old2, 2)
            os.close(old1)
            os.close(old2)

    return module


# ---------------------------
# Tensor helpers
# ---------------------------
def _first_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for t in x:
            if isinstance(t, torch.Tensor):
                return t
    raise TypeError("forward output is not a Tensor (or a sequence containing a Tensor).")

def _to_dev(x, dev):
    return x.to(dev) if isinstance(x, torch.Tensor) else x


def _run_once(model: torch.nn.Module, inp: List[torch.Tensor], dev: torch.device) -> Tuple[Any, float]:
    model.to(dev).eval()
    inp = [_to_dev(x, dev) for x in inp]
    if dev.type == "cpu":
        import time
        t0 = time.time()
        out = model(*inp)
        return out, (time.time() - t0) * 1000.0

    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(dev)
    s.record()
    out = model(*inp)
    e.record()
    e.synchronize()
    return out, s.elapsed_time(e)


def _bench(model: torch.nn.Module, inp: List[torch.Tensor], dev: torch.device, warm: int, rep: int) -> List[float]:
    model.to(dev).eval()
    inp = [_to_dev(x, dev) for x in inp]
    for _ in range(warm):
        model(*inp)

    if dev.type == "cpu":
        import time
        res = []
        for _ in range(rep):
            t0 = time.time()
            model(*inp)
            res.append((time.time() - t0) * 1000.0)
        return res

    torch.cuda.synchronize(dev)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    ts: List[float] = []
    for _ in range(rep):
        s.record()
        model(*inp)
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return ts


# ---------------------------
# Validation helpers
# ---------------------------
def _require_str(payload: Dict[str, Any], key: str) -> str:
    v = payload.get(key, None)
    if not isinstance(v, str) or not v.strip():
        raise KeyError(key)
    return v


def _validate_candidate_code(test_code: str) -> Optional[str]:
    """
    Returns:
      None if OK
      error message if invalid
    """
    # runner 不做 markdown codeblock 提取（这应由上游完成），这里只做“合规性检查”
    if "class ModelNew" not in test_code:
        # 不直接改写（避免隐藏问题），明确提示上游
        return "Candidate code must define `class ModelNew(...)`."
    return None


def _validate_reference_module(ref_mod) -> Optional[str]:
    RefModel = getattr(ref_mod, "Model", None)
    get_inputs = getattr(ref_mod, "get_inputs", None)
    if RefModel is None or not callable(get_inputs):
        return "Reference must define `Model` and callable `get_inputs()`."
    return None


def _get_init_args_kwargs(ref_mod) -> Tuple[List[Any], Dict[str, Any], Optional[str]]:
    init_args: List[Any] = []
    init_kwargs: Dict[str, Any] = {}
    get_init_inputs = getattr(ref_mod, "get_init_inputs", None)
    if callable(get_init_inputs):
        init_obj = get_init_inputs()
        if isinstance(init_obj, dict):
            init_kwargs = dict(init_obj)
        elif isinstance(init_obj, (list, tuple)):
            init_args = list(init_obj)
        elif init_obj is not None:
            return [], {}, "get_init_inputs() must return dict or list/tuple (or None)."
    return init_args, init_kwargs, None


# ---------------------------
# Core bench
# ---------------------------
def compare_and_bench_inline(
    *,
    ref_code: str,
    test_code: str,
    device_idx: int,
    warmup: int,
    repeat: int,
    tol: float,
) -> Dict[str, Any]:
    # candidate validation (code extraction issues should already be handled upstream,
    # but we still provide a clear error kind here)
    msg = _validate_candidate_code(test_code)
    if msg is not None:
        return _fail(KIND_CODE_INVALID, msg, detail={"hint": "Ensure upstream extracts ```python ...``` and passes pure python code."})

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ref_py = td / "ref.py"
        tst_py = td / "test.py"
        ref_py.write_text(ref_code, encoding="utf-8")
        tst_py.write_text(test_code, encoding="utf-8")

        # import/compile stage
        try:
            ref_mod = _capture_import(ref_py)
        except CompilationError as e:
            return _fail(KIND_COMPILE, "Failed to import/compile reference code.", log=e.args[0] if e.args else str(e))
        except Exception:
            return _fail(KIND_COMPILE, "Failed to import reference code (unexpected).", log=traceback.format_exc())

        try:
            tst_mod = _capture_import(tst_py)
        except CompilationError as e:
            return _fail(KIND_COMPILE, "Failed to import/compile candidate code.", log=e.args[0] if e.args else str(e))
        except Exception:
            return _fail(KIND_COMPILE, "Failed to import candidate code (unexpected).", log=traceback.format_exc())

        # reference validation
        msg = _validate_reference_module(ref_mod)
        if msg is not None:
            return _fail(KIND_REF_INVALID, msg)

        RefModel = getattr(ref_mod, "Model")
        get_inputs = getattr(ref_mod, "get_inputs")
        ModelNew = getattr(tst_mod, "ModelNew", None)
        if ModelNew is None:
            # even though we validated string contains ModelNew, it may be conditional / not executed
            return _fail(KIND_CODE_INVALID, "Candidate does not export `ModelNew` after import.")

        init_args, init_kwargs, init_err = _get_init_args_kwargs(ref_mod)
        if init_err is not None:
            return _fail(KIND_REF_INVALID, init_err)

        # device
        dev = torch.device(f"cuda:{device_idx}") if torch.cuda.is_available() else torch.device("cpu")
        if dev.type == "cuda":
            torch.cuda.set_device(dev)

        # inputs
        try:
            inp = get_inputs()
            if not isinstance(inp, (list, tuple)):
                inp = [inp]
            inp = list(inp)
        except Exception:
            return _fail(KIND_REF_INVALID, "Reference get_inputs() failed.", log=traceback.format_exc())

        # instantiate models
        try:
            ref_model = RefModel(*init_args, **init_kwargs)
            test_model = ModelNew(*init_args, **init_kwargs)
        except Exception:
            return _fail(KIND_RUNTIME, "Model construction failed.", log=traceback.format_exc())

        # forward + correctness
        try:
            ref_out, _ = _run_once(ref_model, inp, dev)
            tst_out, _ = _run_once(test_model, inp, dev)

            ref_t = _first_tensor(ref_out).contiguous()
            tst_t = _first_tensor(tst_out).contiguous()
            if ref_t.dtype != tst_t.dtype:
                tst_t = tst_t.to(ref_t.dtype)

            diff = (tst_t - ref_t).abs()
            max_err = float(diff.max().item()) if diff.numel() > 0 else 0.0
            mean_err = float(diff.mean().item()) if diff.numel() > 0 else 0.0

            ok = torch.allclose(ref_t, tst_t, atol=tol, rtol=tol)
            if not ok:
                return {
                    "ok": True,
                    "correct": False,
                    "kind": KIND_CORRECTNESS,
                    "message": f"Outputs are not close (atol/rtol={tol}).",
                    "max_abs_err": max_err,
                    "mean_abs_err": mean_err,
                }

        except Exception:
            return _fail(KIND_RUNTIME, "Runtime failure during forward/correctness check.", log=traceback.format_exc())

        # benchmark
        try:
            ref_times = _bench(ref_model, inp, dev, warmup, repeat)
            tst_times = _bench(test_model, inp, dev, warmup, repeat)
            ref_avg = float(sum(ref_times) / max(len(ref_times), 1))
            tst_avg = float(sum(tst_times) / max(len(tst_times), 1))
            speedup = ref_avg / max(tst_avg, 1e-9)
        except Exception:
            return _fail(KIND_RUNTIME, "Runtime failure during benchmarking.", log=traceback.format_exc())

        return {
            "ok": True,
            "correct": True,
            "ref_avg_ms": ref_avg,
            "tst_avg_ms": tst_avg,
            "speedup": float(speedup),
            "tol": float(tol),
            "warmup": int(warmup),
            "repeat": int(repeat),
            "device": str(dev),
        }


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except Exception:
        _json_print(_fail(KIND_BAD_PAYLOAD, "Failed to parse JSON payload from stdin.", log=traceback.format_exc()))
        return

    # required fields
    try:
        ref_code = _require_str(payload, "ref_code")
        test_code = _require_str(payload, "test_code")
    except KeyError as e:
        _json_print(_fail(KIND_BAD_PAYLOAD, f"Missing required field: {str(e)}", detail={"required": ["ref_code", "test_code"]}))
        return

    # params
    try:
        device_idx = int(payload.get("device_idx", 0))
        warmup = int(payload.get("warmup", 5))
        repeat = int(payload.get("repeat", 20))
        tol = float(payload.get("tol", 1e-4))
    except Exception:
        _json_print(_fail(KIND_BAD_PAYLOAD, "Invalid numeric parameters in payload.", log=traceback.format_exc()))
        return

    res = compare_and_bench_inline(
        ref_code=ref_code,
        test_code=test_code,
        device_idx=device_idx,
        warmup=warmup,
        repeat=repeat,
        tol=tol,
    )
    _json_print(res)


if __name__ == "__main__":
    main()
