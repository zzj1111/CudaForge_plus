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
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from collections import defaultdict


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
    out: Dict[str, Any] = {"ok": False, "kind": kind, "message": message}
    if log:
        out["log"] = log
    if detail:
        out["detail"] = detail
    return out


# ---------------------------
# RNG & determinism (match your reference)
# ---------------------------
def _seed_everything(seed: int | None, device_idx: int | None = None) -> None:
    """Set RNG and (optionally) deterministic backends."""
    if seed is None:
        return

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        if device_idx is not None:
            torch.cuda.set_device(device_idx)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


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


def _to_dev(x: Any, dev: torch.device) -> Any:
    return x.to(dev) if isinstance(x, torch.Tensor) else x


def _run_once(model: nn.Module, inp: List[Any], dev: torch.device) -> Tuple[Any, float]:
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


def _bench(model: nn.Module, inp: List[Any], dev: torch.device, warm: int, rep: int) -> List[float]:
    model.to(dev).eval()
    inp = [_to_dev(x, dev) for x in inp]

    for _ in range(warm):
        model(*inp)

    if dev.type == "cpu":
        import time
        res: List[float] = []
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
    # runner 不做 markdown codeblock 提取（这应由上游完成），这里只做“合规性检查”
    if "class ModelNew" not in test_code:
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
# Param alignment (lightweight but effective)
# ---------------------------
def _named_tensors(model: nn.Module) -> Dict[str, torch.Tensor]:
    named: Dict[str, torch.Tensor] = {}
    for k, p in model.named_parameters(recurse=True):
        named[f"param::{k}"] = p
    for k, b in model.named_buffers(recurse=True):
        named[f"buffer::{k}"] = b
    return named


@torch.no_grad()
def _safe_copy_(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.shape != src.shape:
        return False
    dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
    return True


@torch.no_grad()
def align_params_generic(ref_model: nn.Module, test_model: nn.Module) -> Dict[str, int]:
    """
    Minimal version:
      1) same name + same shape
      2) unique shape matching
    """
    ref_named = _named_tensors(ref_model)
    test_named = _named_tensors(test_model)

    copied_same, unique_shape_copied, skipped = 0, 0, 0
    aligned_test: set[str] = set()

    # 1) same name + same shape
    for name, t_dst in test_named.items():
        t_src = ref_named.get(name, None)
        if t_src is not None and _safe_copy_(t_dst, t_src):
            copied_same += 1
            aligned_test.add(name)

    # 2) unique shape match
    shape2ref: Dict[Tuple[int, ...], List[Tuple[str, torch.Tensor]]] = defaultdict(list)
    shape2test: Dict[Tuple[int, ...], List[Tuple[str, torch.Tensor]]] = defaultdict(list)

    for n, t in ref_named.items():
        shape2ref[tuple(t.shape)].append((n, t))
    for n, t in test_named.items():
        if n in aligned_test:
            continue
        shape2test[tuple(t.shape)].append((n, t))

    for shp, items in shape2test.items():
        if len(items) == 1 and len(shape2ref.get(shp, [])) == 1:
            tname, t_dst = items[0]
            _, t_src = shape2ref[shp][0]
            if _safe_copy_(t_dst, t_src):
                unique_shape_copied += 1
                aligned_test.add(tname)

    # count skipped
    for name in test_named.keys():
        if name not in aligned_test:
            skipped += 1

    return {
        "copied_same_shape": copied_same,
        "unique_shape_copied": unique_shape_copied,
        "skipped": skipped,
        "pair_key": "generic",
    }


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
    seed: Optional[int],
) -> Dict[str, Any]:
    msg = _validate_candidate_code(test_code)
    if msg is not None:
        return _fail(KIND_CODE_INVALID, msg, detail={"hint": "Ensure upstream extracts ```python ...``` and passes pure python code."})

    # seed policy (match your reference)
    if seed is None:
        env_seed = os.environ.get("KERNELBENCH_SEED")
        seed = int(env_seed) if env_seed is not None else None

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
            return _fail(KIND_CODE_INVALID, "Candidate does not export `ModelNew` after import.")

        init_args, init_kwargs, init_err = _get_init_args_kwargs(ref_mod)
        if init_err is not None:
            return _fail(KIND_REF_INVALID, init_err)

        # device
        dev = torch.device(f"cuda:{device_idx}") if torch.cuda.is_available() else torch.device("cpu")
        if dev.type == "cuda":
            torch.cuda.set_device(dev)

        # run under explicit device context (match your reference spirit)
        ctx = torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()
        try:
            with ctx:
                # 固定输入随机性
                _seed_everything(seed, device_idx)
                inp = get_inputs()
                if not isinstance(inp, (list, tuple)):
                    inp = [inp]
                inp = list(inp)

                # 固定参数初始化：两边构造前分别设种子
                _seed_everything(seed, device_idx)
                ref_model = RefModel(*init_args, **init_kwargs)

                _seed_everything(seed, device_idx)
                test_model = ModelNew(*init_args, **init_kwargs)

                # 参数对齐（generic 版）
                align_stats = align_params_generic(ref_model, test_model)

                # forward + correctness
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)

                ref_out, _ = _run_once(ref_model, inp, dev)
                tst_out, _ = _run_once(test_model, inp, dev)

                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)

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
                        "seed": seed,
                        "align_stats": align_stats,
                    }

                # benchmark
                ref_times = _bench(ref_model, inp, dev, warmup, repeat)
                tst_times = _bench(test_model, inp, dev, warmup, repeat)
                ref_avg = float(sum(ref_times) / max(len(ref_times), 1))
                tst_avg = float(sum(tst_times) / max(len(tst_times), 1))
                speedup = ref_avg / max(tst_avg, 1e-9)

                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)

        except Exception:
            return _fail(KIND_RUNTIME, "Runtime failure during forward/benchmark.", log=traceback.format_exc(), detail={"seed": seed})

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
            "seed": seed,
            "align_stats": align_stats,
            "max_abs_err": 0.0,
            "mean_abs_err": 0.0,
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

        # seed policy:
        # - payload has "seed": use it (can be None)
        # - else default 100 (match your reference default)
        seed = payload.get("seed", 100)
        seed = None if seed is None else int(seed)

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
        seed=seed,
    )
    _json_print(res)


if __name__ == "__main__":
    main()

