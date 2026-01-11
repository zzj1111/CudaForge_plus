from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

# ---------------------------
# IMPORTANT: set TORCH_CUDA_ARCH_LIST BEFORE importing torch
# ---------------------------
def _peek_arch_list_from_stdin() -> Tuple[Optional[str], Optional[dict], Optional[str]]:
    """
    Read stdin once, parse JSON, and extract torch_cuda_arch_list if present.
    Return (arch, payload_dict, raw).
    """
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except Exception:
        return None, None, raw
    arch = payload.get("torch_cuda_arch_list", None)
    if arch is not None:
        arch = str(arch).strip()
        if not arch:
            arch = None
    return arch, payload, raw

_arch, _payload0, _raw0 = _peek_arch_list_from_stdin()

# Decide final arch list
# Priority: payload -> existing env -> default
_final_arch = _arch or os.environ.get("TORCH_CUDA_ARCH_LIST") or "9.0"
os.environ["TORCH_CUDA_ARCH_LIST"] = _final_arch

# Now it is safe to import torch / cpp_extension side-effects
import torch
import torch.nn as nn


KIND_BAD_PAYLOAD = "bad_payload"
KIND_CODE_INVALID = "code_invalid"
KIND_REF_INVALID = "reference_invalid"
KIND_COMPILE = "compile_error"
KIND_RUNTIME = "runtime_error"
KIND_CORRECTNESS = "correctness_error"


class CompilationError(RuntimeError):
    """Raised when dynamic import / nvcc build fails. args[0] contains full build log."""


def _now_ms() -> float:
    return time.time() * 1000.0


def _env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "REWARD_CUDA_VISIBLE_DEVICES": os.environ.get("REWARD_CUDA_VISIBLE_DEVICES"),
        "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR"),
        "MAX_JOBS": os.environ.get("MAX_JOBS"),
        "NINJA_NUM_JOBS": os.environ.get("NINJA_NUM_JOBS"),
        "KERNELBENCH_SEED": os.environ.get("KERNELBENCH_SEED"),
        "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        # ✅ record arch list for audit
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    }
    try:
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_count"] = torch.cuda.device_count()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                info["cuda_current_device"] = torch.cuda.current_device()
            except Exception:
                info["cuda_current_device"] = None
            try:
                info["cuda_name_0"] = torch.cuda.get_device_name(0)
            except Exception:
                info["cuda_name_0"] = None
    except Exception:
        info["torch_probe_error"] = traceback.format_exc()
    return info


def _json_print(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.flush()


def _fail(kind: str, message: str, *,
          log: Optional[str] = None,
          detail: Optional[Dict[str, Any]] = None,
          timings: Optional[Dict[str, float]] = None,
          env_info: Optional[Dict[str, Any]] = None,
          dump_path: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False, "kind": kind, "message": message}
    if log:
        out["log"] = log
    if detail:
        out["detail"] = detail
    if timings:
        out["timings_ms"] = timings
    if env_info:
        out["env_info"] = env_info
    if dump_path:
        out["dump_path"] = dump_path
    return out


def _maybe_dump_debug(payload: Dict[str, Any],
                      result: Dict[str, Any],
                      *,
                      stage: str,
                      debug_dir: Optional[str],
                      td: Optional[Path] = None) -> Optional[str]:
    if not debug_dir:
        return None
    try:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        pid = os.getpid()
        h = hashlib.md5((payload.get("test_code", "") + payload.get("ref_code", "")).encode("utf-8", errors="ignore")).hexdigest()[:8]
        p = Path(debug_dir) / f"bench_{ts}_pid{pid}_{h}_{stage}.json"
        dump_obj = {
            "stage": stage,
            "payload_meta": {k: payload.get(k) for k in ("device_idx", "warmup", "repeat", "tol", "seed", "num_inputs")},
            "env_info": _env_info(),
            "result": result,
        }
        if td is not None:
            dump_obj["tmp_dir"] = str(td)
        p.write_text(json.dumps(dump_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)
    except Exception:
        sys.stderr.write("[runner] failed to dump debug file:\n" + traceback.format_exc() + "\n")
        sys.stderr.flush()
        return None


def _seed_everything(seed: Optional[int], device_idx: Optional[int] = None) -> None:
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

            fd_buf.flush(); fd_buf.seek(0)
            sub = fd_buf.read()

        except Exception as exc:
            fd_buf.flush(); fd_buf.seek(0)
            sub = fd_buf.read()
            full_log = (py_buf.getvalue() + sub + "\n" + str(exc)).strip()
            raise CompilationError(full_log) from None

        finally:
            os.dup2(old1, 1)
            os.dup2(old2, 2)
            os.close(old1)
            os.close(old2)

    return module


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
        t0 = _now_ms()
        out = model(*inp)
        return out, _now_ms() - t0

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
        res: List[float] = []
        for _ in range(rep):
            t0 = _now_ms()
            model(*inp)
            res.append(_now_ms() - t0)
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


def _require_str(payload: Dict[str, Any], key: str) -> str:
    v = payload.get(key, None)
    if not isinstance(v, str) or not v.strip():
        raise KeyError(key)
    return v


def _validate_candidate_code(test_code: str) -> Optional[str]:
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
    ref_named = _named_tensors(ref_model)
    test_named = _named_tensors(test_model)

    copied_same, unique_shape_copied, skipped = 0, 0, 0
    aligned_test: set[str] = set()

    for name, t_dst in test_named.items():
        t_src = ref_named.get(name, None)
        if t_src is not None and _safe_copy_(t_dst, t_src):
            copied_same += 1
            aligned_test.add(name)

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

    for name in test_named.keys():
        if name not in aligned_test:
            skipped += 1

    return {
        "copied_same_shape": copied_same,
        "unique_shape_copied": unique_shape_copied,
        "skipped": skipped,
        "pair_key": "generic",
    }


def compare_and_bench_inline(
    *,
    ref_code: str,
    test_code: str,
    device_idx: int,
    warmup: int,
    repeat: int,
    tol: float,
    seed: Optional[int],
    debug_dir: Optional[str],
    num_inputs: int,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    envinfo = _env_info()

    if seed is None:
        env_seed = os.environ.get("KERNELBENCH_SEED")
        seed = int(env_seed) if env_seed is not None else None

    # validate candidate code
    t0 = _now_ms()
    msg = _validate_candidate_code(test_code)
    timings["validate_code"] = _now_ms() - t0
    if msg is not None:
        res = _fail(KIND_CODE_INVALID, msg,
                    detail={"hint": "Ensure upstream extracts ```python ...``` and passes pure python code."},
                    timings=timings, env_info=envinfo)
        dump = _maybe_dump_debug({"ref_code": ref_code, "test_code": test_code, "device_idx": device_idx,
                                  "warmup": warmup, "repeat": repeat, "tol": tol, "seed": seed, "num_inputs": num_inputs},
                                 res, stage="code_invalid", debug_dir=debug_dir)
        if dump:
            res["dump_path"] = dump
        return res

    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        ref_py = td / "ref.py"
        tst_py = td / "test.py"
        ref_py.write_text(ref_code, encoding="utf-8")
        tst_py.write_text(test_code, encoding="utf-8")

        # import reference
        t0 = _now_ms()
        try:
            ref_mod = _capture_import(ref_py)
        except CompilationError as e:
            timings["import_ref"] = _now_ms() - t0
            res = _fail(KIND_COMPILE, "Failed to import/compile reference code.",
                        log=e.args[0] if e.args else str(e),
                        timings=timings, env_info=envinfo)
            dump = _maybe_dump_debug({"ref_code": ref_code, "test_code": test_code, "device_idx": device_idx,
                                      "warmup": warmup, "repeat": repeat, "tol": tol, "seed": seed, "num_inputs": num_inputs},
                                     res, stage="compile_ref", debug_dir=debug_dir, td=td)
            if dump:
                res["dump_path"] = dump
            return res
        except Exception:
            timings["import_ref"] = _now_ms() - t0
            res = _fail(KIND_COMPILE, "Failed to import reference code (unexpected).",
                        log=traceback.format_exc(),
                        timings=timings, env_info=envinfo)
            dump = _maybe_dump_debug({"ref_code": ref_code, "test_code": test_code, "device_idx": device_idx,
                                      "warmup": warmup, "repeat": repeat, "tol": tol, "seed": seed, "num_inputs": num_inputs},
                                     res, stage="compile_ref_unexpected", debug_dir=debug_dir, td=td)
            if dump:
                res["dump_path"] = dump
            return res
        timings["import_ref"] = _now_ms() - t0

        # import candidate
        t0 = _now_ms()
        try:
            tst_mod = _capture_import(tst_py)
        except CompilationError as e:
            timings["import_test"] = _now_ms() - t0
            res = _fail(KIND_COMPILE, "Failed to import/compile candidate code.",
                        log=e.args[0] if e.args else str(e),
                        timings=timings, env_info=envinfo)
            dump = _maybe_dump_debug({"ref_code": ref_code, "test_code": test_code, "device_idx": device_idx,
                                      "warmup": warmup, "repeat": repeat, "tol": tol, "seed": seed, "num_inputs": num_inputs},
                                     res, stage="compile_test", debug_dir=debug_dir, td=td)
            if dump:
                res["dump_path"] = dump
            return res
        except Exception:
            timings["import_test"] = _now_ms() - t0
            res = _fail(KIND_COMPILE, "Failed to import candidate code (unexpected).",
                        log=traceback.format_exc(),
                        timings=timings, env_info=envinfo)
            dump = _maybe_dump_debug({"ref_code": ref_code, "test_code": test_code, "device_idx": device_idx,
                                      "warmup": warmup, "repeat": repeat, "tol": tol, "seed": seed, "num_inputs": num_inputs},
                                     res, stage="compile_test_unexpected", debug_dir=debug_dir, td=td)
            if dump:
                res["dump_path"] = dump
            return res
        timings["import_test"] = _now_ms() - t0

        # validate reference module
        t0 = _now_ms()
        msg = _validate_reference_module(ref_mod)
        timings["validate_ref"] = _now_ms() - t0
        if msg is not None:
            res = _fail(KIND_REF_INVALID, msg, timings=timings, env_info=envinfo)
            dump = _maybe_dump_debug({"ref_code": ref_code, "test_code": test_code, "device_idx": device_idx,
                                      "warmup": warmup, "repeat": repeat, "tol": tol, "seed": seed, "num_inputs": num_inputs},
                                     res, stage="ref_invalid", debug_dir=debug_dir, td=td)
            if dump:
                res["dump_path"] = dump
            return res

        RefModel = getattr(ref_mod, "Model")
        get_inputs = getattr(ref_mod, "get_inputs")
        ModelNew = getattr(tst_mod, "ModelNew", None)
        if ModelNew is None:
            res = _fail(KIND_CODE_INVALID, "Candidate does not export `ModelNew` after import.",
                        timings=timings, env_info=envinfo)
            dump = _maybe_dump_debug({"ref_code": ref_code, "test_code": test_code, "device_idx": device_idx,
                                      "warmup": warmup, "repeat": repeat, "tol": tol, "seed": seed, "num_inputs": num_inputs},
                                     res, stage="code_invalid_post_import", debug_dir=debug_dir, td=td)
            if dump:
                res["dump_path"] = dump
            return res

        init_args, init_kwargs, init_err = _get_init_args_kwargs(ref_mod)
        if init_err is not None:
            res = _fail(KIND_REF_INVALID, init_err, timings=timings, env_info=envinfo)
            dump = _maybe_dump_debug({"ref_code": ref_code, "test_code": test_code, "device_idx": device_idx,
                                      "warmup": warmup, "repeat": repeat, "tol": tol, "seed": seed, "num_inputs": num_inputs},
                                     res, stage="ref_invalid_init", debug_dir=debug_dir, td=td)
            if dump:
                res["dump_path"] = dump
            return res

        # device setup
        t0 = _now_ms()
        dev = torch.device(f"cuda:{device_idx}") if torch.cuda.is_available() else torch.device("cpu")
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
        timings["device_setup"] = _now_ms() - t0

        ctx = torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()

        try:
            with ctx:
                input_errs: List[Dict[str, float]] = []
                inp0: Optional[List[Any]] = None

                t0 = _now_ms()
                for i in range(int(max(num_inputs, 1))):
                    if seed is not None:
                        _seed_everything(int(seed) + i, device_idx)

                    inp = get_inputs()
                    if not isinstance(inp, (list, tuple)):
                        inp = [inp]
                    inp = list(inp)
                    if inp0 is None:
                        inp0 = inp

                    if seed is not None:
                        _seed_everything(int(seed), device_idx)
                    ref_model = RefModel(*init_args, **init_kwargs)
                    if seed is not None:
                        _seed_everything(int(seed), device_idx)
                    test_model = ModelNew(*init_args, **init_kwargs)

                    align_stats = align_params_generic(ref_model, test_model)

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
                    input_errs.append({"max_abs_err": max_err, "mean_abs_err": mean_err})

                timings["forward_multi_inputs"] = _now_ms() - t0

                max_err_mean = float(sum(x["max_abs_err"] for x in input_errs) / max(len(input_errs), 1))
                mean_err_mean = float(sum(x["mean_abs_err"] for x in input_errs) / max(len(input_errs), 1))

                if max_err_mean > tol or mean_err_mean > tol:
                    res = {
                        "ok": True,
                        "correct": False,
                        "kind": KIND_CORRECTNESS,
                        "message": f"Mean errors exceed tolerance (tol={tol}).",
                        "max_abs_err_mean": max_err_mean,
                        "mean_abs_err_mean": mean_err_mean,
                        "per_input_errs": input_errs,
                        "seed": seed,
                        "align_stats": align_stats,
                        "device": str(dev),
                        "timings_ms": timings,
                        "env_info": envinfo,
                    }
                    dump = _maybe_dump_debug({"ref_code": ref_code, "test_code": test_code, "device_idx": device_idx,
                                              "warmup": warmup, "repeat": repeat, "tol": tol, "seed": seed, "num_inputs": num_inputs},
                                             res, stage="correctness_fail", debug_dir=debug_dir, td=td)
                    if dump:
                        res["dump_path"] = dump
                    return res

                assert inp0 is not None
                if seed is not None:
                    _seed_everything(int(seed), device_idx)
                ref_model = RefModel(*init_args, **init_kwargs)
                if seed is not None:
                    _seed_everything(int(seed), device_idx)
                test_model = ModelNew(*init_args, **init_kwargs)

                t0 = _now_ms()
                align_stats = align_params_generic(ref_model, test_model)
                timings["align_params"] = _now_ms() - t0

                t0 = _now_ms()
                ref_times = _bench(ref_model, inp0, dev, warmup, repeat)
                tst_times = _bench(test_model, inp0, dev, warmup, repeat)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                timings["benchmark"] = _now_ms() - t0

                ref_avg = float(sum(ref_times) / max(len(ref_times), 1))
                tst_avg = float(sum(tst_times) / max(len(tst_times), 1))
                speedup = ref_avg / max(tst_avg, 1e-9)

        except Exception:
            res = _fail(
                KIND_RUNTIME,
                "Runtime failure during forward/benchmark.",
                log=traceback.format_exc(),
                detail={"seed": seed, "device": str(dev)},
                timings=timings,
                env_info=envinfo,
            )
            dump = _maybe_dump_debug({"ref_code": ref_code, "test_code": test_code, "device_idx": device_idx,
                                      "warmup": warmup, "repeat": repeat, "tol": tol, "seed": seed, "num_inputs": num_inputs},
                                     res, stage="runtime_fail", debug_dir=debug_dir, td=td)
            if dump:
                res["dump_path"] = dump
            return res

        res = {
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
            "num_inputs": int(max(num_inputs, 1)),
            "align_stats": align_stats,
            "max_abs_err_mean": max_err_mean,
            "mean_abs_err_mean": mean_err_mean,
            "per_input_errs": input_errs,
            "timings_ms": timings,
            "env_info": envinfo,
        }
        dump = _maybe_dump_debug({"ref_code": ref_code, "test_code": test_code, "device_idx": device_idx,
                                  "warmup": warmup, "repeat": repeat, "tol": tol, "seed": seed, "num_inputs": num_inputs},
                                 res, stage="ok", debug_dir=debug_dir, td=td)
        if dump:
            res["dump_path"] = dump
        return res


def main() -> None:
    # ✅ 我们已经在文件开头读过 stdin（_raw0 / _payload0），这里复用避免再次 read 空
    if _payload0 is None:
        # parse failed earlier
        _json_print(_fail(KIND_BAD_PAYLOAD, "Failed to parse JSON payload from stdin.",
                          log=traceback.format_exc(),
                          env_info=_env_info()))
        return

    payload = _payload0

    try:
        ref_code = _require_str(payload, "ref_code")
        test_code = _require_str(payload, "test_code")
    except KeyError as e:
        _json_print(_fail(KIND_BAD_PAYLOAD, f"Missing required field: {str(e)}",
                          detail={"required": ["ref_code", "test_code"]},
                          env_info=_env_info()))
        return

    try:
        device_idx = int(payload.get("device_idx", 0))
        warmup = int(payload.get("warmup", 5))
        repeat = int(payload.get("repeat", 20))
        tol = float(payload.get("tol", 1e-4))
        seed = payload.get("seed", 100)
        seed = None if seed is None else int(seed)
        debug_dir = payload.get("debug_dir", os.environ.get("CUDAFORGE_BENCH_DEBUG_DIR"))
        debug_dir = str(debug_dir) if debug_dir else None
        num_inputs = int(payload.get("num_inputs", 5))
    except Exception:
        _json_print(_fail(KIND_BAD_PAYLOAD, "Invalid numeric parameters in payload.",
                          log=traceback.format_exc(),
                          env_info=_env_info()))
        return

    res = compare_and_bench_inline(
        ref_code=ref_code,
        test_code=test_code,
        device_idx=device_idx,
        warmup=warmup,
        repeat=repeat,
        tol=tol,
        seed=seed,
        debug_dir=debug_dir,
        num_inputs=num_inputs,
    )
    _json_print(res)


if __name__ == "__main__":
    main()
