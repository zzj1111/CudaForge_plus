import subprocess
import json
import re, os, sys, time
from datetime import datetime

_CODEBLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def _extract_python_code(solution_str: str) -> str:
    m = _CODEBLOCK_RE.search(solution_str)
    return (m.group(1) if m else solution_str).strip()

def _safe_tail(s: str, n: int) -> str:
    if not s:
        return ""
    return s[-n:] if len(s) > n else s

def _write_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _decode_maybe_bytes(x, limit: int) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        s = x.decode("utf-8", errors="replace")
    else:
        s = str(x)
    return _safe_tail(s, limit)

def bench(
    solution_str,
    reference_str,
    device_idx=0,
    warmup=5,
    repeat=20,
    tol=1e-3,
    timeout_sec=600,
    *,
    log_dir: str = "./cudaforge_logs",
    log_on_success: bool = False,
    max_code_chars: int = 8000,
    max_io_chars: int = 20000,
):
    t_start = time.time()

    # 0) log filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pid = os.getpid()
    log_path = os.path.join(log_dir, f"bench_{ts}_pid{pid}.jsonl")

    def _log(record: dict) -> None:
        # guarantee essentials
        record.setdefault("ts", ts)
        record.setdefault("pid", pid)
        record.setdefault("elapsed_sec", round(time.time() - t_start, 6))
        record.setdefault("timeout_sec", timeout_sec)
        _write_jsonl(log_path, record)

    # 1) extract candidate code
    try:
        test_code = _extract_python_code(solution_str)
    except Exception as ex:
        _log({
            "ok": False,
            "kind": "code_extract_error",
            "message": f"Failed to extract python code: {repr(ex)}",
        })
        return 0, 0.0

    if "class ModelNew" not in test_code and "class Model(" in test_code:
        test_code = test_code.replace("class Model(", "class ModelNew(", 1)

    payload = {
        "ref_code": reference_str,
        "test_code": test_code,
        "warmup": warmup,
        "repeat": repeat,
        "tol": tol,
        # seed：默认 100；你也可以在上层传入
        "seed": 100,
    }

    runner = "./verl/utils/reward_score/cudaforge_runner.py"
    cmd = [sys.executable, runner]

    # 2) env isolation: reward GPU
    env = os.environ.copy()
    reward_vis = env.get("REWARD_CUDA_VISIBLE_DEVICES", None)
    if reward_vis is not None:
        env["CUDA_VISIBLE_DEVICES"] = reward_vis

    # Each bench unique extensions dir to avoid lock / build collision
    env["TORCH_EXTENSIONS_DIR"] = f"/dev/shm/torch_ext_{pid}_{ts}"

    # runner sees only 1 GPU => device_idx must be 0 inside runner
    payload["device_idx"] = 0

    # (Highly recommended for debug; can be disabled)
    # env["CUDA_LAUNCH_BLOCKING"] = "1"

    # 3) pass runner debug dir (so runner will dump stage-wise json)
    # Use per-bench folder so you can correlate easily
    runner_debug_dir = os.path.join(log_dir, "runner_debug", f"{ts}_pid{pid}")
    payload["debug_dir"] = runner_debug_dir

    # 4) payload tail for logging (avoid huge file)
    payload_for_log = {
        **payload,
        "ref_code": _safe_tail(str(payload.get("ref_code", "")), max_code_chars),
        "test_code": _safe_tail(str(payload.get("test_code", "")), max_code_chars),
    }

    # log bench start (helpful for correlating later)
    _log({
        "phase": "start",
        "ok": True,
        "cmd": cmd,
        "env": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "REWARD_CUDA_VISIBLE_DEVICES": os.environ.get("REWARD_CUDA_VISIBLE_DEVICES"),
            "TORCH_EXTENSIONS_DIR": env.get("TORCH_EXTENSIONS_DIR"),
            "CUDAFORGE_BENCH_DEBUG_DIR": env.get("CUDAFORGE_BENCH_DEBUG_DIR"),
        },
        "payload_meta": {k: payload.get(k) for k in ("device_idx", "warmup", "repeat", "tol", "seed", "debug_dir")},
        "payload_tail": payload_for_log,
    })

    res = None
    out = ""
    err = ""
    try:
        p = subprocess.run(
            cmd,
            input=(json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout_sec,
            check=False,
        )

        out = p.stdout.decode("utf-8", errors="replace").strip()
        err = p.stderr.decode("utf-8", errors="replace").strip()

        # parse stdout JSON
        if out:
            try:
                res = json.loads(out)
            except json.JSONDecodeError:
                res = {
                    "ok": False,
                    "kind": "bad_json",
                    "message": "Runner returned non-JSON output (stdout not parseable).",
                }
        else:
            res = {
                "ok": False,
                "kind": "no_output",
                "message": "Runner returned empty stdout.",
            }

        # attach low-level process info
        res.setdefault("returncode", p.returncode)
        if err:
            res.setdefault("stderr_tail", _safe_tail(err, 2000))

        # if runner wrote its own dump file, keep it (your updated runner returns dump_path)
        runner_dump_path = res.get("dump_path", None)

        # console summary
        print(
            f"[CudaForge bench] ok={res.get('ok')} correct={res.get('correct', None)} "
            f"kind={res.get('kind')} rc={p.returncode} msg={res.get('message','')}"
        )
        if runner_dump_path:
            print(f"[CudaForge bench] runner_dump_path={runner_dump_path}")

        # write log record: failure always, success optional
        ok = bool(res.get("ok", False))
        if (not ok) or log_on_success:
            _log({
                "phase": "finish",
                "timeout": False,
                "runner_returncode": p.returncode,
                "runner_stdout_tail": _safe_tail(out, max_io_chars),
                "runner_stderr_tail": _safe_tail(err, max_io_chars),
                "runner_json": res,
                "runner_dump_path": runner_dump_path,
                "runner_debug_dir": runner_debug_dir,
            })

    except subprocess.TimeoutExpired as e:
        # IMPORTANT: subprocess.run may provide partial stdout/stderr in e.stdout/e.stderr
        partial_out = _decode_maybe_bytes(getattr(e, "stdout", None), max_io_chars)
        partial_err = _decode_maybe_bytes(getattr(e, "stderr", None), max_io_chars)

        # try to extract any JSON fragment if present (best-effort)
        inferred = {"note": "no json inferred"}
        if partial_out:
            # attempt parse whole string
            try:
                inferred = json.loads(partial_out)
            except Exception:
                inferred = {"note": "partial stdout not valid json", "stdout_tail": _safe_tail(partial_out, 2000)}

        _log({
            "phase": "timeout",
            "timeout": True,
            "message": "Runner timed out.",
            "cmd": cmd,
            "env": {
                "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
                "TORCH_EXTENSIONS_DIR": env.get("TORCH_EXTENSIONS_DIR"),
            },
            "payload_meta": {k: payload.get(k) for k in ("device_idx", "warmup", "repeat", "tol", "seed", "debug_dir")},
            "payload_tail": payload_for_log,
            "partial_stdout_tail": partial_out,
            "partial_stderr_tail": partial_err,
            "inferred_from_partial_stdout": inferred,
            "runner_debug_dir": runner_debug_dir,
            "hint": (
                "If import/compile is slow or hangs, check TORCH_EXTENSIONS_DIR build logs; "
                "if CUDA kernel illegal access, consider setting CUDA_LAUNCH_BLOCKING=1 and rerun."
            ),
        })
        print("[CudaForge bench] timeout (see jsonl log for details):", log_path)
        return 0, 0.0

    except FileNotFoundError:
        _log({
            "phase": "error",
            "ok": False,
            "kind": "runner_not_found",
            "message": f"Runner not found: {runner}",
            "cmd": cmd,
            "payload_tail": payload_for_log,
            "runner_debug_dir": runner_debug_dir,
        })
        print("[CudaForge bench] runner_not_found:", runner, "log:", log_path)
        return 0, 0.0

    except Exception as ex:
        _log({
            "phase": "error",
            "ok": False,
            "kind": "bench_exception",
            "message": f"bench() exception: {repr(ex)}",
            "traceback": traceback.format_exc(),
            "cmd": cmd,
            "payload_tail": payload_for_log,
            "runner_debug_dir": runner_debug_dir,
        })
        print("[CudaForge bench] bench_exception (see jsonl log):", log_path)
        return 0, 0.0

    # 5) adapt runner result
    if not res or not res.get("ok", False):
        return 0, 0.0
    if not res.get("correct", False):
        return 0, 0.0

    speedup = float(res.get("speedup", 0.0))
    return 1, speedup


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if extra_info is None or "answer" not in extra_info:
        return 0.0
    correctness, speedup = bench(solution_str, extra_info["answer"])
    print(f"correctness: {correctness}, speedup: {speedup}")
    score = float(correctness) * (float(speedup) + 0.3)
    return min(score, 5)
