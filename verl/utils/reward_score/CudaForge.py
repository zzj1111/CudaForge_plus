import subprocess
import json
import re, os, sys, time
from datetime import datetime
import traceback

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
    device_idx=0,          # 外部 API 保留，但 runner 端会被强制为 0
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pid = os.getpid()
    log_path = os.path.join(log_dir, f"bench_{ts}_pid{pid}.jsonl")

    def _log(record: dict) -> None:
        record.setdefault("ts", ts)
        record.setdefault("pid", pid)
        record.setdefault("elapsed_sec", round(time.time() - t_start, 6))
        record.setdefault("timeout_sec", timeout_sec)
        _write_jsonl(log_path, record)

    # 1) extract candidate code
    try:
        test_code = _extract_python_code(solution_str)
    except Exception as ex:
        _log({"phase": "error", "ok": False, "kind": "code_extract_error", "message": repr(ex)})
        return 0, 0.0

    if "class ModelNew" not in test_code and "class Model(" in test_code:
        test_code = test_code.replace("class Model(", "class ModelNew(", 1)

    # 2) build payload
    payload = {
        "ref_code": reference_str,
        "test_code": test_code,
        "warmup": warmup,
        "repeat": repeat,
        "tol": tol,
        "seed": 100,        # 默认 seed（你需要可上层传参）
        "num_inputs": 5,    # 如果你 runner 已支持多输入 diff，这里也传下去
    }

    runner = "./verl/utils/reward_score/cudaforge_runner.py"
    cmd = [sys.executable, runner]

    # 3) env isolation: reward GPU
    env = os.environ.copy()
    reward_vis = env.get("REWARD_CUDA_VISIBLE_DEVICES", None)
    if reward_vis is not None:
        env["CUDA_VISIBLE_DEVICES"] = reward_vis

    # IMPORTANT: shared extensions cache (avoid cold build every time)
    # You can also set CUDAFORGE_EXT_CACHE in env to override.
    shared_ext = env.get("CUDAFORGE_EXT_CACHE")
    if not shared_ext:
        shared_ext = f"/dev/shm/torch_ext_shared_uid{os.getuid()}"
    env["TORCH_EXTENSIONS_DIR"] = shared_ext

    # runner sees only 1 GPU => device_idx must be 0 inside runner
    payload["device_idx"] = 0

    # runner debug dir (stage-wise dumps & import logs)
    runner_debug_dir = os.path.join(log_dir, "runner_debug", f"{ts}_pid{pid}")
    payload["debug_dir"] = runner_debug_dir

    payload_for_log = {
        **payload,
        "ref_code": _safe_tail(str(payload.get("ref_code", "")), max_code_chars),
        "test_code": _safe_tail(str(payload.get("test_code", "")), max_code_chars),
    }

    _log({
        "phase": "start",
        "ok": True,
        "cmd": cmd,
        "env": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "REWARD_CUDA_VISIBLE_DEVICES": os.environ.get("REWARD_CUDA_VISIBLE_DEVICES"),
            "TORCH_EXTENSIONS_DIR": env.get("TORCH_EXTENSIONS_DIR"),
            "MAX_JOBS": env.get("MAX_JOBS"),
            "NINJA_NUM_JOBS": env.get("NINJA_NUM_JOBS"),
        },
        "payload_meta": {k: payload.get(k) for k in ("device_idx","warmup","repeat","tol","seed","num_inputs","debug_dir")},
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

        if out:
            try:
                res = json.loads(out)
            except json.JSONDecodeError:
                res = {"ok": False, "kind": "bad_json", "message": "Runner returned non-JSON stdout."}
        else:
            res = {"ok": False, "kind": "no_output", "message": "Runner returned empty stdout."}

        res.setdefault("returncode", p.returncode)
        if err:
            res.setdefault("stderr_tail", _safe_tail(err, 2000))

        print(
            f"[CudaForge bench] ok={res.get('ok')} correct={res.get('correct', None)} "
            f"kind={res.get('kind')} rc={p.returncode} msg={res.get('message','')}"
        )
        if res.get("dump_path"):
            print(f"[CudaForge bench] dump_path={res['dump_path']}")
        print(f"[CudaForge bench] runner_debug_dir={runner_debug_dir}")

        ok = bool(res.get("ok", False))
        if (not ok) or log_on_success:
            _log({
                "phase": "finish",
                "timeout": False,
                "runner_returncode": p.returncode,
                "runner_stdout_tail": _safe_tail(out, max_io_chars),
                "runner_stderr_tail": _safe_tail(err, max_io_chars),
                "runner_json": res,
                "runner_debug_dir": runner_debug_dir,
            })

    except subprocess.TimeoutExpired as e:
        partial_out = _decode_maybe_bytes(getattr(e, "stdout", None), max_io_chars)
        partial_err = _decode_maybe_bytes(getattr(e, "stderr", None), max_io_chars)

        _log({
            "phase": "timeout",
            "timeout": True,
            "kind": "runner_timeout",
            "message": "Runner timed out.",
            "cmd": cmd,
            "env": {
                "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
                "TORCH_EXTENSIONS_DIR": env.get("TORCH_EXTENSIONS_DIR"),
                "MAX_JOBS": env.get("MAX_JOBS"),
                "NINJA_NUM_JOBS": env.get("NINJA_NUM_JOBS"),
            },
            "payload_meta": {k: payload.get(k) for k in ("device_idx","warmup","repeat","tol","seed","num_inputs","debug_dir")},
            "payload_tail": payload_for_log,
            "partial_stdout_tail": partial_out,
            "partial_stderr_tail": partial_err,
            "runner_debug_dir": runner_debug_dir,
            "hint": (
                "Go check runner_debug_dir for runner_trace.jsonl and import_test_*.log; "
                "those will tell you exactly where it hung."
            ),
        })
        print("[CudaForge bench] timeout (see jsonl log):", log_path)
        print("[CudaForge bench] runner_debug_dir:", runner_debug_dir)
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
