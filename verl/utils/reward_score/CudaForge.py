import subprocess
import json
import re, os, sys, time, traceback
from datetime import datetime
from pathlib import Path

_CODEBLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def _extract_python_code(solution_str: str) -> str:
    m = _CODEBLOCK_RE.search(solution_str)
    return (m.group(1) if m else solution_str).strip()

def _safe_tail(s: str, n: int) -> str:
    if not s:
        return ""
    return s[-n:] if len(s) > n else s

def _write_jsonl(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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
    # 多输入 diff
    num_inputs: int = 5,
    # 编译并行度
    ninja_jobs: int = 16,
    max_jobs: int = 16,
    # extensions 缓存策略：unique/shared
    ext_dir_mode: str = "shared",
    # ✅ 关键：只编译 H200 架构，避免“全家桶 gencode”
    torch_cuda_arch_list: str = "9.0",
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
        _log({
            "phase": "error",
            "ok": False,
            "kind": "code_extract_error",
            "message": f"Failed to extract python code: {repr(ex)}",
            "traceback": traceback.format_exc(),
        })
        return 0, 0.0

    if "class ModelNew" not in test_code and "class Model(" in test_code:
        test_code = test_code.replace("class Model(", "class ModelNew(", 1)

    # 2) payload
    payload = {
        "ref_code": reference_str,
        "test_code": test_code,
        "warmup": int(warmup),
        "repeat": int(repeat),
        "tol": float(tol),
        "seed": 100,
        "device_idx": 0,               # runner 只看见一张卡 => 必须 0
        "debug_dir": None,             # 后面赋值
        "num_inputs": int(num_inputs), # runner 用于多输入 diff
        # ✅ 传给 runner：让 runner 在 import torch 前就设置
        "torch_cuda_arch_list": str(torch_cuda_arch_list),
    }

    # runner path: 绝对路径避免 cwd 不一致
    runner = os.path.abspath("./verl/utils/reward_score/cudaforge_runner.py")
    cmd = [sys.executable, runner]

    # 3) env isolation: reward GPU + 编译参数
    env = os.environ.copy()

    reward_vis = env.get("REWARD_CUDA_VISIBLE_DEVICES", None)
    if reward_vis is not None:
        env["CUDA_VISIBLE_DEVICES"] = reward_vis

    # ✅ 强制覆盖，不用 setdefault（你之前踩坑就在这里）
    env["MAX_JOBS"] = str(max_jobs)
    env["NINJA_NUM_JOBS"] = str(ninja_jobs)

    # ✅ 同样强制覆盖 TORCH_CUDA_ARCH_LIST（主进程层面也设置一遍，双保险）
    # 注意：最终生效以 runner 在 import torch 前的设置为准（runner 会再覆盖一次）
    env["TORCH_CUDA_ARCH_LIST"] = str(torch_cuda_arch_list)

    # extensions dir 策略（这里“不需要强制重新编译”，所以默认 shared 更合理）
    if ext_dir_mode == "shared":
        vis = env.get("CUDA_VISIBLE_DEVICES", "unknown")
        env["TORCH_EXTENSIONS_DIR"] = f"/tmp/torch_ext_cache_reward_cuda{vis}"
    else:
        env["TORCH_EXTENSIONS_DIR"] = f"/dev/shm/torch_ext_{pid}_{ts}"

    # 4) runner debug dir
    runner_debug_dir = os.path.join(log_dir, "runner_debug", f"{ts}_pid{pid}")
    payload["debug_dir"] = runner_debug_dir

    # 5) payload tail
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
            "TORCH_CUDA_ARCH_LIST": env.get("TORCH_CUDA_ARCH_LIST"),
        },
        "payload_meta": {k: payload.get(k) for k in (
            "device_idx", "warmup", "repeat", "tol", "seed", "num_inputs", "debug_dir", "torch_cuda_arch_list"
        )},
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
                res = {"ok": False, "kind": "bad_json", "message": "Runner stdout is not JSON."}
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
            print(f"[CudaForge bench] runner_dump_path={res.get('dump_path')}")

        ok = bool(res.get("ok", False))
        if (not ok) or log_on_success:
            _log({
                "phase": "finish",
                "timeout": False,
                "runner_returncode": p.returncode,
                "runner_stdout_tail": _safe_tail(out, max_io_chars),
                "runner_stderr_tail": _safe_tail(err, max_io_chars),
                "runner_json": res,
                "runner_dump_path": res.get("dump_path"),
                "runner_debug_dir": runner_debug_dir,
            })

    except subprocess.TimeoutExpired as e:
        partial_out = _decode_maybe_bytes(getattr(e, "stdout", None), max_io_chars)
        partial_err = _decode_maybe_bytes(getattr(e, "stderr", None), max_io_chars)

        inferred = {"note": "no json inferred"}
        if partial_out:
            try:
                inferred = json.loads(partial_out)
            except Exception:
                inferred = {"note": "partial stdout not json", "stdout_tail": _safe_tail(partial_out, 2000)}

        _log({
            "phase": "timeout",
            "timeout": True,
            "message": "Runner timed out.",
            "cmd": cmd,
            "env": {
                "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
                "TORCH_EXTENSIONS_DIR": env.get("TORCH_EXTENSIONS_DIR"),
                "MAX_JOBS": env.get("MAX_JOBS"),
                "NINJA_NUM_JOBS": env.get("NINJA_NUM_JOBS"),
                "TORCH_CUDA_ARCH_LIST": env.get("TORCH_CUDA_ARCH_LIST"),
            },
            "payload_meta": {k: payload.get(k) for k in (
                "device_idx", "warmup", "repeat", "tol", "seed", "num_inputs", "debug_dir", "torch_cuda_arch_list"
            )},
            "payload_tail": payload_for_log,
            "partial_stdout_tail": partial_out,
            "partial_stderr_tail": partial_err,
            "inferred_from_partial_stdout": inferred,
            "runner_debug_dir": runner_debug_dir,
            "hint": "Most timeouts are slow/hanging torch extension compile during import_test. Check runner_debug dump timings_ms.import_test.",
        })
        print("[CudaForge bench] timeout (see jsonl log):", log_path)
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
    return min(score, 3)
