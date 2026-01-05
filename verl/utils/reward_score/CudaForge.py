import subprocess
import json
import re,os
import sys
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

def bench(
    solution_str,
    reference_str,
    device_idx=0,
    warmup=5,
    repeat=20,
    tol=1e-3,
    timeout_sec=120,
    *,
    log_dir: str = "./cudaforge_logs",
    log_on_success: bool = False,
    max_code_chars: int = 8000,   # 防止日志文件爆炸
    max_io_chars: int = 20000     # stdout/stderr/res log 截断
):
    t_start = time.time()

    # 1) 提取候选代码
    test_code = _extract_python_code(solution_str)
    if "class ModelNew" not in test_code and "class Model(" in test_code:
        test_code = test_code.replace("class Model(", "class ModelNew(", 1)

    payload = {
        "ref_code": reference_str,
        "test_code": test_code,
        "warmup": warmup,
        "repeat": repeat,
        "tol": tol,
    }

    runner = "./verl/utils/reward_score/cudaforge_runner.py"

    # 2) 环境隔离
    env = os.environ.copy()
    reward_vis = env.get("REWARD_CUDA_VISIBLE_DEVICES", None)
    if reward_vis is not None:
        env["CUDA_VISIBLE_DEVICES"] = reward_vis

    # 每个 bench 用独立 extensions dir（避免并发冲突）
    env["TORCH_EXTENSIONS_DIR"] = f"/tmp/torch_ext_{os.getpid()}"

    # runner 只看见一张卡，所以 device_idx 必须是 0
    payload["device_idx"] = 0

    # 3) 日志文件名：时间戳 + pid
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = os.path.join(log_dir, f"bench_{ts}_pid{os.getpid()}.jsonl")

    # 为日志准备一个可控大小的 payload（避免 ref_code/test_code 过大）
    payload_for_log = dict(payload)
    payload_for_log["ref_code"] = _safe_tail(payload_for_log["ref_code"], max_code_chars)
    payload_for_log["test_code"] = _safe_tail(payload_for_log["test_code"], max_code_chars)

    try:
        p = subprocess.run(
            [sys.executable, runner],
            input=(json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout_sec,
            check=False,
        )

        out = p.stdout.decode("utf-8", errors="replace").strip()
        err = p.stderr.decode("utf-8", errors="replace").strip()

        # 默认：优先解析 stdout 的 JSON
        if out:
            try:
                res = json.loads(out)
            except json.JSONDecodeError:
                res = {
                    "ok": False,
                    "kind": "bad_json",
                    "message": "Runner returned non-JSON output.",
                }
        else:
            res = {
                "ok": False,
                "kind": "no_output",
                "message": "Runner returned empty stdout.",
            }

        # 附带 stderr/returncode，便于归因
        res.setdefault("returncode", p.returncode)
        if err:
            res.setdefault("stderr_tail", _safe_tail(err, 2000))

        # 控制台摘要
        print(f"[CudaForge bench] kind={res.get('kind')} ok={res.get('ok')} correct={res.get('correct', None)} msg={res.get('message', '')}")

        # 4) 落盘日志（按需：失败必写；成功可选）
        ok = bool(res.get("ok", False))
        if (not ok) or log_on_success:
            record = {
                "ts": ts,
                "pid": os.getpid(),
                "elapsed_sec": round(time.time() - t_start, 6),
                "timeout_sec": timeout_sec,
                "payload_tail": payload_for_log,
                "runner_returncode": p.returncode,
                "runner_stdout_tail": _safe_tail(out, max_io_chars),
                "runner_stderr_tail": _safe_tail(err, max_io_chars),
                "runner_json": res,  # 已解析的 JSON（若 bad_json/no_output 也会记录）
            }
            _write_jsonl(log_path, record)

    except subprocess.TimeoutExpired as e:
        # 超时也要落盘
        record = {
            "ts": ts,
            "pid": os.getpid(),
            "elapsed_sec": round(time.time() - t_start, 6),
            "timeout": True,
            "timeout_sec": timeout_sec,
            "payload_tail": payload_for_log,
            "message": "Runner timed out.",
            "partial_stdout_tail": _safe_tail(getattr(e, "stdout", b"").decode("utf-8", "replace") if getattr(e, "stdout", None) else "", max_io_chars),
            "partial_stderr_tail": _safe_tail(getattr(e, "stderr", b"").decode("utf-8", "replace") if getattr(e, "stderr", None) else "", max_io_chars),
        }
        _write_jsonl(log_path, record)
        print("[CudaForge bench] timeout")
        return 0, 0.0

    except FileNotFoundError:
        record = {
            "ts": ts,
            "pid": os.getpid(),
            "elapsed_sec": round(time.time() - t_start, 6),
            "ok": False,
            "kind": "runner_not_found",
            "message": f"Runner not found: {runner}",
            "payload_tail": payload_for_log,
        }
        _write_jsonl(log_path, record)
        print("[CudaForge bench] runner_not_found")
        return 0, 0.0

    except Exception as ex:
        record = {
            "ts": ts,
            "pid": os.getpid(),
            "elapsed_sec": round(time.time() - t_start, 6),
            "ok": False,
            "kind": "bench_exception",
            "message": f"bench() exception: {repr(ex)}",
            "payload_tail": payload_for_log,
        }
        _write_jsonl(log_path, record)
        print("[CudaForge bench] bench_exception")
        return 0, 0.0

    # 5) 适配 runner 结果
    if not res.get("ok", False):
        return 0, 0.0
    if not res.get("correct", False):
        return 0, 0.0

    speedup = float(res.get("speedup", 0.0))
    return 1, speedup


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # solution_str: LLM generated python code in ```python ... ```
    # extra_info['answer']: pytorch reference (Model + get_inputs + optional get_init_inputs)
    if extra_info is None or "answer" not in extra_info:
        return 0.0
    correctness, speedup = bench(solution_str, extra_info["answer"])
    print(f"correctness: {correctness}, speedup: {speedup}")
    # print(solution_str,extra_info["question"])
    # print(solution_str
    # if(correctness>=0.9):
    #     print(solution_str)
    # You currently want: score = correctness*(speedup+0.1)
    score = float(correctness) * (float(speedup) + 0.3)
    return min(score,5)
