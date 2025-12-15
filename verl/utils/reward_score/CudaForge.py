import subprocess
import json
import re,os
import sys
_CODEBLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def _extract_python_code(solution_str: str) -> str:
    m = _CODEBLOCK_RE.search(solution_str)
    return (m.group(1) if m else solution_str).strip()

def bench(solution_str, reference_str, device_idx=0, warmup=5, repeat=20, tol=1e-4, timeout_sec=120):
    # 1) 提取候选代码（仍由主进程负责）
    test_code = _extract_python_code(solution_str)

    # 可选：你要是仍想做“容错改名”，保留；否则建议删掉以便更明确暴露错误
    if "class ModelNew" not in test_code and "class Model(" in test_code:
        test_code = test_code.replace("class Model(", "class ModelNew(", 1)

    payload = {
        "ref_code": reference_str,
        "test_code": test_code,
        "warmup": warmup,
        "repeat": repeat,
        "tol": tol,
        # 注意：device_idx 不再从主进程透传（runner 会被隔离到单卡）
    }

    runner = "./verl/utils/reward_score/cudaforge_runner.py"

    # 2) 关键：隔离 CUDA —— runner 只看见 REWARD_CUDA_VISIBLE_DEVICES（例如 7）
    env = os.environ.copy()
    reward_vis = env.get("REWARD_CUDA_VISIBLE_DEVICES", None)
    if reward_vis is not None:
        env["CUDA_VISIBLE_DEVICES"] = reward_vis

    # 3) 因为 runner 只看见一张卡，所以 device_idx 必须是 0
    payload["device_idx"] = 0

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

        if out:
            try:
                res = json.loads(out)
            except json.JSONDecodeError:
                res = {
                    "ok": False,
                    "kind": "bad_json",
                    "message": "Runner returned non-JSON output.",
                    "stdout_tail": out[-2000:],
                    "stderr_tail": err[-2000:],
                    "returncode": p.returncode,
                }
        else:
            res = {
                "ok": False,
                "kind": "no_output",
                "message": "Runner returned empty stdout.",
                "stderr_tail": err[-5000:],
                "returncode": p.returncode,
            }

        # 把 stderr/returncode 附到 res（便于你落盘统一分析）
        res.setdefault("returncode", p.returncode)
        if err:
            res.setdefault("stderr_tail", err[-2000:])

        print(f"[CudaForge bench] kind={res.get('kind')} ok={res.get('ok')} correct={res.get('correct', None)} msg={res.get('message', '')}")

    except FileNotFoundError:
        # runner 路径错误
        return 0, 0.0
    except subprocess.TimeoutExpired:
        return 0, 0.0
    except Exception:
        return 0, 0.0

    # 4) 适配 runner：ok=False 代表各类失败；ok=True, correct=False 代表正确性失败（kind=correctness_error）
    if not res.get("ok", False):
        # 你也可以在这里按 kind 做不同惩罚策略
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
    if(correctness>=0.9):
        print(solution_str)
    # You currently want: score = correctness*(speedup+0.1)
    score = float(correctness) * (float(speedup) + 0.1)
    return min(score,5)