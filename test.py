#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_jit_probe.py

目的：
- 单独离线复现你的 runner 中 "import_test 很慢/timeout" 的场景
- 把每一步耗时打出来：import ref / import test(load_inline) / 构造模型 / forward
- 将 load_inline 的 stdout/stderr（ninja/nvcc/ld 等）落盘到日志文件，便于定位卡住点

关键特性：
- 强制每次运行重新编译（不使用 torch extension 缓存）
  1) 本脚本会把 TORCH_EXTENSIONS_DIR 指到本次 probe 目录下的 torch_ext/
  2) TEST_CODE 中的 load_inline 使用唯一 name + 显式 build_directory
  3) 编译前会 rmtree(build_directory)

使用：
  python offline_jit_probe.py

可选环境变量（你也可以不设）：
  CUDA_VISIBLE_DEVICES=0
  MAX_JOBS=16
  NINJA_NUM_JOBS=16
  CUDA_LAUNCH_BLOCKING=1   # 若怀疑 kernel illegal access 导致同步卡住
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Tuple, List

import torch


# -----------------------------
# 你这次复现用的 REF / TEST 代码
# -----------------------------
REF_CODE = r"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool(x)

batch_size = 16
channels = 32
depth = 128
height = 128
width = 256
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    x = torch.rand(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]
"""

# 强制重新编译的关键点在这里：
# - name 使用环境变量 CUDAFORGE_BUILD_SUFFIX 拼接成唯一值
# - build_directory 指到 TORCH_EXTENSIONS_DIR 下的独立目录
# - 编译前先 rmtree(build_directory)，保证不复用任何旧产物
TEST_CODE = r"""
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

suffix = os.environ.get("CUDAFORGE_BUILD_SUFFIX", "")
ext_name = f"avg_pool3d{suffix}"

base_build = os.environ.get("TORCH_EXTENSIONS_DIR", None)
if base_build is None:
    base_build = "/tmp/torch_ext_probe"
build_directory = os.path.join(base_build, ext_name)

# 强制清理，保证每次都是干净编译
shutil.rmtree(build_directory, ignore_errors=True)
os.makedirs(build_directory, exist_ok=True)

source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define KERNEL_SIZE 3
#define STRIDE 2
#define PADDING 1

__global__ void avg_pool3d_kernel(const float* input, float* output,
                                  int batch_size, int channels,
                                  int in_depth, int in_height, int in_width,
                                  int out_depth, int out_height, int out_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * channels * out_depth * out_height * out_width) {
        return;
    }

    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int d = (idx / (out_width * out_height)) % out_depth;
    int c = (idx / (out_width * out_height * out_depth)) % channels;
    int n = idx / (out_width * out_height * out_depth * channels);

    int in_d_start = d * STRIDE - PADDING;
    int in_h_start = h * STRIDE - PADDING;
    int in_w_start = w * STRIDE - PADDING;

    float sum = 0.0f;
    for (int kd = 0; kd < KERNEL_SIZE; ++kd) {
        int id = in_d_start + kd;
        if (id < 0 || id >= in_depth) continue;
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            int ih = in_h_start + kh;
            if (ih < 0 || ih >= in_height) continue;
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                int iw = in_w_start + kw;
                if (iw < 0 || iw >= in_width) continue;
                int input_idx = n * channels * in_depth * in_height * in_width +
                                c * in_depth * in_height * in_width +
                                id * in_height * in_width +
                                ih * in_width +
                                iw;
                sum += input[input_idx];
            }
        }
    }

    output[idx] = sum / (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE);
}

torch::Tensor avg_pool3d_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_depth   = input.size(2);
    int in_height  = input.size(3);
    int in_width   = input.size(4);

    int out_depth  = (in_depth  + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    int out_height = (in_height + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    int out_width  = (in_width  + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;

    auto output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, input.options());
    int total_elements = output.numel();

    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);

    avg_pool3d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width
    );

    return output;
}
'''

cpp_src = r'''
torch::Tensor avg_pool3d_cuda(torch::Tensor input);
'''

# 显式 build_directory + 唯一 name，保证不走缓存
avg_pool3d = load_inline(
    name=ext_name,
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["avg_pool3d_cuda"],
    verbose=True,
    build_directory=build_directory,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool_cuda = avg_pool3d

    def forward(self, x):
        return self.avg_pool_cuda.avg_pool3d_cuda(x)
"""


# -----------------------------
# 工具函数：动态 import + 日志捕获
# -----------------------------
def _now_ms() -> float:
    return time.time() * 1000.0


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _capture_import(path: Path, log_path: Path) -> Any:
    """
    动态导入一个 .py，同时捕获 Python print + ninja/nvcc 子进程输出，写入 log_path。
    """
    mod_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {path}")

    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = module

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

        except Exception:
            fd_buf.flush()
            fd_buf.seek(0)
            sub = fd_buf.read()
            full_log = (py_buf.getvalue() + sub + "\n" + traceback.format_exc()).strip()
            _write_text(log_path, full_log)
            raise

        finally:
            os.dup2(old1, 1)
            os.dup2(old2, 2)
            os.close(old1)
            os.close(old2)

    full_log = (py_buf.getvalue() + sub).strip()
    _write_text(log_path, full_log)
    return module


def _run_forward(model: torch.nn.Module, inp: List[torch.Tensor], dev: torch.device) -> Tuple[Any, float]:
    model.to(dev).eval()
    inp = [x.to(dev) for x in inp]
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = model(*inp)
        e.record()
        e.synchronize()
        return out, s.elapsed_time(e)
    else:
        t0 = _now_ms()
        out = model(*inp)
        return out, _now_ms() - t0


def main() -> None:
    # 固定、可复现、不会自动删除的目录
    ts = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    td = Path(f"./offline_probe_{ts}_pid{pid}")
    td.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # 强制每次重新编译：关键环境设置
    # ------------------------------
    # 每次运行都用独立 TORCH_EXTENSIONS_DIR（本次 probe 目录内）
    torch_ext_dir = td / "torch_ext"
    torch_ext_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(torch_ext_dir.resolve())

    # 唯一 suffix，确保 load_inline(name=...) 每次不同
    os.environ["CUDAFORGE_BUILD_SUFFIX"] = f"_{ts}_pid{pid}"

    print("==== offline_jit_probe ====")
    print("workdir:", td.resolve())
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("TORCH_EXTENSIONS_DIR:", os.environ.get("TORCH_EXTENSIONS_DIR"))
    print("CUDAFORGE_BUILD_SUFFIX:", os.environ.get("CUDAFORGE_BUILD_SUFFIX"))
    print("MAX_JOBS:", os.environ.get("MAX_JOBS"))
    print("NINJA_NUM_JOBS:", os.environ.get("NINJA_NUM_JOBS"))
    print("CUDA_LAUNCH_BLOCKING:", os.environ.get("CUDA_LAUNCH_BLOCKING"))

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("dev:", dev)
    if dev.type == "cuda":
        print("cuda name:", torch.cuda.get_device_name(0))

    ref_py = td / "ref.py"
    test_py = td / "test.py"
    ref_log = td / "import_ref.log"
    test_log = td / "import_test.log"

    ref_py.write_text(REF_CODE, encoding="utf-8")
    test_py.write_text(TEST_CODE, encoding="utf-8")

    # 1) import ref
    t0 = _now_ms()
    ref_mod = _capture_import(ref_py, ref_log)
    t_ref = _now_ms() - t0
    print(f"\n[1] import ref: {t_ref:.3f} ms   (log: {ref_log.resolve()})")

    # 2) import test (关键：load_inline 在这里执行 + 强制重新编译)
    t0 = _now_ms()
    test_mod = _capture_import(test_py, test_log)
    t_test = _now_ms() - t0
    print(f"[2] import test: {t_test:.3f} ms   (log: {test_log.resolve()})")

    # 3) get inputs
    get_inputs = getattr(ref_mod, "get_inputs")
    get_init_inputs = getattr(ref_mod, "get_init_inputs", None)
    init_args = list(get_init_inputs()) if callable(get_init_inputs) else []
    inp = get_inputs()
    if not isinstance(inp, (list, tuple)):
        inp = [inp]
    inp = list(inp)

    # 4) construct models
    RefModel = getattr(ref_mod, "Model")
    ModelNew = getattr(test_mod, "ModelNew")
    t0 = _now_ms()
    ref_model = RefModel(*init_args)
    test_model = ModelNew()
    t_ctor = _now_ms() - t0
    print(f"[3] construct models: {t_ctor:.3f} ms")

    # 5) forward
    t0 = _now_ms()
    out_ref, ms_ref = _run_forward(ref_model, inp, dev)
    out_tst, ms_tst = _run_forward(test_model, inp, dev)
    t_fwd = _now_ms() - t0
    print(f"[4] forward total: {t_fwd:.3f} ms   (ref {ms_ref:.3f} ms, test {ms_tst:.3f} ms)")

    # 6) basic sanity check
    if isinstance(out_ref, torch.Tensor) and isinstance(out_tst, torch.Tensor):
        diff = (out_tst - out_ref).abs()
        print("[5] diff: max=", float(diff.max().item()), "mean=", float(diff.mean().item()))
    else:
        print("[5] forward outputs are not plain tensors; skip diff")

    print("\n==== done ====")
    print("Logs preserved:")
    print(" - import_ref.log :", ref_log.resolve())
    print(" - import_test.log:", test_log.resolve())
    print("Build dir (forced fresh each run):")
    print(" - TORCH_EXTENSIONS_DIR:", Path(os.environ["TORCH_EXTENSIONS_DIR"]).resolve())
    print("If it is slow/hangs, inspect import_test.log; it should contain ninja/nvcc/ld logs.")


if __name__ == "__main__":
    main()
