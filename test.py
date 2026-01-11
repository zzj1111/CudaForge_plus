#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_jit_probe.py

目标：
- 离线复现 runner 中 import_test(load_inline) 很慢/timeout
- 打印每一步耗时，保存 import_ref/import_test 的完整编译日志
- 每次运行强制重新编译（不使用缓存）
- 关键加速：强制 TORCH_CUDA_ARCH_LIST=9.0（H200=sm_90），避免编译一堆无关架构
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
    if (idx >= batch_size * channels * out_depth * out_height * out_width) return;

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


def _now_ms() -> float:
    return time.time() * 1000.0


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _capture_import(path: Path, log_path: Path) -> Any:
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
    ts = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    td = Path(f"./offline_probe_{ts}_pid{pid}")
    td.mkdir(parents=True, exist_ok=True)

    # 1) 强制每次新编译：独立 TORCH_EXTENSIONS_DIR
    torch_ext_dir = td / "torch_ext"
    torch_ext_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(torch_ext_dir.resolve())

    # 2) 强制每次新编译：唯一扩展名 suffix
    os.environ["CUDAFORGE_BUILD_SUFFIX"] = f"_{ts}_pid{pid}"

    # 3) 关键加速：只编译 H200 所需架构（sm_90）
    #    你也可以改成 "9.0+PTX"（会多生成 PTX，兼容性更强但稍慢一点）
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

    # 4) 可选：固定 ninja 并行度（你慢服务器日志里显示 Allowing ninja default…）
    #    如果你希望更可控，可取消注释：
    # os.environ.setdefault("MAX_JOBS", "16")
    # os.environ.setdefault("NINJA_NUM_JOBS", "16")

    print("==== offline_jit_probe ====")
    print("workdir:", td.resolve())
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("TORCH_EXTENSIONS_DIR:", os.environ.get("TORCH_EXTENSIONS_DIR"))
    print("CUDAFORGE_BUILD_SUFFIX:", os.environ.get("CUDAFORGE_BUILD_SUFFIX"))
    print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))
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

    t0 = _now_ms()
    ref_mod = _capture_import(ref_py, ref_log)
    print(f"\n[1] import ref: {_now_ms() - t0:.3f} ms   (log: {ref_log.resolve()})")

    t0 = _now_ms()
    test_mod = _capture_import(test_py, test_log)
    print(f"[2] import test: {_now_ms() - t0:.3f} ms   (log: {test_log.resolve()})")

    get_inputs = getattr(ref_mod, "get_inputs")
    get_init_inputs = getattr(ref_mod, "get_init_inputs", None)
    init_args = list(get_init_inputs()) if callable(get_init_inputs) else []
    inp = get_inputs()
    if not isinstance(inp, (list, tuple)):
        inp = [inp]
    inp = list(inp)

    RefModel = getattr(ref_mod, "Model")
    ModelNew = getattr(test_mod, "ModelNew")
    t0 = _now_ms()
    ref_model = RefModel(*init_args)
    test_model = ModelNew()
    print(f"[3] construct models: {_now_ms() - t0:.3f} ms")

    t0 = _now_ms()
    out_ref, ms_ref = _run_forward(ref_model, inp, dev)
    out_tst, ms_tst = _run_forward(test_model, inp, dev)
    print(f"[4] forward total: {_now_ms() - t0:.3f} ms   (ref {ms_ref:.3f} ms, test {ms_tst:.3f} ms)")

    if isinstance(out_ref, torch.Tensor) and isinstance(out_tst, torch.Tensor):
        diff = (out_tst - out_ref).abs()
        print("[5] diff: max=", float(diff.max().item()), "mean=", float(diff.mean().item()))
    else:
        print("[5] forward outputs are not plain tensors; skip diff")

    print("\n==== done ====")
    print("Logs preserved:")
    print(" - import_ref.log :", ref_log.resolve())
    print(" - import_test.log:", test_log.resolve())
    print("If import_test is still slow, open import_test.log and check where it pauses (c++/nvcc/ld).")


if __name__ == "__main__":
    main()
