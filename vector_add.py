"""
E0 — Setup & ceiling. Install Triton, verify on your GPU.
Write a benchmark harness using triton.testing.do_bench that measures torch.matmul TFLOPS at
sizes 512, 1024, 2048, 4096. That's your ceiling (it calls cuBLAS).
"""

import torch
import triton
import triton.language as tl
import triton.testing as tt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def matmul(mat1, mat2):
  result = mat1 @ mat2
  return result

RTX_3090_TI_PEAK_FLOPS = {
    torch.float32: 40.0,    # CUDA cores
    torch.float16: 160.0,   # Tensor Cores, dense (320 sparse, but cuBLAS won't hit that)
}

RTX_3090_TI_PEAK_BANDWIDTH_GB_S = 1008  # theoretical; ~850 GB/s realistic

for dtype in [torch.float32, torch.float16]:
  for size in [512, 1024, 2048, 1536, 2560, 3072, 3584, 4096]:
    mat1 = torch.rand(size, size, device=device, dtype=dtype)
    mat2 = torch.rand(size, size, device=device, dtype=dtype)

    mean_time = tt.do_bench(lambda : matmul(mat1, mat2), warmup=100, return_mode="mean")
    tflops = (2 * size**3 / (1e-3 * mean_time * 10**12))
    pct_of_peak = tflops / RTX_3090_TI_PEAK_FLOPS[dtype] * 100
    print(f"dtype: {dtype} size: {size} time: {mean_time} tflops: {tflops} pct_of_peak: {pct_of_peak}")


"""
E1 — Vector add. Compute c = a + b. Learn tl.program_id, BLOCK_SIZE, offsets, tail masking. Target: ≥80% of peak memory bandwidth.
"""

@triton.jit
def add_kernel(
    result_ptr, a_ptr, b_ptr,
    num_elements,
    BLOCK_SIZE : tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < num_elements
    # Triton automatically handles tensor to pointer conversion when tensors are passed to the kernel
    a = tl.load(a_ptr + block_start + offsets, mask=mask)
    b = tl.load(b_ptr + block_start + offsets, mask=mask)
    result = a + b
    tl.store(result_ptr + block_start + offsets, result, mask=mask)

N = 2**24

a = torch.rand(N, device=device, dtype=torch.float32)
b = torch.rand(N, device=device, dtype=torch.float32)
c = torch.empty_like(a)

BLOCK_SIZE = 128

def run(block_size):
  add_kernel[triton.cdiv(N, block_size),](c, a, b, N, BLOCK_SIZE=block_size)

for block_size in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
  ms = tt.do_bench(lambda : run(block_size), warmup=100, rep=10, return_mode="mean")
  bytes_per_sec = 3 * N * 4 / (1e-3 * ms)
  gb_per_sec = bytes_per_sec / 1e9
  print(f"My kernel block_size: {block_size}, t: {ms}, gb_oer_secs: {gb_per_sec}")

ms = tt.do_bench(lambda: torch.add(a, b, out=c), return_mode="mean")
bytes_per_sec = 3 * N * 4 / (1e-3 * ms)
gb_per_sec = bytes_per_sec / 1e9
print(f"Torch t: {ms}, gb_per_secs: {gb_per_sec}")

#print(c)
"""
correct = torch.allclose(c, a + b)
print(f"N : {N} correct: {torch.allclose(c, a + b)}")
if not correct:
  print("max diff:", (c - (a + b)).abs().max().item())
  diff = (c - (a + b)).abs()
  print("num correct elements:", (diff == 0).sum().item())
  print("first wrong index:", (diff > 0).nonzero().flatten()[:5])

"""
