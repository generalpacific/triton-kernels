import torch
import triton
import triton.language as tl
import triton.testing as tt
import triton.language.extra.cuda.libdevice as nv_fast_math

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


"""
E2 — Fused elementwise. Implement y = gelu(x*w + b) as one kernel. Compare to the same expression in eager PyTorch — you should win 3–5×, all from saved memory traffic.
"""
@triton.jit
def gelu(
    result_ptr, x_ptr, w_ptr, b_ptr,
    num_elements,
    BLOCK_SIZE : tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < num_elements
    # Triton automatically handles tensor to pointer conversion when tensors are passed to the kernel
    x = tl.load(x_ptr + block_start + offsets, mask=mask)
    w = tl.load(w_ptr + block_start + offsets, mask=mask)
    b = tl.load(b_ptr + block_start + offsets, mask=mask)
    v = x * w + b
    result = 0.5 * v * (1.0 + nv_fast_math.erf(v * 0.70710678118))
    tl.store(result_ptr + block_start + offsets, result, mask=mask)

N = 2**12
M = 2**12

x = torch.rand(N, M, device=device, dtype=torch.float32)
w = torch.rand(M, device=device, dtype=torch.float32)
b = torch.rand(N, device=device, dtype=torch.float32)
result = torch.empty_like(x)

BLOCK_SIZE = 128

def run(block_size):
  gelu[triton.cdiv(N, block_size),](result, x, w, b, N, BLOCK_SIZE=block_size)
"""
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
run(BLOCK_SIZE)
v = x * w + b
non_kernel_result = 0.5 * v * (1.0 + torch.erf(v * 0.70710678118))
correct = torch.allclose(result, non_kernel_result)
print(f"N : {N} correct: {correct})")



