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
E2 — Fused elementwise. Implement y = gelu(x*w + b) as one kernel. Compare to the sam expression in eager PyTorch — you should win 3–5×, all from saved memory traffic.

Framing:
* How many full reads/writes for naive impl for 1d tensor of size N ?
  * My intuition is for naive pytorch for each operation say x 
    into w you're reading two vectors, so you're reading two n and 
    you're writing t1 which is vector vector of size n, so three n. 
    Similarly for t2 we are writing reading and writing three n. 
    And for the final, we are reading and writing N, so 2N. 
    Total 8 N bytes will be read and written, 
    but if it were a kernel it would be you're reading all three vectors once, 
    so it's three n and all the calculation happens in the kernel, 
    so you are writing only n, so total will be four n. 
    Now taking the ratio of eight n to four n expected speed up is 2x.

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

N = 2**24

x = torch.rand(N, device=device, dtype=torch.float32)
w = torch.rand(N, device=device, dtype=torch.float32)
b = torch.rand(N, device=device, dtype=torch.float32)
result = torch.empty_like(x)

BLOCK_SIZE = 128

def non_kernel_gelu(x, w, b):
    v = x * w + b
    return 0.5 * v * (1.0 + torch.erf(v * 0.70710678118))

def run(block_size):
  gelu[triton.cdiv(N, block_size),](result, x, w, b, N, BLOCK_SIZE=block_size)


run(BLOCK_SIZE)
v = x * w + b
non_kernel_result = non_kernel_gelu(x, w, b)
correct = torch.allclose(result, non_kernel_result)
print(f"N : {N} correct: {correct})")

for block_size in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
  ms = tt.do_bench(lambda : run(block_size), warmup=100, rep=10, return_mode="mean")
  bytes_per_sec = 4 * N * 4 / (1e-3 * ms)
  gb_per_sec = bytes_per_sec / 1e9
  print(f"My kernel block_size: {block_size}, t: {ms}, gb_oer_secs: {gb_per_sec}")

ms = tt.do_bench(lambda: non_kernel_gelu(x, w, b), return_mode="mean")
bytes_per_sec = 8 * N * 4 / (1e-3 * ms)
gb_per_sec = bytes_per_sec / 1e9
print(f"Torch t: {ms}, gb_per_secs: {gb_per_sec}")

#print(c)

"""
Result: 
E2 fused gelu(x*w+b) (fp32, N=2²⁴)
  triton: 924 GB/s        (92% of peak; useful_bytes)
  torch:  396 GB/s actual (39% of peak; actual 8N moved)
  → triton ~4.7× faster
    = 2× (fewer bytes) × 2.35× (better saturation)

"""





