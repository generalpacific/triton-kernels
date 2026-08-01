import torch
import triton
import triton.language as tl
import triton.testing as tt
import triton.language.extra.cuda.libdevice as nv_fast_math

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


props = torch.cuda.get_device_properties(0)
print(f"Name: {props.name}")
print(f"Memory bus: {props.memory_bus_width} bits")
print(f"Memory clock: {props.memory_clock_rate / 1e6} MHz")
peak_gb_s = props.memory_bus_width * props.memory_clock_rate * 2 / 8 / 1e6
print(f"Peak HBM bandwidth: {peak_gb_s:.0f} GB/s")

flush = torch.empty(int(50e6), dtype=torch.float32, device='cuda')

@triton.jit
def copy_kernel(y_ptr, x_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

N = 2**28
x_flat = torch.randn(N, device='cuda')
y_flat = torch.empty_like(x_flat)

def copy():
    flush.fill_(0)
    copy_kernel[(triton.cdiv(N, 1024),)](y_flat, x_flat, N, BLOCK_SIZE=1024)

ms = tt.do_bench(copy, return_mode="mean")
practical_peak = 2 * N * 4 / (1e-3 * ms) / 1e9
print(f"Practical peak bandwidth: {practical_peak:.0f} GB/s")

"""
Transpose an (M, N) matrix to an (N, M) matrix. Compare to x.T.contiguous().
"""
@triton.jit
def transpose_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M : tl.constexpr,
    BLOCK_N : tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # offsets within this program tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn

    x_mask = offs_m[:, None] < M
    x_mask = x_mask & (offs_n[None, :] < N)

    tile = tl.load(x_ptrs, mask=x_mask)

    y_ptrs = y_ptr + offs_n[:, None] * stride_ym + offs_m[None, :] * stride_yn
    y_mask = offs_n[:, None] < N
    y_mask = y_mask & (offs_m[None, :] < M)
    tl.store(y_ptrs, tl.trans(tile), mask=y_mask)
    

M, N = 4096, 4096

x = torch.rand((M, N), device=device, dtype=torch.float32)
y = torch.empty((N, M), device=device, dtype=torch.float32) # output

block_size_m = 128
block_size_n = 128


def non_kernel_transpose(x):
    flush.fill_(0)
    return x.T.contiguous()

def non_kernel_transpose1(x):
    flush.fill_(0)
    return x.transpose(0, 1).contiguous()

def non_kernel_transpose2(x):
    flush.fill_(0)
    return torch.empty((x.shape[1], x.shape[0]), device=x.device, dtype=x.dtype).copy_(x.transpose(0, 1))

def non_kernel_transpose3(x):
    flush.fill_(0)
    return x.T.clone(memory_format=torch.contiguous_format)


def run(block_size_m, block_size_n):
    flush.fill_(0)
    grid = (triton.cdiv(M, block_size_m), triton.cdiv(N, block_size_n))
    transpose_kernel[grid](x, y, M, N, N, 1, M, 1, BLOCK_M=block_size_m, BLOCK_N=block_size_n)
    return y

print(f"x: shape={x.shape}, strides={x.stride()}")
print(f"y: shape={y.shape}, strides={y.stride()}")
print(f"original x: {x}")
kernel_result = run(128, 128)
print(f"Kernel result: {kernel_result}")
non_kernel_result = non_kernel_transpose(x)
print(f"Non-kernel result: {non_kernel_result}")
correct = torch.allclose(kernel_result, non_kernel_result, atol=1e-5)
print(f"Correct: {correct}")


for block_size_m in [16, 32, 64, 128, 256]:
  for block_size_n in [16, 32, 64, 128, 256]:
    print(f"Running with block_size: {block_size_m} x {block_size_n}")
    ms = tt.do_bench(lambda : run(block_size_m, block_size_n), warmup=100, rep=10, return_mode="mean")
    bytes_per_sec = 2 * M * N * 4 / (1e-3 * ms)
    gb_per_sec = bytes_per_sec / 1e9
    print(f"My kernel block_size: {block_size_m} x {block_size_n}, t: {ms}, gb_per_secs: {gb_per_sec}, percent of peak: {gb_per_sec / practical_peak * 100:.2f}%")

ms = tt.do_bench(lambda : non_kernel_transpose(x), warmup=100, rep=10, return_mode="mean")
bytes_per_sec = 2 * M * N * 4 / (1e-3 * ms)
gb_per_sec = bytes_per_sec / 1e9
print(f"Torch t: {ms}, gb_per_secs: {gb_per_sec}, percent of peak: {gb_per_sec / practical_peak * 100:.2f}%")

ms = tt.do_bench(lambda : non_kernel_transpose1(x), warmup=100, rep=10, return_mode="mean")
bytes_per_sec = 2 * M * N * 4 / (1e-3 * ms)
gb_per_sec = bytes_per_sec / 1e9
print(f"Torch t: {ms}, gb_per_secs: {gb_per_sec}, percent of peak: {gb_per_sec / practical_peak * 100:.2f}%")

ms = tt.do_bench(lambda : non_kernel_transpose2(x), warmup=100, rep=10, return_mode="mean")
bytes_per_sec = 2 * M * N * 4 / (1e-3 * ms)
gb_per_sec = bytes_per_sec / 1e9
print(f"Torch t: {ms}, gb_per_secs: {gb_per_sec}, percent of peak: {gb_per_sec / practical_peak * 100:.2f}%")

ms = tt.do_bench(lambda : non_kernel_transpose3(x), warmup=100, rep=10, return_mode="mean")
bytes_per_sec = 2 * M * N * 4 / (1e-3 * ms)
gb_per_sec = bytes_per_sec / 1e9
print(f"Torch t: {ms}, gb_per_secs: {gb_per_sec}, percent of peak: {gb_per_sec / practical_peak * 100:.2f}%")

"""

Result: 
E5 naive transpose (fp32, M=N=4096, L2-flushed)
  practical peak (copy):  811 GB/s
  triton:                 369 GB/s (45% of practical)
  torch (all APIs):       251 GB/s (31% of practical)
  → triton ~1.47× faster (torch's copy_ isn't optimized for pure transpose)
  → both bottlenecked by write-side coalescing;

"""





