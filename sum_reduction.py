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
Sum all elements of a 1D vector to produce a single scalar.

"""
@triton.jit
def sum_reduction(
    result_ptr, x_ptr,
    num_elements,
    BLOCK_SIZE : tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < num_elements
    # Triton automatically handles tensor to pointer conversion when tensors are passed to the kernel
    x = tl.load(x_ptr + block_start + offsets, mask=mask)
    v = tl.sum(x)
    tl.store(result_ptr + pid, v)

BLOCK_SIZE = 128
N = 2**24

x = torch.rand(int(N), device=device, dtype=torch.float32)
result = torch.zeros(triton.cdiv(N, 32), device=device, dtype=torch.float32)  # max needed
def non_kernel_sum_reduction(x):
    return torch.sum(x)

def run(block_size):
  sum_reduction[int(triton.cdiv(N, block_size)),](result, x, N, BLOCK_SIZE=block_size)
  return torch.sum(result)


kernel_result = run(BLOCK_SIZE)
print(f"Kernel result: {kernel_result}")
non_kernel_result = non_kernel_sum_reduction(x)
print(f"Non-kernel result: {non_kernel_result}")
correct = torch.allclose(kernel_result, non_kernel_result)
print(f"N : {N} correct: {correct})")


for block_size in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
  ms = tt.do_bench(lambda : run(block_size), warmup=100, rep=10, return_mode="mean")
  bytes_per_sec = 4 * N / (1e-3 * ms)
  gb_per_sec = bytes_per_sec / 1e9
  print(f"My kernel block_size: {block_size}, t: {ms}, gb_oer_secs: {gb_per_sec}")

ms = tt.do_bench(lambda: non_kernel_sum_reduction(x), return_mode="mean")
bytes_per_sec = 4 *N / (1e-3 * ms)
gb_per_sec = bytes_per_sec / 1e9
print(f"Torch t: {ms}, gb_per_secs: {gb_per_sec}")


#print(c)

"""
Result: 

(vllm-env) prashant@H2O:~/mle/triton-kernels$ python sum_reduction.py 
Using device: cuda
GPU: NVIDIA GeForce RTX 3090 Ti
Kernel result: 8389658.0
Non-kernel result: 8389659.0
N : 16777216 correct: True)
My kernel block_size: 32, t: 0.35681999288499355, gb_oer_secs: 188.07484260454493
My kernel block_size: 64, t: 0.2011439971625805, gb_oer_secs: 333.63592722957225
My kernel block_size: 128, t: 0.1090933329736193, gb_oer_secs: 615.1509186746371
My kernel block_size: 256, t: 0.08936177718418616, gb_oer_secs: 750.9795140004878
My kernel block_size: 512, t: 0.0863253327983397, gb_oer_secs: 777.3947904350356
My kernel block_size: 1024, t: 0.0862494809208093, gb_oer_secs: 778.0784682242503
My kernel block_size: 2048, t: 0.08513422117189125, gb_oer_secs: 788.2713094244799
My kernel block_size: 4096, t: 0.08587377711578652, gb_oer_secs: 781.4826161601677
Torch t: 0.08424573858048169, gb_per_secs: 796.5846715901187

E3 sum reduction (fp32, N=2²⁴)
  triton: 798 GB/s (79% of peak; sweet spot BLOCK_SIZE=2048)
  torch:  806 GB/s (80% of peak)
  → matched torch.sum
"""





