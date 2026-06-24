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
For an (M, K) matrix, compute row-wise softmax
"""
@triton.jit
def row_wise_softmax(
    result_ptr, x_ptr,
    x_row_stride, y_row_stride,
    num_cols,
    BLOCK_SIZE : tl.constexpr
):
    pid = tl.program_id(axis=0)
    #tl.device_print("pid", pid)
    # 4 bytes per float32
    x_row_ptr = x_ptr + pid * x_row_stride 
    y_row_ptr = result_ptr + pid * y_row_stride 
    mask = tl.arange(0, BLOCK_SIZE) < num_cols
    # load the row into registers
    x_row = tl.load(x_row_ptr + tl.arange(0, BLOCK_SIZE), mask=mask)
    # calculate softmax
    max_val = tl.max(x_row, axis=0)
    x_row = x_row - max_val
    exp_x_row = nv_fast_math.exp(x_row)
    sum_exp_x_row = tl.sum(exp_x_row, axis=0)
    softmax_row = exp_x_row / sum_exp_x_row
    # store the result
    tl.store(y_row_ptr + tl.arange(0, BLOCK_SIZE), softmax_row, mask=mask)

M, K = 4096, 4096
BLOCK_SIZE = triton.next_power_of_2(K) 


x = torch.rand((M, K), device=device, dtype=torch.float32)
result = torch.empty_like(x, device=device, dtype=torch.float32)

# Allocate something L2-sized to flush it
flush = torch.empty(int(50e6), dtype=torch.float32, device='cuda')

def non_kernel_row_wise_softmax(x):
    flush.fill_(0)
    return torch.softmax(x, dim=1)


def run(block_size):
    flush.fill_(0)
    grid = (M,)
    row_wise_softmax[grid](result, x, x_row_stride = K, 
                           y_row_stride = K, num_cols = K, BLOCK_SIZE=block_size)
    return result



kernel_result = run(BLOCK_SIZE)
print(f"Kernel result: {kernel_result}")
non_kernel_result = non_kernel_row_wise_softmax(x)
print(f"Non-kernel result: {non_kernel_result}")
correct = torch.allclose(kernel_result, non_kernel_result, atol=1e-5)
print(f"M : {M}, K : {K} correct: {correct}")


for block_size in [BLOCK_SIZE]:
  ms = tt.do_bench(lambda : run(block_size), warmup=100, rep=10, return_mode="mean")
  bytes_per_sec = 4 * M * K * 4 / (1e-3 * ms)
  gb_per_sec = bytes_per_sec / 1e9
  print(f"My kernel block_size: {block_size}, t: {ms}, gb_per_secs: {gb_per_sec}")

ms = tt.do_bench(lambda: non_kernel_row_wise_softmax(x), return_mode="mean")
bytes_per_sec = 4 * M * K * 4 / (1e-3 * ms)
gb_per_sec = bytes_per_sec / 1e9
print(f"Torch t: {ms}, gb_per_secs: {gb_per_sec}")


"""
Result: 


"""





