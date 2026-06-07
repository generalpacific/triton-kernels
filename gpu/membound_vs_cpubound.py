import torch

N = 1 << 26  # 64M elements

# Set this based on what you want to measure:
# - False: forces FP32 CUDA cores (slow path; "real" FP32)
# - True:  allows TF32 tensor cores (default in PyTorch since Ampere)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

spec_3090_tflop = 40
spec_3090_mem_bandwidth = 1008

def memory_bound():
    x = torch.randn(N, device='cuda')
    y = torch.empty_like(x)  # allocate once, outside loop

    # Warmup
    for _ in range(10):
        y.copy_(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        y.copy_(x)
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end) / 1000  # seconds
    bytes_moved = 100 * 2 * N * 4  # read + write, fp32
    bandwidth_obs = bytes_moved / elapsed / 1e9
    print(f"Bandwidth: {bandwidth_obs} GB/s")
    print(f"Bandwidth peak %: {bandwidth_obs / spec_3090_mem_bandwidth} %")

def compute_bound():
    M = 4096
    a = torch.randn(M, M, device='cuda')  # NOTE: M×M matrix, not 1D!
    b = torch.randn(M, M, device='cuda')

    # Warmup
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        c = a @ b
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end) / 1000
    flops = 100 * 2 * M**3  # matmul is 2*M^3 flops
    tflops_obs = flops / elapsed / 1e12
    print(f"Compute: {tflops_obs} TFLOPS")
    print(f"Compute peak %: {tflops_obs / spec_3090_tflop} %")

memory_bound()
compute_bound()
