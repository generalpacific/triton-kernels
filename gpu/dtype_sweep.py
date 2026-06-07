import torch

def run_matmul(dtype, label, use_tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    M = 4096
    a = torch.randn(M, M, device='cuda', dtype=dtype)
    b = torch.randn(M, M, device='cuda', dtype=dtype)

    # Warmup
    for _ in range(200):
        c = a @ b
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(2000):
        c = a @ b
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end) / 1000
    flops = 2000 * 2 * M**3
    print(f"{label:20s}: {flops / elapsed / 1e12:7.1f} TFLOPS")

run_matmul(torch.float32,  "FP32 (no TF32)", use_tf32=False)
run_matmul(torch.float32,  "FP32 (TF32 on)", use_tf32=True)
run_matmul(torch.bfloat16, "BF16",           use_tf32=False)
run_matmul(torch.float16,  "FP16",           use_tf32=False)
