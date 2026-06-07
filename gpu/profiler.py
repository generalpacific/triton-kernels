import torch
from torch.profiler import profile, ProfilerActivity, record_function

M = 2048
x = torch.randn(M, M, device='cuda')
y = torch.randn(M, M, device='cuda')

# CRITICAL: warmup before profiling, or first-iteration costs pollute the trace
for _ in range(50):
    z = x @ y
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    with record_function("matmul"):
        for _ in range(50):
            z = x @ y

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
print("\nOpen chrome://tracing in Chrome, then Load → trace.json")
