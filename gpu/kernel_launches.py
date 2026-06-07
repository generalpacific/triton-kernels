import torch

x = torch.randn(1024, device='cuda')

# Warmup
for _ in range(100):
    x = x + 1
torch.cuda.synchronize()

# Many tiny launches
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(10000):
    x = x + 1
end.record()
torch.cuda.synchronize()
many = start.elapsed_time(end) / 1000  # seconds

# One fused launch
torch.cuda.synchronize()
start.record()
x = x + 10000  # equivalent total work in one kernel
end.record()
torch.cuda.synchronize()
one = start.elapsed_time(end) / 1000

print(f"10,000 launches: {many*1000:.1f}ms ({many/10000*1e6:.1f}μs per launch)")
print(f"1 launch:        {one*1e6:.1f}μs")
print(f"Speedup ratio:   {many/one:.0f}x")
