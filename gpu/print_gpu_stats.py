import torch

props = torch.cuda.get_device_properties(0)
print(f"Device name: {props.name}")
print(f"Device capability: {props.major}.{props.minor}")
print(f"Device L2 cache: {props.L2_cache_size / 1024 / 1024:.2f} MB")
print(f"Device SMs: {props.multi_processor_count}")
print(f"Threads per SM: {props.max_threads_per_multi_processor}")
print(f"Shared memory per block: {props.shared_memory_per_block} bytes")
print(f"Total memory: {props.total_memory / 1e9:.2f} GB")
print(f"Warp size: {props.warp_size}")
print(f"Regs per SM: {props.regs_per_multiprocessor}")

# Derived numbers worth computing yourself
total_resident_threads = props.multi_processor_count * props.max_threads_per_multi_processor
warps_per_sm = props.max_threads_per_multi_processor // props.warp_size
print(f"\nDerived:")
print(f"  Total resident threads: {total_resident_threads:,}")
print(f"  Warps per SM: {warps_per_sm}")
print(f"  Regs per thread at full occupancy: {props.regs_per_multiprocessor // props.max_threads_per_multi_processor}")
