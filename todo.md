[ ] TODO-1: Understand BLOCK_SIZE plateau behavior in detail.
    Specifically: why does the sweet spot in E2 span 128–512 with all
    within ~1% of each other? What's the mechanism that makes a range
    of block sizes hit bandwidth equivalently? And what causes the
    decline at 2048+ (occupancy? register pressure? something else)?
    Pin this down with profiling later — Nsight Compute would show
    SM occupancy and warp stall reasons clearly.

TODO-2: GPU benchmark hygiene
  - Always sanity-check measured GB/s vs hardware peak.
    If result > peak, the measurement is wrong, not the kernel.
  - do_bench by default reuses the same input buffers, so reads
    from large reusable inputs get L2-cached after the first
    iteration → inflated GB/s numbers.
  - Fix: flush L2 between iterations, OR use input sizes
    much larger than L2 (3090 Ti L2 = 6 MB).
