[ ] TODO-1: Understand BLOCK_SIZE plateau behavior in detail.
    Specifically: why does the sweet spot in E2 span 128–512 with all
    within ~1% of each other? What's the mechanism that makes a range
    of block sizes hit bandwidth equivalently? And what causes the
    decline at 2048+ (occupancy? register pressure? something else)?
    Pin this down with profiling later — Nsight Compute would show
    SM occupancy and warp stall reasons clearly.
