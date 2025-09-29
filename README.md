
# Kernel Tiling size Optimization

## Methodologies

### Single-thread Workload and Resouce Requirement
Achieving a good single-thread workload is a good way to make better use of SMs on the GPU: Too many tiny threads contributes to overhead of thread management while Fewer coarse-grained threads suffer from larger latency with more sequential accesses to all the required data.

Resouces required by a single thread can bottleneck the occupancy of SMs since a warp using up the shared memory of the whole block can not parallelize with even one more warp in the same block.


## Kernel Tuning

Since most of the original tiling sizes before optimization is exactly the smallest for avoiding memory access bottlenecks, efforts here are mostly increasing tiling sizes and making heavier threads to see whether it is to make a improvement.

This work focuses on the kernels showing a low computation/memory throughput and occupancy.



