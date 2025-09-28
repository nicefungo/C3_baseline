
# Kernel Memory Access Optimization

## Methodologies

### Coalesced Global Memory Access

Accesses to global memory are organized as memory transactions on GPU, which are sized multiplies of 32, that is, if a warp tries to access 32 consecutive bytes on global memory in a single SIMT step, it takes one physical memory operation while the number becomes two even the necessary bytes increment by one single bit.


### Shared Memory Access with Efficient Bank Utilization

Shared memory access are reconstricted by memory banks which may be corelated to its physical layout in GPU, in which case for a single warp, only addresses satispying $${ADS}_1 % 32 \neq {ADS}_2 % 32$$ can be executed parallelly. 



## Kernel Tuning

### Fused cv1 and m.0.cv1



