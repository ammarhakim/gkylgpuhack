# Reductions

Data reduction is a common operation in scientific computing (e.g.
dot products, finding min/max, global sums). Doing it efficiently
in parallel programs takes more thought than a serial implementation,
especially in GPUs.
 
In this folder we place useful resources for the development of
reductions.

# Folders:

- momentCalc: sample code mimicking the reduction that takes place
              in computing moments of the distribution function in
              Gkeyll.

# Useful links:

- Old presentation covering some of the ideas important for GPU
  reductions is [found here](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf).

- Some efficient implementations use [shuffle instructions](https://devblogs.nvidia.com/faster-parallel-reductions-kepler/).

- Examples with and without shuffle instructions are in the
  [CUDA samples repo](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/reduction).

- In understanding some of the above examples (and optimizing
  other codes), one may find it useful to learn about [warp-level primitives](Warp-Level Primitives),
  [grid stride loops](https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) and
  [cooperative groups](https://devblogs.nvidia.com/cooperative-groups/).


