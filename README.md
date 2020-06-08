# About this repository

This repo contains some documents from the Gkeyll GPU
Hackathon. Please add GPU tutorial links and other useful materials
here.

# Folder contents

- clusterInfo: Information for using gpus on clusters (e.g. Portal, Adroit, Traverse).

- gentran: Notes and patch to use the code-generating library GENTRAN.

- linearSolvers: Useful resources for solving linear problems on GPUs.

- reductions: Examples and info on performing reductions on GPUs.

# General useful CUDA knowledge.

- [Grid-stride](https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
  loops are important for maintaining coalesced memory accesses
  that GPUs like for better performance.

# Relevant options in nvcc compiler.

- nvcc can treat other files without the ".cu" extension as CUDA files
  via the [-x flag](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-altering-compiler-linker-behavior-x).

- If your device kernels call device kernels located in other files,
  special precautions may be needed to compile the project, [see here](https://devblogs.nvidia.com/separate-compilation-linking-cuda-device-code/).

# Links to useful GPU tutorials and papers

- See Princeton University [GPU Intro
  github](https://github.com/PrincetonUniversity/gpu_programming_intro)
  page.

- The most comprehensive programming documentation is [NVIDIA's C
  Programing
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

- The complete CUDA toolkit is documented [here](https://docs.nvidia.com/cuda/index.html)

- Paper with gory details on [Volta GPU Architectures via
  Microbenchmarking](https://arxiv.org/pdf/1804.06826.pdf). Alternatively this [short
  white paper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
  also has useful details (e.g. number of SMs, blocks, threads, etc).

- Blog post on implementing [finite-difference codes on
  GPUs](https://devblogs.nvidia.com/finite-difference-methods-cuda-cc-part-1/)
  
- It currently seems that maximizing register usage (at the expense of occupancy) is optimal for our kernels. Here is a [talk](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf) that supports this idea (it's from 2010, but I think a lot of the ideas are still true). The key is that instruction-level parallelism (each thread has a lot of work to do) is an alternative (and sometimes more optimal) way to hide latency, instead of running more threads (higher occupancy). Our kernels have a lot of ILP. 

- one problem that we might face is that shared memory is still slower than registers. so even though using shared memory improves the initial accesses (via allowing coalescing), subsequent fetches of the data in shared memory is slower than if the data was in registers.  I found something interesting called register caching, where threads in a warp (groups of 32 threads) can read each others’ registers. [this](https://devblogs.nvidia.com/register-cache-warp-cuda/) CUDA blog post has some of the gory details, but I also found this library called [trove](https://github.com/bryancatanzaro/trove) that uses register caching to optimize coalesced memory accesses for chunked data layouts like ours. the diagram on the page gives an idea of how this works. for our applications, say we have nComp=32 (which is convenient because this is the warp size, so that we don’t have to worry about the cell indices being contiguous). in our algorithm, each thread in a warp is responsible for doing operations in a cell. but instead of having each thread read its own 32 elements of cell data, we could use register caching to have thread i read the ith element in each cell, and after looping over cells, we would have all the data in all the cells. we could then share the data between all 32 threads, and each thread could then work on its own 32 element chunk. using register caching for all of this avoids using shared memory, allowing us to have the benefits of coalesced memory access while keeping the data in fast registers.
