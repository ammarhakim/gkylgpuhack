# Linear solver info and tests

This folder contains useful information for GPUfying linear solvers
(e.g. Poisson solvers, weak division). There are also some minimum
working examples.

# Folders

- weakDiv: an example mimicking the weak division operation found
  in the binOp updater of Gkeyll.

# Useful links:

- Through CUDA one would probably use CUBLAS or cuSolver routines (modeled
  after LAPACK). [CUBLAS](https://docs.nvidia.com/cuda/cublas/index.html), 
  [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html).

- Sometimes one needs to perform batch solves (solving many small problems).
  CUBLAS has batched routines. Useful examples: [C++](https://devtalk.nvidia.com/default/topic/767806/gpu-accelerated-libraries/matrix-inversion-with-cublassgetri/), and [Fortran](https://devblogs.nvidia.com/cuda-pro-tip-how-call-batched-cublas-routines-cuda-fortran/).

- The Eigen library is a popular linear algebra package. It is possible to
  combine CUDA and Eigen, [see here](https://eigen.tuxfamily.org/dox/TopicCUDA.html).
  
- A google search for "gpu finite element assembly" could perhaps give a good starting point for how to design kernels for FEM assembly on GPU. Here are a few search results that look interesting:

   - https://mc.stanford.edu/cgi-bin/images/e/e1/Darve_FEM_GPU_2009.pdf
   - http://on-demand.gputechconf.com/gtc/2015/presentation/S5403-Nelson-Inoue%20.pdf



