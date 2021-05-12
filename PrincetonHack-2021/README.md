# Preliminary thoughts for 2021 Princeton GPU Hackathon

Due to ongoing development on gkylzero, a new lower level C-driven
interface for gkyl, our priority for the 2021 Princeton GPU Hackathon
are components of the code which will be most easily transferred
to the larger infrastructure work on gkylzero. Because many updaters
are being re-written in C, it is not necessarily a good use of our time
to port many of the updaters to GPUs yet, as this porting will be more natural
with the gkylzero versions of these updaters.

Thus, we prioritize the exploration of CUDA toolkit libraries which 
are necessary and/or useful (for performance and maintainability)
for the current operations, updaters, and solvers within gkyl.

The full list of CUDA-accelerated libraries can be found [here](https://developer.nvidia.com/gpu-accelerated-libraries)

# [cuSolver](https://developer.nvidia.com/cusolver)

- BinOp updater (binary weak operations) loops over grid and performs small linear solves within each grid cell.
  Operations are currently done with Eigen on the CPU. Using cuSolver, we can probably do batched dense linear solves
  such that every thread on the GPU is doing a small matrix inversion (since every thread owns a single grid cell).

- Direct Poisson solver. Current ongoing work with respect to Poisson solvers is split between direct solves
  (which are very fast for small systems), multi-grid (which scales well but is still being experimented with for
  the exact optimal relaxation parameters), and iterative methods (which have not yet been extended to continuous
  finite element methods, a requirement for the gyrokinetic and Vlasov-Poisson Poisson solves). There are an enormous
  number of options within cuSolver (cuSolverSP, which does sparse QR factorization, cuSolverRF, a sparse refactorization
  package, and different options for the dense linear solve such as LU and QR). Many of these routines also support batched
  implementations and multi-GPU implementations depending on how "parallel" the Poisson solve can be made. Demonstration of
  a Poisson solve on a GPU (and potentially a full Vlasov-Poisson simulation using the current Vlasov infrastructure) would
  be a significant step forward.

# [CUB](https://nvlabs.github.io/cub/)

- Currently the reduction operations (sums over velocity space cells for moments, finding the stable time-step calling a minimum
  function on the cflRateByCell, etc.) are handled by hand with reductions over warps that only work for certain numbers of grid
  cells. The performance is good, but there are entire libraries such as CUB (CUDA UnBound) which ostensibly provide superior
  performance and long term maintainability, providing the explicit reduction operations we require across threads, warps, and blocks.
  An exploration of whether a moment calculation routine using CUB provides at least comparable performance (or better) while allowing
  more general velocity space grids and cleaner code (or better yet, parallelism in velocity space across GPUs) would be an ideal outcome.

# [cuTensor](https://developer.nvidia.com/cutensor)

- Now that the tensor cores on Nvidia Amperes support double precision arithmetic, can we leverage cuTensor for all our tensor-tensor
  operations within gkyl. This includes reduction operations (moment calculation and minimum CFL) as well as update formulas for 
  the volume and surface integrals. *Note that this may be highly non-trivial to implement given the syntax expected for the use
  of cuTensor, but if our mentors are experts perhaps this is something that can be experimented with*.