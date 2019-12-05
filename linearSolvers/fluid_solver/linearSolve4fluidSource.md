
### Solves a linear system of multiple right-hand sides: `A * X = B`
Needed in multifluid-maxwell source as well as DG moment calculation, etc.

#### CPU/`Eigen` implementation of the multifluid source
The following methods are optional:
  - [`Eigen::PartialPivLU`](https://eigen.tuxfamily.org/dox/classEigen_1_1PartialPivLU.html): LU decomposition of a matrix with partial pivoting.
  - [`Eigen::ColPivHouseholderQR`](https://eigen.tuxfamily.org/dox/classEigen_1_1ColPivHouseholderQR.html): Householder rank-revealing QR decomposition of a matrix with column-pivoting.
  - Examples and benchmarks of different methods:
  	- https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
	- https://eigen.tuxfamily.org/dox/group__DenseDecompositionBenchmark.html

#### GPU/[`cuSOLVER`](https://docs.nvidia.com/cuda/cusolver/index.html) implementation (tentative):
  - From [Intel MKL doc](https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapackro1.htm):
    - To solve a system of linear equations with a general matrix, call [?getrf](https://software.intel.com/en-us/mkl-developer-reference-c-getrf) (LU factorization) and then [?getrs](https://software.intel.com/en-us/mkl-developer-reference-c-getrs) (computing the solution).
    - Alternatively, use [?gesv](https://software.intel.com/en-us/mkl-developer-reference-c-gesv), which performs all these tasks in one call.
    - Naming convention: `ge` for general matrix, `trf` for triangular factorization, `trs` for s for triangular solving, `sv` for solving a linear system 
  - [`cuSolver`](https://docs.nvidia.com/cuda/cusolver/index.html) counterpart:
    - [`getrf`](https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf): Factorize A using partial pivoting LU.
    - [`getrs`](https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs): Solve using the factored matrix.
    - [`gesv`](https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesv): Computes the solution to the system of linear equations with a square matrix A and multiple right-hand sides.

#### GPU/[`cuBLAS`](https://docs.nvidia.com/cuda/cublas/index.html) implementation (tentative):
- Only two-step procedures (factorization and solver) are available: [getrfBatched](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrfbatched) + [getrsBatched](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrsbatched)
- Reular `Blas` do not seem to include `getrf`/`getrs` as they should belong to the higher-level `Lapack` library.

#### Batched vs non-batched?
- In `cuSolver`, none of LU/QR/Cholesky operations seem to be batched, therefore we would need to loop over each cell.
- In `cuBlas`, the procedures are batched.
- Performance? Interface complexity?
