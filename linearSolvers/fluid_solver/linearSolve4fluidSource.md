
### Solves a linear system of multiple right-hand sides: A * X = B
Needed in multifluid-maxwell source as well as DG moment calculation, etc.

#### CPU/`Eigen` implementation of the multifluid source
The following methods are optional.
  - `Eigen::PartialPivLU`: LU decomposition of a matrix with partial pivoting.
  -`Eigen::ColPivHouseholderQR`: Householder rank-revealing QR decomposition of a matrix with column-pivoting.
  - Comparison of different methods:
	  - https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
	  - https://eigen.tuxfamily.org/dox/group__DenseDecompositionBenchmark.html

#### GPU/`cuSOLVER`
  - The first way is two-step: `getrf` + `getrs`.
    - `getrf`: factorize A using partial pivoting LU
    - `getrs`: solve using the factored matrix
    - batched?
    - getrf: https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf
    - getrs: https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs
    - example: https://docs.nvidia.com/cuda/cusolver/index.html#lu_examples
  - Another way is a one-step call to `getsv` which maps Lapack's DSGESV
    - DSGESV first attempts to factorize the matrix in SINGLE PRECISION  and use this factorization within an iterative refinement procedure  to produce a solution with DOUBLE PRECISION normwise backward error quality (see below). If the approach fails the method switches to a DOUBLE PRECISION factorization and solve.
    - https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesv
  - batched: potrf (Cholesky factorization),  gesvd (QR algorithm), gesvdj (Jacobi method)

#### GPU/`cuBLAS`
- getrfBatched + getrsBatched: LU factorization and solver
