// ............................................................. //
//
// Function computing weak division A.x=B to obtain x in
// multiple cells, using CUDA. The vectors x and B can have
// multiple components (may not be fully supported yet).
//
// Manaure Francisquez.
// November 2019.
//
// ............................................................. //
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

int *solveLinearSystemsCUDA() {
  // Solve nProbs linear systems A.x=B of size probSize,
  // where x and B have xDim components.

  int *solveStatus;

  const int nCells = 10;    // Number of cells.
  const int nBasis = 3;     // Number of monomials in basis.
  const int uDim   = 1;     // Number of velocity (vector) components.

  // Allocate arrays containing zeroth moment (mom0), first
  // moment (mom1) and mean flow velocity (u).
  double *mom0, *mom1, *u;
  mom0 = (double*) calloc (nCells*nBasis,sizeof(double));
  if (mom0==NULL) printf("error here 1\n");
  mom1 = (double*) calloc (nCells*nBasis,sizeof(double));
  if (mom1==NULL) printf("error here 2\n");
  u    = (double*) calloc (nCells*nBasis*uDim,sizeof(double));
  if (u==NULL) printf("error here 3\n");

  // Assign some coefficients to mom0 and mom1 in each cell.
  for (int i=0; i<nCells; i++) {

    int k = i*nBasis;

    mom0[k+0] = 1.0; 
    mom0[k+1] = 0.50; 
    mom0[k+2] = 0.01; 
  
    mom1[k+0] = 1.0; 
    mom1[k+1] = 1.0; 
    mom1[k+2] = 1.0; 
  };

  // Store the left-side matrix as an array of matrices.
  double **As = (double **)malloc(nCells*sizeof(double *));
  if (As==NULL) printf("error here 4\n");
  for (int i=0; i<nCells; i++) {

    int k = i*nBasis;

    As[i] = (double*) calloc (nBasis*nBasis,sizeof(double));
    if (As[i]==NULL) printf("error here 5\n");
    As[i][0] = 0.7071067811865475*mom0[k+0];
    As[i][1] = 0.7071067811865475*mom0[k+1];
    As[i][2] = 0.7071067811865475*mom0[k+2];
    As[i][3] = 0.7071067811865475*mom0[k+1];
    As[i][4] = 0.6324555320336759*mom0[k+2]+0.7071067811865475*mom0[k+0];
    As[i][5] = 0.6324555320336759*mom0[k+1];
    As[i][6] = 0.7071067811865475*mom0[k+2];
    As[i][7] = 0.6324555320336759*mom0[k+1];
    As[i][8] = 0.4517539514526256*mom0[k+2]+0.7071067811865475*mom0[k+0];
  };

  // Store the right-side vector as an array of vectors.
  double **Bs = (double **)malloc(nCells*sizeof(double *));
  if (Bs==NULL) printf("error here 6\n");
  for (int i=0; i<nCells; i++) {
    Bs[i] = (double*) calloc (nBasis,sizeof(double));
    if (Bs[i]==NULL) printf("error here 7\n");
    Bs[i] = mom1+i*nBasis;
  };


  // Create CUBLAS solver.
  cublasHandle_t cublasHandle = NULL;
  cublascall(cublasCreate(&cublasHandle));

  // CUBLAS batched routines expect an array of pointers, each pointer
  // addressing a different linear problem. The procedure followed here is
  // to:
  //   1) Allocate a device-array of pointers, one entry per cell (A_d),
  //      and a device-array with the data of all cells (Aflat_d).
  //   2) Construct a host-array of pointers (A_h), each pointing to the address
  //      of the left-side matrix of the corresponding cell on the device.
  //   3) Copy the array of pointers (A_h) to the device-array of pointers (A_d).
  //   4) Copy the left-side matrix data (in As) to the device-array (Aflat_d),
  //      one cell at a time.
  double **A_d, *Aflat_d;
  cudacall(cudaMalloc(&A_d,nCells*sizeof(double *)));
  cudacall(cudaMalloc(&Aflat_d, nBasis*nBasis*nCells*sizeof(double)));
  double **A_h = (double **)malloc(nCells*sizeof(double *));
  A_h[0] = Aflat_d;
  for (int i = 1; i < nCells; i++)
    A_h[i] = A_h[i-1]+(nBasis*nBasis);
  cudacall(cudaMemcpy(A_d,A_h,nCells*sizeof(double *),cudaMemcpyHostToDevice));
  for (int i = 0; i < nCells; i++)
    cudacall(cudaMemcpy(Aflat_d+(i*nBasis*nBasis), As[i], nBasis*nBasis*sizeof(double), cudaMemcpyHostToDevice));

  // Perform the LU decomposition.
  const int lda = nBasis;
  int *P_d;       // Pivots.
  int *info_d;    // Error info.
  cudacall(cudaMalloc(&P_d, nBasis * nCells * sizeof(int)));
  cudacall(cudaMalloc(&info_d, nCells*sizeof(int)));
  cublascall(cublasDgetrfBatched(cublasHandle,nBasis,A_d,lda,P_d,info_d,nCells));

  // Check that LU decomposition was successful.
  int infos[nCells];
  cudacall(cudaMemcpy(infos,info_d,nCells*sizeof(int),cudaMemcpyDeviceToHost));
  for (int i = 0; i < nCells; i++)
    if (infos[i]  != 0)
    {
      fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
      cudaDeviceReset();
      exit(EXIT_FAILURE);
    }

  // Allocate and assign the right-side device-vectors, following a procedure
  // similar to that in allocating and assigning the left-side device-matrices. 
  double **B_d, *Bflat_d;
  cudacall(cudaMalloc(&B_d,nCells*sizeof(double *)));
  cudacall(cudaMalloc(&Bflat_d, nBasis*uDim*nCells*sizeof(double)));
  double **B_h = (double **)malloc(nCells*sizeof(double *));
  B_h[0] = Bflat_d;
  for (int i = 1; i < nCells; i++)
    B_h[i] = B_h[i-1] + nBasis*uDim;
  cudacall(cudaMemcpy(B_d,B_h,nCells*sizeof(double *),cudaMemcpyHostToDevice));
  for (int i = 0; i < nCells; i++)
    cudacall(cudaMemcpy(Bflat_d+(i*nBasis*uDim), Bs[i], nBasis*uDim*sizeof(double), cudaMemcpyHostToDevice));

  // Compute the solution to the linear systems.
  int *info;    // Error info. NOTE: getrsBatched expects a host info, not a device info.
  const int ldb = nBasis;
  cublascall(cublasDgetrsBatched(cublasHandle,CUBLAS_OP_N,nBasis,uDim,(const double **)A_d,lda,P_d,B_d,ldb,info,nCells));

  // Copy solutions from device to host.
  for (int i = 0; i < nCells; i++)
    cudacall(cudaMemcpy(u+(i*nBasis), Bflat_d + (i*nBasis), nBasis*sizeof(double), cudaMemcpyDeviceToHost));

  // Free device memory.
  cublasDestroy_v2(cublasHandle);
  cudaFree(A_d); cudaFree(Aflat_d); cudaFree(info_d);
  cudaFree(P_d);
  cudaFree(B_d); cudaFree(Bflat_d);

  // Print solutions.
  for(int i=0; i<nCells; i++){ 
    for(int k=0; k<nBasis; k++){ 
      printf(" %f ",u[i*nBasis+k]);
    };
    printf("\n");
  };

  // Free host memory.
  free(mom0); free(mom1); free(u);
  free(As); free(Bs);
  free(A_h); free(B_h);

  solveStatus = 0;

  return solveStatus;
}
