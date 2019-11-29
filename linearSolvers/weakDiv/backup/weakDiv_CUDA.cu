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

int *solveLinearSystemsCUDA(const double *lhsA,const int probSize, double *rhsB,double *x,const int xDim,const int nProbs) {
  // Solve nProbs linear systems A.x=B of size probSize,
  // where x and B have xDim components.

  int *solveStatus;

  // Store the left-side matrix as an array of matrices.
  double **As = (double **)malloc(nProbs*sizeof(double *));
  for (int i=0; i<nProbs; i++) {

    int k = i*probSize;

    As[i] = (double*) calloc (probSize*probSize,sizeof(double));
    As[i][0] = 0.7071067811865475*lhsA[k+0];
    As[i][1] = 0.7071067811865475*lhsA[k+1];
    As[i][2] = 0.7071067811865475*lhsA[k+2];
    As[i][3] = 0.7071067811865475*lhsA[k+1];
    As[i][4] = 0.6324555320336759*lhsA[k+2]+0.7071067811865475*lhsA[k+0];
    As[i][5] = 0.6324555320336759*lhsA[k+1];
    As[i][6] = 0.7071067811865475*lhsA[k+2];
    As[i][7] = 0.6324555320336759*lhsA[k+1];
    As[i][8] = 0.4517539514526256*lhsA[k+2]+0.7071067811865475*lhsA[k+0];
  };

  // Store the right-side vector as an array of vectors.
  double **Bs = (double **)malloc(nProbs*sizeof(double *));
  for (int i=0; i<nProbs; i++) {
    Bs[i] = rhsB+i*probSize;
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
  cudacall(cudaMalloc(&A_d,nProbs*sizeof(double *)));
  cudacall(cudaMalloc(&Aflat_d, probSize*probSize*nProbs*sizeof(double)));
  double **A_h = (double **)malloc(nProbs*sizeof(double *));
  A_h[0] = Aflat_d;
  for (int i = 1; i < nProbs; i++)
    A_h[i] = A_h[i-1]+(probSize*probSize);
  cudacall(cudaMemcpy(A_d,A_h,nProbs*sizeof(double *),cudaMemcpyHostToDevice));
  for (int i = 0; i < nProbs; i++)
    cudacall(cudaMemcpy(Aflat_d+(i*probSize*probSize), As[i], probSize*probSize*sizeof(double), cudaMemcpyHostToDevice));

  // Perform the LU decomposition.
  const int lda = probSize;
  int *P_d;       // Pivots.
  int *info_d;    // Error info.
  cudacall(cudaMalloc(&P_d, probSize * nProbs * sizeof(int)));
  cudacall(cudaMalloc(&info_d, nProbs*sizeof(int)));
  cublascall(cublasDgetrfBatched(cublasHandle,probSize,A_d,lda,P_d,info_d,nProbs));

  // Check that LU decomposition was successful.
  int infos[nProbs];
  cudacall(cudaMemcpy(infos,info_d,nProbs*sizeof(int),cudaMemcpyDeviceToHost));
  for (int i = 0; i < nProbs; i++)
    if (infos[i]  != 0)
    {
      fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
      cudaDeviceReset();
      exit(EXIT_FAILURE);
    }

  // Allocate and assign the right-side device-vectors, following a procedure
  // similar to that in allocating and assigning the left-side device-matrices. 
  double **B_h = (double **)malloc(nProbs*sizeof(double *));
  double **B_d, *Bflat_d;
  cudacall(cudaMalloc(&B_d,nProbs*sizeof(double *)));
  cudacall(cudaMalloc(&Bflat_d, probSize*xDim*nProbs*sizeof(double)));
  B_h[0] = Bflat_d;
  for (int i = 1; i < nProbs; i++)
    B_h[i] = B_h[i-1] + probSize*xDim;
  cudacall(cudaMemcpy(B_d,B_h,nProbs*sizeof(double *),cudaMemcpyHostToDevice));
  for (int i = 0; i < nProbs; i++)
    cudacall(cudaMemcpy(Bflat_d+(i*probSize*xDim), Bs[i], probSize*xDim*sizeof(double), cudaMemcpyHostToDevice));

  // Compute the solution to the linear systems.
  int *info;    // Error info. NOTE: getrsBatched expects a host info, not a device info.
  const int ldb = probSize;
  cublascall(cublasDgetrsBatched(cublasHandle,CUBLAS_OP_N,probSize,xDim,(const double **)A_d,lda,P_d,B_d,ldb,info,nProbs));

//  // Copy solutions from device to host.
//  for (int i = 0; i < nProbs; i++)
//    cudacall(cudaMemcpy(x+(i*probSize), Bflat_d + (i*probSize), probSize*sizeof(double), cudaMemcpyDeviceToHost));

  // Free device memory.
  cublasDestroy_v2(cublasHandle);
  cudaFree(A_d); cudaFree(Aflat_d); cudaFree(info_d);
  cudaFree(P_d);
  cudaFree(B_d); cudaFree(Bflat_d);

  // Free host memory.
  free(As); free(Bs);
  free(A_h); free(B_h);

  solveStatus = 0;

  return solveStatus;
}


