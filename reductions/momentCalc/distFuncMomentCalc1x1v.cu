// ................................................ //
//
// Create a dummy distribution function in 1x1v
// and compute its zeroth velocity moment.
//
// Manaure Francisquez.
// November 2019.
//
// ................................................ //
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

#define warpSize 32


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

void printSolution(const double *x, const int nX, const int xDim, const int nCoeffs) {
  // Function to print solution to screen.
//  for(int i=0; i<nX; i++){ 
  for(int i=0; i<5; i++){ 
    for(int k=0; k<nCoeffs; k++){ 
      printf(" %f ",x[i*nCoeffs+k]);
    };
    printf("\n");
  };
}

__inline__ __device__ void warpReduceComponentsSum(double *vals, int nComps) {
  // Perform 'nComps' independent (sum) reductions across a warp,
  // one for each component in 'vals'.

  // MF: I think this assumes warpSize is a power of 2.
  for (unsigned int k = 0; k < nComps; k++) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
      vals[0+k] += __shfl_down_sync(0xffffffff, vals[0+k], offset, warpSize);
    }
  }
}

__inline__ __device__ void blockReduceComponentsSum(double *vals, int nComps) {
  // Perform 'nComps' independent (sum) reductions across a block,
  // one for each component in 'nComps'.

  extern __shared__ double blockSum[]; // Stores partial sums.
  int lane   = threadIdx.x % warpSize;
  int warpID = threadIdx.x / warpSize;

  warpReduceComponentsSum(vals, nComps);            // Each warp performs partial reduction.

  if (lane==0) {
    // Write reduced value to shared memory.
    for (unsigned int k = 0; k < nComps; k++) {
      blockSum[warpID*nComps+k] = vals[k];
    }
  }

  __syncthreads();                     // Wait for all partial reductions.

  // Read from shared memory (only for by the first warp).
  for (unsigned int k = 0; k < nComps; k++) {
    vals[k] = (threadIdx.x < blockDim.x / warpSize) ? blockSum[lane*nComps+k] : 0;
  }

  if (warpID==0) warpReduceComponentsSum(vals, nComps); // Final reduce within first warp.

}

__host__ __device__ void MomentCalc1x1vSer_M0_P1(const double *w, const double *dxv, const double *f, double *out)
{
  const double volFact = dxv[1]/2;
  out[0] += 1.414213562373095*f[0]*volFact;
  out[1] += 1.414213562373095*f[1]*volFact;
}

__global__ void calcMom1x1vSer_M0_P1(int *nCells, double *w, double *dxv, double *fIn, double *out) {
  // Calculate the zeroth moment of the distribution function. We will first assign
  // whole configuration-space cells to a single block. Then one must perform a reduction
  // across a block for each conf-space basis coefficient.
  // Index of the current phase-space cell.
  unsigned int phaseGridIdx = blockIdx.x*blockDim.x + threadIdx.x;

  const unsigned int pDim = 2;    // Phase space dimension;
  const unsigned int nP   = 4;    // Number of phase-space basis functions.
  const unsigned int nC   = 2;    // Number of configuration-space basis functions.

  // Configuration and velocity space indexes.
  unsigned int confIdx = phaseGridIdx/nCells[1];
  unsigned int velIdx  = phaseGridIdx-confIdx*nCells[1];

  // Index of the first phase-space memory address to access.
  unsigned int phaseFldIdx = phaseGridIdx*nP;

  double localSum[nC];
  for (unsigned int k = 0; k < nC; k++) {
    localSum[k] = 0.0;  // Need to zero this out because kernel below increments.
  }

  // Pointers to quantities expected by the moment kernel.
  double *distF       = &fIn[phaseFldIdx];
  double *cellCenter  = &w[0];
  double *cellSize    = &dxv[0];
  double *localSumPtr = &localSum[0];

  MomentCalc1x1vSer_M0_P1(cellCenter, cellSize, distF, localSumPtr);

  blockReduceComponentsSum(localSumPtr, nC);
  if (threadIdx.x==0) {
    out[confIdx*nC]   = localSumPtr[0];
    out[confIdx*nC+1] = localSumPtr[1];
  }
}

int main()
{

  const int nPhaseBasisComps = 4;             // Number of monomials in phase-space basis.
  const int nConfBasisComps  = 2;             // Number of monomials in configuration-space basis.
  const int nCells[2]        = { 512*512, 128 };   // Number of cells in x and v.

  // In choosing the following two also bear in mind that on Nvidia V100s (80 SMs):
  //   Max Warps / SM         = 64
  //   Max Threads / SM       = 2048
  //   Max Thread Blocks / SM = 32 
  const int nBlocks  = 512*512;          // Number of device blocks. Max=2560 on V100s. 
  const int nThreads = 128;           // Number of device threads per block. Max=1024.

  const int totCells     = nCells[0]*nCells[1];               // Total number of cells.
  const int pDim         = sizeof(nCells)/sizeof(nCells[0]);  // Phase space dimensions.
  const int confCells[1] = { nCells[0] };

  // Allocate the grid's cell center and length. Give some dummy values here.
  double *cellSize, *cellCenter;
  cellSize   = (double*) calloc (pDim, sizeof(double));
  cellCenter = (double*) calloc (pDim, sizeof(double));
  cellSize[0]   = 1.0;
  cellSize[1]   = 0.10;
  cellCenter[0] = 1.0;
  cellCenter[1] = 1.0;

  // Distribution function and zeroth moment.
  double *distF, *mom0;
  distF = (double*) calloc (totCells*nPhaseBasisComps, sizeof(double));
  mom0  = (double*) calloc (confCells[0]*nConfBasisComps, sizeof(double));

  // Assign some coefficients to distF and mom0 in each cell. For now
  // assume that the basis index is the fastest changing index, followed
  // by the velocity-space index and then the configuration-space index.
  // That is, if f[i,j,k]=f(x(i),v(j),k) where k indexes the basis monomial,
  // then the index in the following loop corresponds to
  //   idx = (i*nCells[1]+j)*nPhaseBasisComps+k
  //   i = (idx+1)/(nCells[1]*nPhaseBasisComps)
  //   j = (idx-i*(nCells[1]*nPhaseBasisComps)+1)/nPhaseBasisComps
  //   k = idx-i*(nCells[1]*nPhaseBasisComps)-j*nPhaseBasisComps
  // Note: for a flattened 2D array one would have the following mapping:  
  //   idx = i*nCells[1]+j
  //   i = ((idx+1)/nCells[1])
  //   j = idx-i*nCells[1]
  for (int idx=0; idx<totCells; idx++) {

    int k = idx*nPhaseBasisComps;

    distF[k]   = 3.14159;
    distF[k+1] = 0.20;
    distF[k+2] = 0.0;
    distF[k+3] = 0.0;
  };

  // Allocate and assign device memory containing the fields and grid.
  int *d_nCells;
  double *d_cellSize, *d_cellCenter;
  cudacall(cudaMalloc(&d_nCells, pDim*sizeof(int)));
  cudacall(cudaMalloc(&d_cellSize, pDim*sizeof(double)));
  cudacall(cudaMalloc(&d_cellCenter, pDim*sizeof(double)));
  cudacall(cudaMemcpy(d_nCells, nCells, pDim*sizeof(int), cudaMemcpyHostToDevice));
  cudacall(cudaMemcpy(d_cellSize, cellSize, pDim*sizeof(double), cudaMemcpyHostToDevice));
  cudacall(cudaMemcpy(d_cellCenter, cellCenter, pDim*sizeof(double), cudaMemcpyHostToDevice));
  double *d_distF, *d_mom0;
  cudacall(cudaMalloc(&d_distF, totCells*nPhaseBasisComps*sizeof(double)));
  cudacall(cudaMalloc(&d_mom0, confCells[0]*nConfBasisComps*sizeof(double)));
  cudacall(cudaMemcpy(d_distF, distF, totCells*nPhaseBasisComps*sizeof(double), cudaMemcpyHostToDevice));

  // Launch kernel.
  calcMom1x1vSer_M0_P1<<<nBlocks, nThreads, nConfBasisComps*(nThreads/warpSize)*sizeof(double)>>>(d_nCells, d_cellCenter, d_cellSize, d_distF, d_mom0);

  // Copy result to host.
  cudacall(cudaMemcpy(mom0, d_mom0, confCells[0]*nConfBasisComps*sizeof(double), cudaMemcpyDeviceToHost));

  printf("CUDA kernel moment:\n");
  printSolution(mom0, confCells[0], 1, nConfBasisComps);

  // Free device memory.
  cudaFree(d_cellCenter); cudaFree(d_cellSize); cudaFree(d_nCells); 
  cudaFree(d_mom0); cudaFree(d_distF);

  // Free dynamically allocated memory.
  free(distF); free(mom0);
  free(cellSize); free(cellCenter);

  return 0;
}

