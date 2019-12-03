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

__inline__ __device__ double warpReduceSum(double val) {
  // MF: I think this assumes warpSize is a power of 2.
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(0xffffffff, val, offset, warpSize);
  return val;
}

__inline__ __device__ double blockReduceSum(double val) {

  static __shared__ double warpSum[warpSize]; // Stores partial sums.
  int lane   = threadIdx.x % warpSize;
  int warpID = threadIdx.x / warpSize;

  val = warpReduceSum(val);            // Each warp performs partial reduction.

  if (lane==0) warpSum[warpID] = val;  // Write reduced value to shared memory.

  __syncthreads();                     // Wait for all partial reductions.

  // Read from shared memory (only for by the first warp).
  val = (threadIdx.x < blockDim.x / warpSize) ? warpSum[lane] : 0;

  if (warpID==0) val = warpReduceSum(val); // Final reduce within first warp.

  return val;
}

__global__ void calcMom0Ser1x1vP1(int *nCells, double *w, double *dxv, double *fIn, double *out) {
  // Calculate the zeroth moment of the distribution function. We will first assign
  // whole configuration-space cells to a single block. Then one must perform a reduction
  // across a block for each conf-space basis coefficient.
  unsigned int blockSize    = blockDim.x;
  // Configuration and velocity space index.
  unsigned int confIdx      = blockIdx.x;
  unsigned int velIdx       = threadIdx.x;
  // Index of the current phase-space cell.
  unsigned int phaseGridIdx = confIdx*blockSize + velIdx;
  // Index of the first phase-space memory address to access.
  unsigned int phaseFldIdx  = phaseGridIdx*4;  // This *4 is for polyOrder=1.

  double mySum[2];
  const double volFact = dxv[phaseGridIdx*2+1]/2;
  mySum[0] = 1.414213562373095*volFact*fIn[phaseFldIdx];
  mySum[1] = 1.414213562373095*volFact*fIn[phaseFldIdx+1];

  double blockSum;
  blockSum = blockReduceSum(mySum[0]);
  if (threadIdx.x==0)
      out[confIdx*2] = blockSum;
  blockSum = blockReduceSum(mySum[1]);
  if (threadIdx.x==0)
      out[confIdx*2+1] = blockSum;

}

int main()
{

  const int nPhaseBasisComps = 4;             // Number of monomials in phase-space basis.
  const int nConfBasisComps  = 2;             // Number of monomials in configuration-space basis.
  const int nCells[2]        = {1024, 128};   // Number of cells in x and v.

  // In choosing the following two also bear in mind that on Nvidia V100s (80 SMs):
  //   Max Warps / SM         = 64
  //   Max Threads / SM       = 2048
  //   Max Thread Blocks / SM = 32 
  const int nBlocks  = 1024;          // Number of device blocks. Max=2560 on V100s. 
  const int nThreads = 128;           // Number of device threads per block. Max=1024.

  const int totCells = nCells[0]*nCells[1];               // Total number of cells.
  const int pDim     = sizeof(nCells)/sizeof(nCells[0]);  // Phase space dimensions.
  const int confCells[1] = {nCells[0]};

  // Allocate the grid's cell center and length. Give some dummy values here.
  double *cellSize, *cellCenter;
  cellSize   = (double*) calloc (totCells*pDim, sizeof(double));
  cellCenter = (double*) calloc (totCells*pDim, sizeof(double));
  for (int idx=0; idx<totCells; idx++) {

    const unsigned int k = idx*pDim;

    cellSize[k]     = 1.0;
    cellSize[k+1]   = 1.0;
    cellCenter[k]   = 1.0;
    cellCenter[k+1] = 1.0;
  };

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
  //   k = idx-i*(nCells[1]*nPhaseBasisComps)-k*nPhaseBasisComps
  // Note: for a flattened 2D array one would have the following mapping:  
  //   idx = i*nCells[1]+j
  //   i = ((idx+1)/nCells[1])
  //   j = idx-i*nCells[1]
  for (unsigned int idx=0; idx<totCells; idx++) {

    const unsigned int k = idx*nPhaseBasisComps;

    distF[k]   = 3.14159;
    distF[k+1] = 0.20;
    distF[k+2] = 0.0;
    distF[k+3] = 0.0;
  };

  // Allocate and assign device memory containing the fields and grid.
  int *d_nCells;
  double *d_cellSize, *d_cellCenter;
  cudacall(cudaMalloc(&d_nCells, pDim*sizeof(int)));
  cudacall(cudaMalloc(&d_cellSize, pDim*totCells*sizeof(double)));
  cudacall(cudaMalloc(&d_cellCenter, pDim*totCells*sizeof(double)));
  cudacall(cudaMemcpy(d_nCells, nCells, pDim*sizeof(int), cudaMemcpyHostToDevice));
  cudacall(cudaMemcpy(d_cellSize, cellSize, pDim*totCells*sizeof(double), cudaMemcpyHostToDevice));
  cudacall(cudaMemcpy(d_cellCenter, cellCenter, pDim*totCells*sizeof(double), cudaMemcpyHostToDevice));
  double *d_distF, *d_mom0;
  cudacall(cudaMalloc(&d_distF, totCells*nPhaseBasisComps*sizeof(double)));
  cudacall(cudaMalloc(&d_mom0, confCells[0]*nConfBasisComps*sizeof(double)));
  cudacall(cudaMemcpy(d_distF, distF, totCells*nPhaseBasisComps*sizeof(double), cudaMemcpyHostToDevice));

  // Launch kernel.
  calcMom0Ser1x1vP1<<<nBlocks, nThreads>>>(d_nCells, d_cellCenter, d_cellSize, d_distF, d_mom0);

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

