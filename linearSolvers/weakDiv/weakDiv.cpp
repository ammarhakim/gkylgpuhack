// ................................................ //
//
// Perform the weak division mom0*u=mom1 with EIGEN
// and CUDA. This operation is simply a solution of the
// linear problem Au = B, where A arises from the weak
// product of mom0 and u, and B is the projection of
// mom1 on the basis. We will solve this in a
// number of cells (nCells).
//
// Manaure Francisquez.
// November 2019.
//
// ................................................ //
#include <math.h>
#include <iostream>
#include <stdio.h>
//#include "weakDiv_Eigen.cpp"
#include "weakDiv_CUDA.cu"

void printSolution(const double *x, const int nX, const int xDim, const int nCoeffs) {
  // Function to print solution to screen.
  for(int i=0; i<nX; i++){ 
    for(int k=0; k<nCoeffs; k++){ 
      printf(" %f ",x[i*nCoeffs+k]);
    };
    printf("\n");
  };
}


int main()
{

  const int nCells = 10;    // Number of cells.
  const int nBasis = 3;     // Number of monomials in basis.
  const int uDim   = 1;     // Number of velocity (vector) components.

  // Allocate arrays containing zeroth moment (mom0), first
  // moment (mom1) and mean flow velocity (u).
//  double *mom0, *mom1, *u;
  double *mom3;
  mom3 = (double*) calloc (nCells*nBasis,sizeof(double));
  free(mom3);
//  mom1 = (double*) calloc (nCells*nBasis,sizeof(double));
//  u    = (double*) calloc (nCells*nBasis*uDim,sizeof(double));
//
//  // Asign some coefficients to mom0 and mom1 in each cell.
//  for (int i=0; i<nCells; i++) {
//
//    int k = i*nBasis;
//
//    mom0[k+0] = 1.0;
//    mom0[k+1] = 0.50;
//    mom0[k+2] = 0.01;
//
//    mom1[k+0] = 1.0;
//    mom1[k+1] = 1.0;
//    mom1[k+2] = 1.0;
//  };

  int *statusFlag;
//  // Solve the problem with Eigen.
//  statusFlag = solveLinearSystemsEIGEN(mom0,nBasis,mom1,u,uDim,nCells);
//
//  printf("Eigen solves:\n");
//  printSolution(u, nCells, uDim, nBasis);
//
//  // Clear solution vector u.
//  for(int i=0; i<nCells; i++){ 
//    for(int k=0; k<nBasis; k++){ 
//      u[i*nBasis+k] = 0.0;
//    };
//  };

  // Solve the problem with CUDA.
//  statusFlag = solveLinearSystemsCUDA(mom0,nBasis,mom1,u,uDim,nCells);
  statusFlag = solveLinearSystemsCUDA();

  printf("CUDA solves:\n");
//  printSolution(u, nCells, uDim, nBasis);

  // Free dynamically allocated memory.
//  free(mom0); free(mom1); free(u);

  return 0;
}
