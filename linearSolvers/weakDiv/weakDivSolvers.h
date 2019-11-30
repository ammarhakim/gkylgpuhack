#ifndef _weakDivSolvers_H_
#define _weakDivSolvers_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

int solveLinearSystemsEIGEN(double *lhsA, int probSize, double *rhsB, double *x, int xDim, int nProbs);
int solveLinearSystemsCUBLAS(double *lhsA, int probSize, double *rhsB, double *x, int xDim, int nProbs);

#endif

