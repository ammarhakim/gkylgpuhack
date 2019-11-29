// ................................................ //
//
// Perform the weak division mom0*u=mom1 with EIGEN.
// This operation is simply a solution of the
// linear problem Au = B, where A arises from the weak
// product of mom0 and u, and B is the projection of
// mom1 on the basis. We will solve this in a
// number of cells (Ncells).
//
// Manaure Francisquez.
// November 2019.
//
// ................................................ //
#include <math.h>
#include <Eigen/Dense>
#include <iostream>
#include <stdio.h>

using namespace Eigen;

int main()
{

  int Ncells = 10;
  int nbasis = 3;

  double *A, *B, *u;
  A = (double*) calloc (Ncells*nbasis,sizeof(double));
  B = (double*) calloc (Ncells*nbasis,sizeof(double));
  u = (double*) calloc (Ncells*nbasis,sizeof(double));

  for (int i=0; i<Ncells; i++) {

    int k = i*nbasis;

    A[k+0] = 1.0; 
    A[k+1] = 0.50; 
    A[k+2] = 0.01; 
  
    B[k+0] = 1.0; 
    B[k+1] = 1.0; 
    B[k+2] = 1.0; 
  };

  Eigen::MatrixXd A_EM;
  Eigen::VectorXd B_EV;
  Eigen::VectorXd u_EV;

  // Left side Eigen matrix.
  A_EM = Eigen::MatrixXd::Zero(nbasis,nbasis);
  // Right side Eigen Vector.
  B_EV = Eigen::VectorXd::Zero(nbasis);
  // Eigen vector containing the solution.
  u_EV = Eigen::VectorXd::Zero(nbasis);

  // Loop over cells.
  for (int i=0; i<Ncells; i++) {

    int k = i*nbasis;

    // Fill the left side matrix according to the weak product of A and u.
    A_EM = Eigen::MatrixXd::Zero(nbasis,nbasis);
    A_EM(0,0) = 0.7071067811865475*A[k+0];
    A_EM(0,1) = 0.7071067811865475*A[k+1];
    A_EM(0,2) = 0.7071067811865475*A[k+2];
    A_EM(1,0) = 0.7071067811865475*A[k+1];
    A_EM(1,1) = 0.6324555320336759*A[k+2]+0.7071067811865475*A[k+0];
    A_EM(1,2) = 0.6324555320336759*A[k+1];
    A_EM(2,0) = 0.7071067811865475*A[k+2];
    A_EM(2,1) = 0.6324555320336759*A[k+1];
    A_EM(2,2) = 0.4517539514526256*A[k+2]+0.7071067811865475*A[k+0];

    // Assign the right side vector B.
    B_EV << B[k+0],B[k+1],B[k+2];

    // Solve the system of equations.
    u_EV = A_EM.colPivHouseholderQr().solve(B_EV);

    // Extract solution from Eigen vector and place it in c++ array.
    Eigen::Map<VectorXd>(u+k,nbasis,1) = u_EV;

  };

  for(int i=0; i<Ncells; i++){ 
    for(int k=0; k<nbasis; k++){ 
      printf(" %f ",u[i*nbasis+k]);
    };
    printf("\n");
  };

  // Free dynamically allocated memory.
  free (A);
  free (B);
  free (u);
  return 0;
}
