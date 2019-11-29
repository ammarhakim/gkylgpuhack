// ............................................................. //
//
// Function computing weak division A.x=B to obtain x in
// multiple cells, using Eigen. The vectors x and B can have
// multiple components (may not be fully supported yet).
//
// Manaure Francisquez.
// November 2019.
//
// ............................................................. //

#include <Eigen/Dense>

using namespace Eigen;

int *solveLinearSystemsEIGEN(const double *lhsA,const int probSize, double *rhsB,double *x,const int xDim,const int nProbs) {
  // Solve nProbs linear systems A.x=B of size probSize,
  // where x and B have xDim components.

  int *solveStatus;

  Eigen::MatrixXd A_EM;
  Eigen::VectorXd B_EV;
  Eigen::VectorXd u_EV;

  // Left side Eigen matrix.
  A_EM = Eigen::MatrixXd::Zero(probSize,probSize);
  // Right side Eigen Vector.
  B_EV = Eigen::VectorXd::Zero(probSize);
  // Eigen vector containing the solution.
  u_EV = Eigen::VectorXd::Zero(probSize);

  // Loop over cells.
  for (int i=0; i<nProbs; i++) {

    int k = i*probSize;

    // Fill the left side matrix according to the weak product of A and u.
    A_EM = Eigen::MatrixXd::Zero(probSize,probSize);
    A_EM(0,0) = 0.7071067811865475*lhsA[k+0];
    A_EM(0,1) = 0.7071067811865475*lhsA[k+1];
    A_EM(0,2) = 0.7071067811865475*lhsA[k+2];
    A_EM(1,0) = 0.7071067811865475*lhsA[k+1];
    A_EM(1,1) = 0.6324555320336759*lhsA[k+2]+0.7071067811865475*lhsA[k+0];
    A_EM(1,2) = 0.6324555320336759*lhsA[k+1];
    A_EM(2,0) = 0.7071067811865475*lhsA[k+2];
    A_EM(2,1) = 0.6324555320336759*lhsA[k+1];
    A_EM(2,2) = 0.4517539514526256*lhsA[k+2]+0.7071067811865475*lhsA[k+0];

    // Assign the right side vector B.
    B_EV << rhsB[k+0],rhsB[k+1],rhsB[k+2];

    // Solve the system of equations.
    u_EV = A_EM.colPivHouseholderQr().solve(B_EV);


    // Extract solution from Eigen vector and place it in c++ array.
    Eigen::Map<VectorXd>(x+k,probSize,1) = u_EV;

  };

  solveStatus = 0;

  return solveStatus;
}
