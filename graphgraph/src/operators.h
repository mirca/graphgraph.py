#ifndef OPERATORS_H
#define OPERATORS_H

#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

typedef const Eigen::Matrix<double, Eigen::Dynamic, 1> c_vector_t;
typedef const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> c_matrix_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

/* Computes the Laplacian linear operator which maps a vector of weights into
 a valid Laplacian matrix.

 param: weights weight vector of the graph
 return: Lw the Laplacian matrix
*/
matrix_t laplacian_op(c_vector_t& weights) {
  int j;
  int k = weights.size();
  const int n = .5 * (1 + sqrt(1. + 8. * k));
  matrix_t Lw = matrix_t::Zero(n, n);

  for (int i = n-2; i > -1; --i) {
    j = n - i - 1;
    Lw.row(i).tail(j) = -weights.head(k).tail(j);
    k -= j;
  }

  vector_t LwColSum = (Lw + Lw.transpose()).colwise().sum();
  Lw.diagonal() -= LwColSum;
  return Lw.selfadjointView<Upper>();
}

/* Computes the Adjacency linear operator which maps a vector of weights into
  a valid Adjacency matrix.

  param: weights weight vector of the graph
  return: Aw the Adjacency matrix
*/
matrix_t adjacency_op(c_vector_t& weights) {
  int j;
  int k = weights.size();
  const int n = .5 * (1 + sqrt(1. + 8. * k));
  matrix_t Aw = matrix_t::Zero(n, n);

  for (int i = n-2; i > -1; --i) {
    j = n - i - 1;
    Aw.row(i).tail(j) = weights.head(k).tail(j);
    k -= j;
  }
  return Aw.selfadjointView<Upper>();
}


/* Computes the adjoint of the Laplacian operator.
/
/ param: in_matrix input matrix
/ return: out_vector output vector
*/
vector_t laplacian_op_T(c_matrix_t& in_matrix) {
  int ncols = in_matrix.cols();
  int k = .5 * ncols * (ncols - 1);
  int j = 0;
  int l = 1;
  vector_t out_vector(k);

  for (int i = 0; i < k; ++i) {
    out_vector(i) = in_matrix(j, j) + in_matrix(l, l) - (in_matrix(l, j) + in_matrix(j, l));
    if (l == (ncols - 1)) {
      l = (++j) + 1;
    } else {
      ++l;
    }
  }
  return out_vector;
}


/* Computes the adjoint of the adjacency operator.

  param: in_matrix input matrix
  return: out_vector output vector
*/
vector_t adjacency_op_T(c_matrix_t& in_matrix) {
  int ncols = in_matrix.cols();
  int k = .5 * ncols * (ncols - 1);
  int j = 0;
  int l = 1;
  vector_t out_vector(k);

  for (int i = 0; i < k; ++i) {
    out_vector(i) = in_matrix(l, j) + in_matrix(j, l);
    if (l == (ncols - 1)) {
      l = (++j) + 1;
    } else {
      ++l;
    }
  }
  return out_vector;
}

/* Computes the inverse of the A operator.
/
/ param in_matrix input Adjacency matrix
/ return weights the weight vector of the graph
*/
vector_t inv_adjacency_op(c_matrix_t& in_matrix) {
  int ncols = in_matrix.cols();
  int k = .5 * ncols * (ncols - 1);
  int l = 0;
  vector_t weights(k);

  for (int i = 0; i < ncols-1; ++i) {
    for (int j = i+1; j < ncols; ++j) {
      weights(l) = in_matrix(i, j);
      ++l;
    }
  }
  return weights;
}


/* Computes the inverse of the L operator.
/
/ param: in_matrix input Laplacian matrix
/ return: out_vector the weight vector of the graph
*/
vector_t inv_laplacian_op(c_matrix_t& in_matrix) {
  return -inv_adjacency_op(in_matrix);
}


/* Computes the degree operator from the vector of edge weights.
/
/ param: weights vector of graph weights
/ return: out_vector degree vector
*/
vector_t degree_op(c_vector_t& weights) {
  return adjacency_op(weights).colwise().sum();
}


/* Computes the adjoint of the degree operator, i.e., the adjoint of the D operator.
/
/ param: in_vector input vector
/ return: out_vector adjoint of the degree operator
*/
vector_t degree_op_T(c_vector_t in_vector) {
  return laplacian_op_T(in_vector.asDiagonal());
}


matrix_t normalized_laplacian_op(c_vector_t& weights) {
  matrix_t Aw = adjacency_op(weights);
  vector_t dw = degree_op(weights);
  vector_t invsqrt_dw = (1.0 / dw.array().sqrt()).matrix();
  matrix_t Tmp = Aw.array().colwise() * invsqrt_dw.array();
  Tmp = Tmp.array().rowwise() * invsqrt_dw.transpose().array();
  Tmp.diagonal().array() -= 1.0;
  return (0.0 - Tmp.array()).matrix();
}

#endif
