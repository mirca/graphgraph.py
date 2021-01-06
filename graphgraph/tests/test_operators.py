import pytest
import numpy as np
import inspect
import random
from graphgraph import operators as op


class LaplacianConstraintError(RuntimeError):
    pass

class LaplacianConstraints:

    def __init__(self, in_matrix):
        self.in_matrix = in_matrix
        self.validate()

    def is_symmetric(self):
        if not np.all(self.in_matrix == self.in_matrix.T):
            raise LaplacianConstraintError("input matrix is not symmetric")

    def is_diagonal_nonnegative(self):
        if not np.all(np.diag(self.in_matrix) >= 0):
            raise LaplacianConstraintError("diagonal elements are not all nonnegative")

    def is_off_diagonal_nonpositive(self):
        if not np.all(self.in_matrix[np.triu_indices(n=self.in_matrix.shape[0], k=1)] <= 0):
            raise LaplacianConstraintError("off-diagonal elements are not all nonpositive")

    def is_row_sum_zero(self):
        if not np.all(np.abs(np.sum(self.in_matrix, axis=0)) < 1e-9):
            raise LaplacianConstraintError("row sum is not identically zero")

    def validate(self):
        constraints = [m[1] for m in inspect.getmembers(LaplacianConstraints,
                                                        predicate=inspect.isfunction)
                            if m[0].startswith("is")]
        for c in constraints:
            c(self)

def test_laplacian_operator():
    weights = np.random.uniform(size=6)
    LaplacianConstraints(op.laplacian_op(weights)).validate()

@pytest.mark.parametrize("inv_op_name, op_name",
                         [(op.inv_laplacian_op, op.laplacian_op),
                          (op.inv_adjacency_op, op.adjacency_op)])
def test_inverse_operators(inv_op_name, op_name):
    weights = np.random.uniform(size=6)
    weights_ = inv_op_name(op_name(weights))
    np.testing.assert_allclose(weights, weights_)

@pytest.mark.parametrize("adj_op_name, op_name",
                         [(op.adj_laplacian_op, op.laplacian_op),
                          (op.adj_adjacency_op, op.adjacency_op)])
def test_adjoint_operators(adj_op_name, op_name):
    "test the inner product equality between the linear operator and its adjoint"
    # number of nodes
    p = random.randint(a = 3, b = 100)
    # number of edges
    m = int(.5 * p * (p - 1))
    X = np.random.uniform(size=p*p).reshape((p, p))
    weights = np.random.uniform(size=m)
    np.testing.assert_almost_equal(np.sum(X * op_name(weights)),
                                   np.sum(weights * adj_op_name(X)))