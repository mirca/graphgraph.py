import pytest
import numpy as np
import inspect
import random
from graphgraph import operators as op
from graphgraph import operators_py as op_py


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
        if not np.all(
            self.in_matrix[np.triu_indices(n=self.in_matrix.shape[0], k=1)] <= 0
        ):
            raise LaplacianConstraintError(
                "off-diagonal elements are not all nonpositive"
            )

    def is_row_sum_zero(self):
        if not np.all(np.abs(np.sum(self.in_matrix, axis=0)) < 1e-9):
            raise LaplacianConstraintError("row sum is not identically zero")

    def validate(self):
        constraints = [
            m[1]
            for m in inspect.getmembers(
                LaplacianConstraints, predicate=inspect.isfunction
            )
            if m[0].startswith("is")
        ]
        for c in constraints:
            c(self)


def test_laplacian_operator():
    p = random.randint(a=3, b=100)
    weights = np.random.uniform(size=int(0.5 * p * (p - 1)))
    LaplacianConstraints(op.laplacian_op(weights)).validate()


def test_normalized_laplacian_operator():
    p = random.randint(a=3, b=100)
    weights = np.random.uniform(size=int(0.5 * p * (p - 1)))
    L_normalized = op.normalized_laplacian_op(weights)
    L = op.laplacian_op(weights)
    D_ = np.diag(1 / np.sqrt(np.diagonal(L)))
    np.testing.assert_array_almost_equal(L_normalized, D_ @ L @ D_)


@pytest.mark.parametrize(
    "inv_op_name, op_name",
    [(op.inv_laplacian_op, op.laplacian_op), (op.inv_adjacency_op, op.adjacency_op)],
)
def test_inverse_operators(inv_op_name, op_name):
    weights = np.random.uniform(size=6)
    weights_ = inv_op_name(op_name(weights))
    np.testing.assert_allclose(weights, weights_)


@pytest.mark.parametrize(
    "op_name_T, op_name",
    [(op.laplacian_op_T, op.laplacian_op), (op.adjacency_op_T, op.adjacency_op)],
)
def test_adjoint_operators(op_name_T, op_name):
    "test the inner product equality between the linear operator and its adjoint"
    p = random.randint(a=3, b=100)
    X = np.random.uniform(size=p * p).reshape((p, p))
    weights = np.random.uniform(size=int(0.5 * p * (p - 1)))
    np.testing.assert_almost_equal(
        np.sum(X * op_name(weights)), np.sum(weights * op_name_T(X))
    )


def test_degree_adjoint_operator():
    p = random.randint(a=3, b=100)
    x = np.random.uniform(size=p)
    weights = np.random.uniform(size=int(0.5 * p * (p - 1)))
    np.testing.assert_almost_equal(
        np.sum(x * op.degree_op(weights)), np.sum(weights * op.degree_op_T(x))
    )


def test_degree_mat_adjoint_operator():
    p = random.randint(a=3, b=100)
    degree_mat = op_py.degree_op_mat(p)
    x = np.random.uniform(size=p)
    weights = np.random.uniform(size=int(0.5 * p * (p - 1)))
    np.testing.assert_almost_equal(
        np.sum(x * degree_mat @ (weights.T)), np.sum(weights * op.degree_op_T(x))
    )
