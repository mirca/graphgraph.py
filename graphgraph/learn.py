import cvxpy as cp
import numpy as np


def learn_laplacian_cvxpy(samples, scale_input=True):
    if scale_input:
        S = np.corrcoef(samples.T)
    else:
        S = np.cov(samples.T)
    p = np.shape(S)[0]
    L = cp.Variable((p, p), PSD=True)
    J = (1 / p) * np.ones((p, p))
    objective = cp.Minimize(-cp.log_det(L + J) + cp.trace(L @ S))
    constraints = [cp.sum(L, axis=0) == 0, L - cp.diag(cp.diag(L)) <= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return L.value
