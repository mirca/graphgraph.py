import numpy as np
from scipy import linalg
from tqdm import tqdm

from .operators import *


class NormalizedLaplacian(object):
    """
    Estimate a normalized Laplacian matrix given samples.
    """

    def __init__(
        self,
        samples,
        scale_input=True,
        k=1,
        rho=1,
        mu=1,
        tau=1,
        eta=0,
        maxiter=1000,
        tol=1e-4,
    ) -> None:
        if scale_input:
            self.S = np.corrcoef(samples.T)
        else:
            self.S = np.cov(samples.T)
        self.rho = rho
        self.mu = mu
        self.tau = tau
        self.k = k
        self.eta = eta
        self.maxiter = maxiter
        self.tol = tol
        self.p = np.shape(self.S)[0]
        self.primal_residual = []
        self.dual_residual = []
        self.aug_lagrangian = []

    def learn(self):
        # initialize constants
        I = np.eye(self.p)
        # initialize dual variables
        Y = np.zeros((self.p, self.p))
        z = np.zeros(self.p)
        # initialize primal variables
        Sinv = np.linalg.pinv(self.S)
        w = np.maximum(inv_laplacian_op(Sinv), 0)
        Theta = normalized_laplacian_op(w)
        Theta_copy = np.copy(Theta)
        Psi = np.eye(self.p)
        # Psi = np.diag(1 / np.sqrt(degree_op(w)))
        _, V = np.linalg.eigh(laplacian_op(w))
        V = V[:, : self.k]
        self.aug_lagrangian.append(
            self.get_augmented_lagrangian(
                w, Theta, Psi, V, self.S, Y, z, self.rho, self.eta
            )
        )
        # admm loop
        for i in tqdm(range(self.maxiter)):
            # update primal variables
            subproblem_Theta = SubproblemTheta(Theta, Psi, w, Y, self.rho, self.k)
            Theta = subproblem_Theta.optimize()
            subproblem_weights = SubproblemGraphWeights(
                w, Theta, Psi, V, self.S, Y, z, self.rho, self.eta
            )
            w = subproblem_weights.optimize()
            subproblem_V = SubproblemV(w, self.k)
            V = subproblem_V.optimize()
            subproblem_Psi = SubproblemPsi(Psi, Theta, w, self.S, Y, z, self.rho)
            Psi = subproblem_Psi.optimize()
            # update dual variables
            primal_residual = Theta - (I - Psi @ adjacency_op(w) @ Psi)
            dual_residual = Theta - Theta_copy
            Y = Y + self.rho * primal_residual
            diag_Psi = np.diagonal(Psi)
            z = z + self.rho * (1 / (diag_Psi * diag_Psi) - degree_op(w))
            # save augmented lagrangian
            self.aug_lagrangian.append(
                self.get_augmented_lagrangian(
                    w, Theta, Psi, V, self.S, Y, z, self.rho, self.eta
                )
            )
            # update rho
            s = self.rho * np.linalg.norm(dual_residual)
            self.dual_residual.append(s)
            r = np.linalg.norm(primal_residual)
            self.primal_residual.append(r)
            if r > self.mu * s:
                self.rho *= self.tau
            elif s > self.mu * r:
                self.rho /= self.tau
            # update eta
            # n_zero_eigenvalues = np.sum(np.linalg.eigvalsh(laplacian_op(w)) < 1e-9)
            # if self.k < n_zero_eigenvalues:
            #    self.eta *= 0.5
            # elif self.k > n_zero_eigenvalues:
            #    self.eta *= 2.0
            # else:
            #    break
            # check convergence
            has_converged = (s < self.tol) and (r < self.tol) and (i > 1)
            if has_converged:
                break
            Theta_copy = Theta
        self.Y = Y
        self.Psi = Psi
        self.Theta = Theta
        self.z = z
        self.w = w
        self.normalized_laplacian = normalized_laplacian_op(self.w)

    def get_augmented_lagrangian(self, w, Theta, Psi, V, S, Y, z, rho, eta):
        Aw = adjacency_op(w)
        Lw = laplacian_op(w)
        dw = degree_op(w)
        I = np.eye(self.p)
        eigvals = linalg.eigvalsh(Theta)
        PsiAwPsi = Psi @ Aw @ Psi
        diag_Psi = np.diagonal(Psi)
        term1 = -np.sum(PsiAwPsi * S)
        term2 = -np.log(np.sum(eigvals[self.k :]))
        Tmp = Theta - I + PsiAwPsi
        term3 = np.linalg.norm(Tmp)
        term3 = 0.5 * rho * term3 * term3
        tmp = 1 / (diag_Psi * diag_Psi) - dw
        term4 = np.linalg.norm(tmp)
        term4 = 0.5 * term4 * term4
        term5 = np.sum(Tmp * Y)
        term6 = np.sum(tmp * z)
        term7 = eta * np.sum(Lw * (V @ V.T))
        return term1 + term2 + term3 + term4 + term5 + term6 + term7


class SubproblemGraphWeights(object):
    def __init__(self, w, Theta, Psi, V, S, Y, z, rho, eta, maxiter=1) -> None:
        self.w = w
        self.Theta = Theta
        self.Psi = Psi
        self.V = V
        self.Y = Y
        self.S = S
        self.z = z
        self.rho = rho
        self.eta = eta
        self.p = np.shape(self.Theta)[0]
        self.I = np.eye(self.p)
        self.lr = 1e-4
        self.maxiter = maxiter

    def get_objective_function(self, w) -> float:
        diag_Psi = np.diagonal(self.Psi)
        Aw = adjacency_op(w)
        Lw = laplacian_op(w)
        dw = degree_op(w)
        term1 = np.linalg.norm(self.Theta - self.I + self.Psi @ Aw @ self.Psi)
        term1 = 0.5 * self.rho * term1 * term1
        term2 = np.linalg.norm(dw - 1 / (diag_Psi * diag_Psi))
        term2 = 0.5 * self.rho * term2 * term2
        term3 = np.sum((self.Psi @ Aw @ self.Psi) * (self.Y - self.S))
        term4 = -np.sum(dw * self.z)
        term5 = self.eta * np.sum(Lw * (self.V @ self.V.T))
        return term1 + term2 + term3 + term4 + term5

    def get_gradient(self, w):
        diag_Psi = np.diagonal(self.Psi)
        Aw = adjacency_op(w)
        dw = degree_op(w)
        term1 = 2.0 * adjacency_op_T(
            self.Psi @ self.Psi @ Aw @ self.Psi @ self.Psi
            + self.Psi @ (self.Theta - self.I) @ self.Psi
        )
        term2 = 2.0 * degree_op_T(dw - 1 / (diag_Psi * diag_Psi))
        term3 = adjacency_op_T(self.Psi @ (self.Y - self.S) @ self.Psi)
        term4 = -degree_op_T(self.z)
        term5 = self.eta * laplacian_op_T(self.V @ self.V.T)
        return term1 + term2 + term3 + term4 + term5

    def optimize(self):
        # obj0 = self.get_objective_function(self.w)
        for i in range(self.maxiter):
            # w_copy = np.copy(self.w)
            delta_w = self.get_gradient(self.w)
            self.w = np.maximum(self.w - self.lr * delta_w, 0)
            # print(obj0 - self.get_objective_function(self.w))
            # while True:
            #    w_update = np.maximum(self.w - lr * delta_w, 0)
            #    if (
            #        np.abs(w_update - self.w) <= 0.5 * 1e-4 * (w_update + self.w)
            #    ).all():
            #        break
            #    obj_update = self.get_objective_function(w_update)
            #    if obj_update < (
            #        obj0
            #        + np.sum(delta_w * (w_update - self.w))
            #        + 0.5 * (1.0 / lr) * np.linalg.norm(w_update - self.w) ** 2.0
            #    ):
            #        self.w = w_update
            #        lr = 2.0 * lr
            #        obj0 = obj_update
            #        break
            #    else:
            #        lr = 0.5 * lr
            # if (np.abs(w_copy - self.w) <= 0.5 * 1e-4 * (w_copy + self.w)).all():
            #    break
        return self.w


class SubproblemTheta(object):
    def __init__(self, Theta, Psi, w, Y, rho, k) -> None:
        self.Theta = Theta
        self.Psi = Psi
        self.w = w
        self.Y = Y
        self.rho = rho
        self.k = k
        self.Aw = adjacency_op(w)
        self.p = np.shape(self.Theta)[0]
        self.I = np.eye(self.p)

    def get_objective_function(self) -> float:
        Tmp = self.Theta - self.I + self.Psi @ self.Aw @ self.Psi
        eigvals = linalg.eigvalsh(self.Theta)
        term1 = -np.log(np.sum(eigvals[self.k :]))
        term2 = np.linalg.norm(Tmp)
        term2 = 0.5 * self.rho * term2 * term2
        term3 = np.sum(self.Theta * self.Y)
        return term1 + term2 + term3

    def optimize(self):
        gamma, U = np.linalg.eigh(
            self.rho * (self.I - self.Psi @ self.Aw @ self.Psi) - self.Y
        )
        gamma = gamma[self.k :]
        U = U[:, self.k :]
        Gamma = np.diag(gamma + np.sqrt(gamma * gamma + 4 * self.rho))
        self.Theta = (0.5 / self.rho) * U @ Gamma @ U.T
        return self.Theta


class SubproblemV(object):
    def __init__(self, w, k) -> None:
        self.w = w
        self.k = k

    def optimize(self):
        Lw = laplacian_op(self.w)
        _, V = np.linalg.eigh(Lw)
        V = V[:, : self.k]
        return V


class SubproblemPsi(object):
    def __init__(self, Psi, Theta, w, S, Y, z, rho, maxiter=1) -> None:
        self.Psi = Psi
        self.Theta = Theta
        self.w = w
        self.Y = Y
        self.S = S
        self.z = z
        self.rho = rho
        self.Aw = adjacency_op(w)
        self.dw = degree_op(w)
        self.p = np.shape(self.Theta)[0]
        self.I = np.eye(self.p)
        self.maxiter = maxiter
        self.lr = 1.0

    def get_objective_function(self, Psi) -> float:
        diag_Psi = np.diagonal(Psi)
        term1 = np.linalg.norm(self.Theta - np.eye(self.p) + Psi @ self.Aw @ Psi)
        term1 = 0.5 * self.rho * term1 * term1
        term2 = np.linalg.norm(1 / (diag_Psi**2) - self.dw)
        term2 = 0.5 * self.rho * term2 * term2
        term3 = np.sum((Psi @ self.Aw @ Psi) * (self.Y - self.S))
        term4 = np.sum((1 / (diag_Psi**2)) * self.z)
        return term1 + term2 + term3 + term4

    def get_gradient(self, Psi):
        diag_Psi = np.diagonal(Psi)
        AwPsiPsiAw = self.Aw @ Psi @ Psi @ self.Aw
        term1 = self.rho * Psi @ AwPsiPsiAw + AwPsiPsiAw @ Psi
        term2 = 2.0 * self.rho * np.diag(self.dw * diag_Psi ** (-3) - diag_Psi ** (-5))
        Tmp = self.Y - self.S + self.rho * (self.Theta - self.I)
        term3 = Tmp @ Psi @ self.Aw
        term4 = self.Aw @ Psi @ Tmp
        term5 = -2 * np.diag((diag_Psi ** (-3)) * self.z)
        return term1 + term2 + term3 + term4 + term5

    def optimize(self):
        return np.eye(self.p)
        # obj0 = self.get_objective_function(self.Psi)
        # for i in range(self.maxiter):
        #    Psi_copy = np.copy(self.Psi)
        #    delta_Psi = self.get_gradient(self.Psi)
        #    while True:
        #        Psi_update = np.diag(np.diagonal(self.Psi - self.lr * delta_Psi))
        #        if (
        #            np.linalg.norm(Psi_update - self.Psi) / np.linalg.norm(self.Psi)
        #            < 1e-4
        #        ):
        #            break
        #        obj_update = self.get_objective_function(Psi_update)
        #        if obj_update < (
        #            obj0
        #            + np.sum(delta_Psi * (Psi_update - self.Psi))
        #            + 0.5
        #            * (1.0 / self.lr)
        #            * np.linalg.norm(Psi_update - self.Psi) ** 2.0
        #        ):
        #            if np.all(np.diagonal(Psi_update) > 1e-6):
        #                self.Psi = Psi_update
        #                self.lr = 2.0 * self.lr
        #                obj0 = obj_update
        #                break
        #        else:
        #            self.lr = 0.5 * self.lr
        #    if np.linalg.norm(Psi_copy - self.Psi) / np.linalg.norm(Psi_copy) < 1e-4:
        #        break
        # return Psi_update
