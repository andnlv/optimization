import numpy as np
import scipy
from scipy.special import expit

class BaseSmoothOracle(object):
    """
    Base class for smooth function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        raise NotImplementedError('Grad is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class BaseProxOracle(object):
    """
    Base class for proximal h(x)-part in a composite function f(x) + h(x).
    """

    def func(self, x):
        """
        Computes the value of h(x).
        """
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + h(y) }.
        """
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle(object):
    """
    Base class for the composite function.
    phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part.
    """

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        """
        Computes the f(x) + h(x).
        """
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        """
        Computes the gradient of f(x).
        """
        return self._f.grad(x)

    def prox(self, x, alpha):
        """
        Computes the proximal mapping.
        """
        return self._h.prox(x, alpha)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class LeastSquaresOracle(BaseSmoothOracle):
    """
    Oracle for least-squares regression.
        f(x) = 0.5 ||Ax - b||_2^2
    """
    def __init__(self, matvec_Ax, matvec_ATx, b):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b

    def func(self, x):
        return np.linalg.norm(self.matvec_Ax(x) - self.b)**2 / 2

    def grad(self, x):
        return self.matvec_ATx(self.matvec_Ax(x) - self.b)

class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer.
        h(x) = regcoef * ||x||_1.
    """

    def __init__(self, regcoef):
        self.regcoef = regcoef

    def func(self, x):
        return self.regcoef*np.linalg.norm(x, 1)

    def prox(self, x, alpha):
        return np.sign(x)*np.max([np.abs(x) - self.regcoef*alpha, np.zeros(x.size)], axis=0)


class LassoProxOracle(BaseCompositeOracle):
    """
    Oracle for 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        f(x) = 0.5 * ||Ax - b||_2^2 is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """

    def duality_gap(self, x):
        Ax_b = self._f.matvec_Ax(x) - self._f.b
        mu = min(1, self._h.regcoef/np.linalg.norm(self._f.matvec_ATx(Ax_b), np.inf)) * Ax_b
        return self.func(x) + np.linalg.norm(mu)**2 /2 + self._f.b @ mu


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    mu = min(1, regcoef/np.linalg.norm(ATAx_b, np.inf))* Ax_b
    return np.linalg.norm(Ax_b)**2 / 2 + regcoef * np.linalg.norm(x, 1) + np.linalg.norm(mu)**2 /2 + b @ mu
    

def create_lasso_prox_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)
    return LassoProxOracle(LeastSquaresOracle(matvec_Ax, matvec_ATx, b),
                           L1RegOracle(regcoef))


class LassoDualOracle(BaseSmoothOracle):
    def __init__(self, A, b, regcoef, t0):
        self.A = A
        self.b = b
        self.ATA = A.T @ A
        self.ATb = A.T @ b
        self.regcoef = regcoef
        self.t = t0

    def increase_t(self, gamma):
        self.t *= gamma

    def _func(self, x, u):
        res = self.t*(np.linalg.norm(self.A @ x - self.b) ** 2 / 2 + self.regcoef * u.sum()) - np.log(u + x).sum() - np.log(u - x).sum()
        return res

    def _grad(self, x, u):
        z = u**2 - x**2
        return np.concatenate((self.t*(self.ATA @ x - self.ATb) + 2 * x / z,
                              self.regcoef*self.t - 2 * u / z))

    def _hess(self, x, u):
        z = (u**2 - x**2)**2
        p = 2 * (u**2 + x**2) / z
        q = - 4 * u * x / z
        return np.block([[self.t * self.ATA + np.diag(p), np.diag(q)], [np.diag(q), np.diag(p)]])

    def func(self, x):
        return self._func(x[:x.size // 2], x[x.size // 2:])

    def grad(self, x):
        return self._grad(x[:x.size // 2], x[x.size // 2:])

    def hess(self, x):
        return self._hess(x[:x.size // 2], x[x.size // 2:])
