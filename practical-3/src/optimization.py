from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve, LinAlgError
import scipy
from time import time
import datetime
from .optimization1 import newton, LineSearchTool
from .oracles import LassoDualOracle
from .oracles import lasso_duality_gap as standard_gap

def proximal_gradient_method(oracle, x_0, L_0=1, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Gradient method for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k  = x_0
    time_begin = time()
    L = L_0

    for i in range(max_iter):
        duality_gap = oracle.duality_gap(x_k)

        if display:
            print(i)
        if trace:
            history['func'].append(oracle.func(x_k))
            history['time'].append(time() - time_begin)
            history['duality_gap'].append(duality_gap)
            if x_k.size < 3:
                history['x'].append(x_k)
        if duality_gap < tolerance:
            return x_k, 'success', history
        grad = oracle.grad(x_k)
        x_next = oracle.prox(x_k - grad / L, 1 / L)
        while oracle._f.func(x_next) > oracle._f.func(x_k) + oracle.grad(x_k) @ (x_next - x_k) + L * np.linalg.norm(x_next - x_k)**2 /2:
            L *= 2
            x_next = oracle.prox(x_k - grad / L, 1 / L)
        x_k = x_next
        L /= 2
    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append(time() - time_begin)
        history['duality_gap'].append(duality_gap)
        if x_k.size < 3:
            history['x'].append(x_k)
    return x_k, 'iterations_exceeded', history


def proximal_fast_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Fast gradient method for composite minimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(best_point) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
    """

    history = defaultdict(list) if trace else None
    x_k = x_0
    x_best = x_0
    func_best = oracle.func(x_best)
    time_begin = time()
    L = L_0
    A_k = 0
    v_k = x_k
    conv_sum_grad = np.zeros(x_k.size)
    for i in range(max_iter):
        duality_gap = oracle.duality_gap(x_best)

        if display:
            print(i)
        if trace:
            history['func'].append(func_best)
            history['time'].append(time() - time_begin)
            history['duality_gap'].append(duality_gap)
            if x_k.size < 3:
                history['x'].append(x_k)
        if duality_gap < tolerance:
            return x_best, 'success', history
        while True:
            a_k = (1 + np.sqrt(1 + 4*L*A_k))/(2*L)
            A_next = A_k + a_k
            y_k = (A_k*x_k + a_k*v_k)/A_next
            grad = oracle.grad(y_k)
            v_next = oracle.prox(x_0 - conv_sum_grad - a_k * grad, A_next)
            x_next = (A_k * x_k + a_k * v_next) / A_next
            if oracle.func(x_next) < func_best:
                x_best = x_next
                func_best = oracle.func(x_best)
            if oracle.func(y_k) < func_best:
                x_best = y_k
                func_best = oracle.func(x_best)
            if display:
                print('a_k', a_k)
                print('A_next', A_next)
                print('y_k', y_k)
                print('grad', grad)
                print('v_next', v_next)
                print('x_next', x_next)
            if oracle._f.func(x_next) <= oracle._f.func(y_k) + grad @ (x_next - y_k) + L * np.linalg.norm(x_next - y_k)**2 / 2:
                conv_sum_grad += a_k * grad
                x_k = x_next
                A_k = A_next
                v_k = v_next
                L /= 2
                break
            L *= 2
    if trace:
        history['func'].append(func_best)
        history['time'].append(time() - time_begin)
        history['duality_gap'].append(duality_gap)
        if x_k.size < 3:
            history['x'].append(x_k)
    return x_best, 'iterations_exceeded', history


class BarrierLineSearch(LineSearchTool):
    def __init__(self, Q, theta=0.99, **kwargs):
        self.theta = theta
        self.Q = scipy.sparse.csr_matrix(Q)
        super(BarrierLineSearch, self).__init__(method='Armijo', kwargs=kwargs)

    def line_search(self, oracle, x_k, d_k):
        p = self.Q @ d_k
        z = (self.Q @ x_k)/ p
        p_a = -self.theta * np.max(z[p > 0]) if (p > 0).any() else 1
        return super(BarrierLineSearch, self).line_search(oracle, x_k, d_k,
                                                         previous_alpha=min(p_a, 1))


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5, 
                         tolerance_inner=1e-8, max_iter=100, 
                         max_iter_inner=20, t_0=1, gamma=10, 
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary 
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    oracle = LassoDualOracle(A, b, reg_coef, t_0)
    history = defaultdict(list) if trace else None
    x_k = np.concatenate([x_0, u_0])
    time_begin = time()
    line_search_tool = BarrierLineSearch(np.block([[-np.eye(x_0.size), -np.eye(x_0.size)],
                                                  [np.eye(x_0.size), -np.eye(x_0.size)]]), c1=c1)
    if lasso_duality_gap is None:
        lasso_duality_gap = standard_gap

    def gap(x):
        Ax_b = A @ x - b
        return lasso_duality_gap(x, Ax_b, A.T @ Ax_b, b, reg_coef)
    for i in range(max_iter):
        duality_gap = gap(x_k[:x_0.size])
        if display:
            print(i)
        if trace:
            history['func'].append(np.linalg.norm(A @ x_k[:x_0.size] - b) **2 / 2)
            history['time'].append(time() - time_begin)
            history['duality_gap'].append(duality_gap)
            if x_k.size < 3:
                history['x'].append(x_k)
        if duality_gap < tolerance:
            return (x_k[:x_0.size], x_k[x_0.size:]), 'success', history
        if display:
            print(x_k)
        x_k, msg, _ = newton(oracle, x_k, tolerance=tolerance_inner,
                     max_iter=max_iter_inner, line_search_options=line_search_tool, display=display)
        if msg == 'computational_error':
            return (x_k[:x_0.size], x_k[x_0.size:]), 'computational_error', history
        oracle.increase_t(gamma)

    if trace:
        history['func'].append(np.linalg.norm(A @ x_k[:x_0.size] - b) ** 2 / 2)
        history['time'].append(time() - time_begin)
        history['duality_gap'].append(duality_gap)
        if x_k.size < 3:
            history['x'].append(x_k)
    return (x_k[:x_0.size], x_k[x_0.size:]), 'iterations_exceeded', history