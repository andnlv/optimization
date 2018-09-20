import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from .utils import get_line_search_tool
from time import time
from collections import deque
from numpy.linalg import LinAlgError

def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

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
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    g_prev = matvec(x_0) - b
    d = -np.copy(g_prev)

    history = defaultdict(list) if trace else None
    if not max_iter:
        max_iter = x_0.size
    x_k = np.copy(x_0)
    time_begin = time()
    for i in range(max_iter+1):
        if display:
            print(i)
        if trace:
            history['time'].append(time() - time_begin)
            history['residual_norm'].append(np.linalg.norm(g_prev))
            if x_k.size < 3:
                history['x'].append(x_k)
        if np.linalg.norm(g_prev) <= tolerance * np.linalg.norm(b):
            return x_k, 'success', history

        Ad = matvec(d)
        a_k = np.linalg.norm(g_prev)**2 / (Ad @ d)
        x_k = x_k + a_k*d
        g_next = g_prev + a_k*Ad
        d = -g_next + (np.linalg.norm(g_next)/np.linalg.norm(g_prev))**2 * d
        if display:
            print('a_k', a_k)
            print('d', d)
            print('g_p', g_prev)
            print('x_k', x_k)
            print('Ad', Ad)
        g_prev = np.copy(g_next)
    return x_k, 'iterations_exceeded', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    def lbfgs_multiply(v, iter, g):
        try:
            s, y = iter.__next__()
            z =  lbfgs_multiply(v - (s @ v) / (y @ s) * y, iter, g)
            return  z + (s @ v - y @ z) / (y @ s) * s
        except StopIteration:
            return g*v

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    time_begin = time()
    g_prev = oracle.grad(x_k)
    tolerance = tolerance * np.linalg.norm(g_prev) ** 2
    memory = deque()
    gamma = 1
    for i in range(max_iter + 1):
        if display:
            print(i)
        if trace:
            history['time'].append(time() - time_begin)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(g_prev))
            if x_k.size < 3:
                history['x'].append(x_k)
        if np.linalg.norm(g_prev) ** 2 <= tolerance:
            return x_k, 'success', history

        d = lbfgs_multiply(-g_prev, memory.__iter__(), gamma)
        a = line_search_tool.line_search(oracle, x_k, d)
        s = a*d
        x_k = x_k + s
        g_next = oracle.grad(x_k)
        y = g_next - g_prev
        g_prev = g_next

        memory.appendleft((s, y))
        if len(memory) > memory_size:
            memory.pop()
        gamma = (y @ s) / (y @ y)

    return x_k, 'iterations_exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    time_begin = time()
    tolerance = tolerance * np.linalg.norm(oracle.grad(x_k)) ** 2
    for i in range(max_iter + 1):
        g_k = oracle.grad(x_k)
        if display:
            print(i)
        if trace:
            history['time'].append(time() - time_begin)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(g_k))
            if x_k.size < 3:
                history['x'].append(x_k)
        if display:
            print(np.linalg.norm(g_k) ** 2, 'vs', tolerance)
        if np.linalg.norm(g_k) ** 2 <= tolerance:
            return x_k, 'success', history
        mu = min(0.5, np.sqrt(np.linalg.norm(g_k)))
        d_k = -g_k
        while True:
            d_k, msg, _ = conjugate_gradients(lambda v: oracle.hess_vec(x_k, v), b=d_k, x_0=d_k, tolerance=mu)
            if display and msg == 'iterations_exceeded':
                print('cg_msg = fail')
            if d_k @ g_k < 0:
                break
            mu = mu / 10

        a_k = line_search_tool.line_search(oracle, x_k, d_k)
        x_k = x_k + a_k * d_k
    return x_k, 'iterations_exceeded', history


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
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
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    a_k = None

    # Use line_search_tool.line_search() for adaptive step size.
    time_begin = time()
    tolerance = tolerance * np.linalg.norm(oracle.grad(x_k)) ** 2
    try:
        for i in range(max_iter+1):
            grad = oracle.grad(x_k)
            if display:
                print(i)
            if trace:
                history['time'].append(time() - time_begin)
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(np.linalg.norm(grad))
                if x_k.size < 3:
                    history['x'].append(x_k)
            if np.linalg.norm(grad) ** 2 <= tolerance:
                return x_k, 'success', history
            a_k = line_search_tool.line_search(oracle, x_k, -grad, a_k)
            x_k = x_k - a_k * grad
    except LinAlgError:
        return x_k, 'computational_error', history
    return x_k, 'iterations_exceeded', history
