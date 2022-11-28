import numpy as np
import cvxpy as cp

from typing import Callable

f_type = Callable[[np.array], float]


def f(v, Q, p):
    """The original objective function f_0."""
    return v.T @ Q @ v + p.T @ v

def grad_f(v, Q, p):
    """The gradient of f."""
    return 2 * Q @ v + p


def hess_f(v, Q, p):
    """The hessian of f."""
    return 2 * Q


def g(v, A, b, i):
    """The log affine constraints f_i."""
    ai = A[i, :]
    bi = b[i]
    return float(np.nan_to_num(-np.log(bi - ai.T @ v), nan=np.inf))

def grad_g(v, A, b, i):
    """The gradient of the log affine constraints f_i."""
    ai = A[i, :]
    bi = b[i]
    return ai / (bi - ai.T @ v)

def hess_g(v, A, b, i):
    """The hessian of the log affine constraints f_i."""
    ai = A[i, :]
    bi = b[i]
    return ai.reshape(-1, 1) @ ai.reshape(-1, 1).T / (bi - ai.T @ v) ** 2


def phi(v, A, b):
    """The log barrier function phi."""
    m = len(b)
    return np.sum([g(v, A, b, i) for i in range(m)])


def grad_phi(v, A, b):
    """The gradient of the log barrier function."""
    m = len(b)
    return np.sum(np.array([grad_g(v, A, b, i) for i in range(m)]), axis=0)


def hess_phi(v, A, b):
    """The hessian of the log barrier function."""
    m = len(b)
    return np.sum(np.array([hess_g(v, A, b, i) for i in range(m)]), axis=0)

def dual_func(v, y):
    """The dual function (the inf of the Lagragian)."""
    return - 0.5 * v.T @ v - v.T @ y

def lasso_func(w, X, y, ld):
    """The lasso function."""
    return 0.5 * np.linalg.norm(X @ w - y)**2 + ld * np.linalg.norm(w, 1)

def get_w_star(v_star, X, y):
    return np.linalg.lstsq(X, y + v_star, rcond=None)[0]

def newton_step(x: np.array, grad_f: f_type, hess_f: f_type) -> np.array:
    step = -np.linalg.inv(hess_f(x)) @ grad_f(x)
    return step

def newton_decrement(x: np.array, grad_f: f_type, hess_f: f_type) -> float:
    decrement = np.sqrt(grad_f(x).transpose() @ np.linalg.inv(hess_f(x)) @ grad_f(x))
    return decrement


def backtracking_line_search(
    x: np.array, f: f_type, grad_f: f_type, delta_x: np.array,
    alpha: float, beta: float, verbose: bool = False
) -> float:
    assert alpha > 0 and alpha < 1 / 2, "Alpha must be between 0 and 0.5"
    assert beta > 0 and beta < 1, "Beta must be between 0 and 1"

    t = 1
    _ = 0
    while f(x + t * delta_x) >= f(x) + alpha * t * grad_f(x) @ delta_x:
        t *= beta
        _ += 1
        if _ >= 100:
            raise ValueError("Backtracking not converging")
        
    return t

def newton_method(
    x0: np.array,
    f: f_type,
    grad_f: f_type,
    hess_f: f_type,
    epsilon: float,
    alpha: float = 0.1,
    beta: float = 0.5,
    verbose=True
):

    x = x0
    decrement = newton_decrement(x, grad_f, hess_f)

    xs = [x0]
    decrements = [decrement]
    steps = []
    ts = []

    _ = 0
    while decrement**2 / 2 > epsilon:
        try:
            if verbose:
                print(f"\tCriterion: {decrement**2:.4e}")
            decrement = newton_decrement(x, grad_f, hess_f)
            step = newton_step(x, grad_f, hess_f)
            t = backtracking_line_search(
                x, f, grad_f, step, alpha, beta, verbose=verbose
            )
            x += t * step

            decrements.append(decrement)
            steps.append(step)
            ts.append(t)
            xs.append(x)

            _ += 1
            if _ >= 20:
                raise ValueError("Newton method not converging")
            
        except ValueError as e:
            if verbose:
                print(e)
            break

    return x, f(x), {"xs": xs, "decrements": decrements, "steps": steps, "ts": ts}

def centering_step(Q, p, A, b, t, v0, eps, alpha=0.1, beta=0.5, verbose=True):
    x_opt, f_opt, iterates = newton_method(
        x0=v0,
        f=lambda v: t * f(v, Q, p) + phi(v, A, b),
        grad_f=lambda v: t * grad_f(v, Q, p) + grad_phi(v, A, b),
        hess_f=lambda v: t * hess_f(v, Q, p) + hess_phi(v, A, b),
        epsilon=eps,
        alpha=alpha,
        beta=beta,
        verbose=verbose
    )

    return x_opt, f_opt, iterates

def barr_method(Q, p, A, b, v0, eps, t0=1, mu=15, alpha=0.1, beta=0.5, verbose=True):
    t = t0
    m = len(b)
    v = v0
    vs = [v]
    ts = [t]
    l_x_opt = []
    l_f_opt = []
    l_iterates = []

    while m / t >= eps:
        if verbose:
            print(f"t = {t} and m/t = {m/t: .2e}")
        x_opt, f_opt, iterates = centering_step(
            Q, p, A, b, t, v, eps, alpha, beta, verbose=verbose
        )
        v = iterates["xs"][-1]
        t *= mu

        vs.append(v)
        ts.append(t)
        l_iterates.append(iterates)

    return vs, ts, l_iterates

def generate_data(N=20, D=50, ld=10, sigma=5, density=0.2):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(D)
    idxs = np.random.choice(range(D), int((1-density)*D), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    X = np.random.randn(N,D)
    y = X.dot(beta_star) + np.random.normal(0, sigma, size=N)
    Q = 0.5*np.identity(N)
    p = y
    A = np.vstack((X.T, -X.T))
    b = ld * np.ones(2*D)
    return X, y, beta_star, Q, p, A, b



# =================================================================
if __name__ == "__main__":
    # Generate the data
    N = 20
    D = 50
    M = 2*D
    ld = 10

    X, y, beta_star, Q, p, A, b = generate_data(
        N=N,
        D=D,
        ld=ld,
        sigma=1,
        density=0.2
    )

    v0 = np.zeros(N)
    t0 = 1
    
    # Optimise with the barrier method
    vs, ts, l_iterates = barr_method(
        Q, p, A, b, v0, eps=1e-8, t0=t0, mu=20, alpha=0.4, beta=0.5, verbose=False
    )
    
    v_star = vs[-1]
    
    print(f"Optimal point: {v_star}")
    print(f"Value of the function at that point: {f(v_star, Q, p): .6g}")
    
    print(f"Check if the constraints are respected: {(v_star @ A.T - b).max(): .6g}")
    

    # Posing the dual problem
    x = cp.Variable(N)
    dual_prob = cp.Problem(
        cp.Minimize(
            cp.quad_form(x, Q) + p.T @ x
            ),
        [A @ x <= b]
    )

    # Solving the problem
    dual_prob.solve()

    # Printing the optimal solution
    print(f"Min value of the problem with CVXPY: {dual_prob.value:.6g}")
    print(f"Optimal point: {x.value}")

    
    # Difference between custom implementation and CVXPy
    print(f"Difference between custom implementation and CVXPy: {f(v_star, Q, p) - dual_prob.value:.6g}")
    
    # Difference between custom implementation and CVXPy
    print(f"Difference between custom implementation and CVXPy: {np.linalg.norm(v_star - x.value):.6g}")
    
    
    # Pose the LASSO problem
    w = cp.Variable(D)
    prob = cp.Problem(
        cp.Minimize(
            cp.norm2(X @ w - y)**2 + ld * cp.norm1(w)
        )
    )

    # Solve the problem
    prob.solve()

    # Optimal solution
    print(f"Min value of the problem with CVXPY: {prob.value:.6g}")
    print(f"Optimal point: {w.value}")