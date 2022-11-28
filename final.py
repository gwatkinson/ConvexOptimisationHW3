#!/usr/bin/env python
# coding: utf-8

# # Convex Optimization - Homework 3
# 
# <font color='green'>
# 
# <i>Report plots, comments and theoretical results in a pdf file. Send your code together with the requested functions and a main script reproducing all your experiments. You can use Matlab, Python or Julia.</i>
# 
# Given $x_1,...,x_n \in \mathbb{R}^d$ data vectors and $y_1,...,y_n \in \mathbb{R}$ observations, we are searching for regression parameters $w \in \mathbb{R}^d$ which fit data inputs to observations $y$ by minimizing their squared difference. In a high dimensional setting (when $n \ll d$) a $\ell_1$ norm penalty is
# often used on the regression coefficients $w$ in order to enforce sparsity of the solution (so that $w$ will only have a few non-zeros entries). Such penalization has well known statistical properties, and makes the model both more interpretable, and faster at test time. 
# 
# From an optimization point of view we want to solve the following problem called LASSO (which stands for Least Absolute Shrinkage Operator and Selection Operator)
# 
# \begin{equation}
# \tag{LASSO}
# \begin{array}{ll}
# \text{minimize} & \frac{1}{2} \left\lVert Xw - y \right\rVert^2_2 + \lambda \left\lVert w \right\rVert_1
# \end{array}
# \end{equation}
# 
# in the variable $w \in \mathbb{R}^d$, where $X = (x^T_1, \ldots, x^T_n) \in \mathbb{R}^{n\times d},\, y = (y_1, \ldots, y_n) \in \mathbb{R}^n$ and $\lambda > 0$ is a regularization parameter.
# </font>

# ## Question 1

# <font color='navyblue'>
# Derive the dual problem of LASSO and format it as a general Quadratic Problem as follows
# 
# 
# \begin{equation}
# \tag{QP}
# \begin{array}{ll}
# \text{minimize} & v^TQv + p^Tv \\
# \text{subject to} & Av \preceq b
# \end{array}
# \end{equation}
# 
# in variable $v\in \mathbb{R}^n$, where $Q \succeq 0$.
# </font>

# We can repose the LASSO problem as the following problem
# 
# \begin{equation}
# \tag{LASSO'}
# \begin{array}{ll}
# \displaystyle \min_{w \in \mathbb{R}^d,\, z \in \mathbb{R}^n} & \frac{1}{2} \left\lVert z \right\rVert^2_2 + \lambda \left\lVert w \right\rVert_1 \\
# \text{subject to} & z = Xw - y 
# \end{array}
# \end{equation}
# 
# The associated Lagragian is
# 
# \begin{align*}
# \mathcal{L}(w, z, v) &= \frac{1}{2} z^T z + \lambda \left\lVert w \right\rVert_1 - v^T (y-Xw+z)
# \end{align*}
# 
# We can then calculate the dual function
# 
# \begin{align*}
# g(v) &= \inf_{w, z} \frac{1}{2} z^T z + \lambda \left\lVert w \right\rVert_1 - v^T (y-Xw+z) \\
# &= - v^T y + \inf_z \{ \frac{1}{2} z^T z - v^T z\} + \lambda \inf_w \{\left\lVert w \right\rVert_1 + \frac{1}{\lambda} v^T X w\} \\
# &= \left\{ \begin{array}{ll} - v^T y - \frac{1}{2} v^T v & \text{if } \left\lVert X^T v \right\lVert_{\infty} \le \lambda \\
# - \infty & \text{otherwise} \end{array}\right.
# \end{align*}
# 
# We can obtain the dual problem 
# 
# \begin{equation}
# \tag{LASSO*}
# \begin{array}{ll}
# \text{maximize} & - v^T y - \frac{1}{2} v^T v \\
# \text{subject to} & \left\lVert X^T v \right\lVert_{\infty} \le \lambda \\
# \end{array}
# \end{equation}
# 
# which can be simplified as follows
# 
# \begin{equation}
# \tag{LASSO*}
# \begin{array}{ll}
# \displaystyle \min_{v \in \mathbb{R}^n} & \frac{1}{2} v^T v  + y^T v \\
# \text{subject to} & - X^T v \le \lambda \cdot \mathbb{1}_d \\
# & X^T v \le \lambda \cdot \mathbb{1}_d
# \end{array}
# \end{equation}
# 
# Finally, we can rewrite it as 
# 
# \begin{equation}
# \tag{QP}
# \begin{array}{ll}
# \text{minimize} & v^TQv + p^Tv \\
# \text{subject to} & Av \preceq b
# \end{array}
# \end{equation}
# 
# 
# with
# 
# - $Q = \frac{1}{2} I_n \in \mathbb{R}^{n \times n}$, we have $Q \succeq 0$.
# - $p = y \in \mathbb{R}^{n}$
# - $b = \lambda \cdot \mathbb{1}_{2d} \in \mathbb{R}^{2d}$
# 
# and 
# \begin{equation*}
# A = \left(
#     \begin{array}{c}
#     X^T \\
#     -X^T
#     \end{array}
# \right)
# \in \mathbb{R}^{2d \times n}
# \end{equation*}
# 
# 
# For the next question, we pose $m=2d$.

# ## Question 2 and 3

# <font color='navyblue'>
# Impliment the barrier method to solve QP.
# 
# - Write a function `v_seq = centering_step(Q, p, A, b, t, v0, eps)` which impliments the Newton method to solve the centering step given the inputs $(Q, p, A, b)$, the barrier method parameter $t$ (see lectures), initial variable $v_0$ and a target precision $\epsilon$. The function outputs the sequence of variables iterates $(v_i)_{i=1, \ldots, n_{\epsilon}}$, where $n_{\epsilon}$ is the number of iterations to obtain the $\epsilon$ precision. Use a backtracking line search with appropriate parameters.
# - Write a function `v_seq = barr_method(Q, p, A, b, v0, eps)` which implements the barrier method to solve QP using precedent function given the data inputs $(Q, p, A, b)$, a feasible point $v_0$, a precision criterion $\epsilon$. The function ouptuts the sequence of variables iterates $(v_i)_{i=1, \ldots, n_{\epsilon}}$, where $n_{\epsilon}$ is the number of iterations to obtain the $\epsilon$ precision.
# - Test your function on randomly generated matrices $X$ and observations $y$ withz $\lambda = 10$. Plot precision criterion and gap $f(v_t) - f^*$ in semilog scale (using the best value found for $f$ as a surrogate for $f^*$). Repeat for different values of the barrier method parameter $\mu = 2, 15, 50, 100, \ldots$ and check the impact on $w$. What would be an appropriate choice for $\mu$ ?
# </font>

# We can write
# 
# $$
# A = \left(\begin{array}{ccc}
# & a_1^T & \\
# \hline
# & \vdots & \\
# \hline
# & a_m^T &
# \end{array}\right)
# $$
# with $a_i \in \mathbb{R}^n$.
# 
# The barrier problem can be written as
# 
# \begin{equation}
# \tag{Barrier}
# 
# \begin{array}{ll}
# \displaystyle \min_{v \in \mathbb{R}^n} & t (v^T Q v + p^T v) + \phi(v) \\
# \text{subject to} & \forall i \in [1, \cdots, m], \; a_i^T v - b_i \le 0
# \end{array}
# 
# \end{equation}
# 
# We pose $\forall i \in [1, \cdots, m], \; f_i(v) =  a_i^T v - b_i \in \mathbb{R}$. With this notation, $\phi(v) = - \displaystyle \sum_{i=1}^m \log(-f_i(v))$.
# 
# From there, we can look at
# 
# \begin{align*}
# \nabla \phi(v) &= \sum_{i=1}^m \frac{1}{- f_i(v)} \nabla f_i(v) \\
# &= \sum_{i=1}^m \frac{1}{b_i - a_i^T v} a_i
# \end{align*}
# 
# and
# 
# \begin{align*}
# \nabla^2 \phi(v) &= \sum_{i=1}^m \frac{1}{f_i(v)^2} \nabla f_i(v) \nabla f_i(v)^T \\
# &= \sum_{i=1}^m \frac{1}{(b_i - a_i^T v)^2} a_i a_i^T
# \end{align*}
# 
# since $\nabla^2 f_i(v) = 0$.
# 
# 
# Lastly, to get $w^*$ from the dual problem, we can use the fact that the Lagragian is minimized by the optimal point, the gradient of the Lagragian with respoect to $z$ vanishes. Which gives us:
# 
# \begin{align*}
#     \nabla_z \mathcal{L}(w^*, z^*, v^*) &= \nabla_z \left[ \frac{1}{2} (z^*)^T z^* + \lambda \left\lVert w^* \right\rVert_1 - (v^*)^T (y-Xw^*+z^*) \right] \\
#     &= 0 \\
#     &= z^* - v^* \\
#     &= X w^* - y - v^*
# \end{align*}
# 
# Given $v^*, X$ and $y$, we can get $w^*$ by resolving the least squares :
# 
# \begin{align*}
#     w^* = X^{\dagger}(y + v^*)
# \end{align*}

# ## Code for the testing
# 
# I decided to use Python to implement the testing on the data.
# 
# The following code only a part of the total code I wrote to do this, I only kept the essential for readibility reasons.
# You can find the complete code on this GitHub repository: [https://github.com/gwatkinson/HW3](https://github.com/gwatkinson/HW3).

# ## Rapid implementation
# 
# In this section, I implemented the mentioned function quite fast, and tested it on generated data.

# In[1]:


import numpy as np
import cvxpy as cp

from typing import Callable

f_type = Callable[[np.array], float]


# ### Defining the functions
# 
# This cell defines the function of the problem, given the necessary matrices.

# In[2]:


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


# ### Algorithms
# 
# This cell implements the algorithms used for resolving the convex problem.
# 
# We first define the `backtracking_line_search` abd the `newton_method` that works on any function, given the gradient and hessian.
# 
# Then, the `centering_step`, just applies the newton method to the constrained quadratic problem.
# 
# Lastly, the `barr_method` implements the barrier method for a given $\mu$.

# In[3]:


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


# ### Generating the data
# 
# This cell generates random data for a regression model. It was taken from the [CVXPY documentation](https://www.cvxpy.org/examples/machine_learning/lasso_regression.html), and adapted for our needs.
# 
# It defines $\beta$ coefficients, then remove a portion of them to create a sparse problem. Then, it generates a random $X$ dataset.
# Finally, we have 
# $$
# y = X \beta + \mathcal{N}(0, \sigma^2)
# $$
# 
# We can then define the other matrices of the problem $Q$, $p$, $A$ and $b$.

# In[62]:


def generate_data(N=100, D=50, ld=10, sigma=1, density=0.2):
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


if __name__ == "__main__":

# ## Test if the algorithms on the data
# 
# This section tests the preceding function on generated data.

# In[63]:


# Generate the data
    N = 100
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


# In[64]:


# Optimise with the barrier method
    """
    `vs` contains the variables iterates.
    `l_iterates` contains the other iterates generated during the centering steps.
    """
    print('Solving the optimization problem...')
    vs, ts, l_iterates = barr_method(
        Q, p, A, b, v0, eps=1e-10, t0=t0, mu=20, alpha=0.4, beta=0.5, verbose=False
    )


# In[65]:


# The final optimal point
    v_star = vs[-1]
    print()
    print(f"Optimal point: {v_star}")
    
    

# We see that the found point is inside the constraints and that $ \| Xv \|_{\infty} \le \lambda$.

# In[66]:
    print()
    print(f"Check if the constraints are respected: {(v_star @ A.T - b).max(): .6g}")



# In[67]:


# Value of the quadratic function evaluated at the optimal point.
    print()
    print(f"Value of the function at that point: {f(v_star, Q, p): .6g}")


# We can then look at the optimal $w^*$ obtrained by
# \begin{equation*}
#     w^* = X^{\dagger} (y + v^*)
# \end{equation*}

# In[68]:


    w_star = np.linalg.lstsq(X, y+v_star, rcond=None)[0]

    # Print the coefs of w_star >= 1e-5
    print()
    print(f"Value of lasso optimal point w* (>=1e-5): ")
    print(np.where(w_star >= 1e-5, w_star, 0))


# We obtain a sparse vector, which is a good sign as it is the objective of a LASSO regression.

# ### Confirmation with CVXPY
# 
# In this section, we confirm the results by using a reliable Python library `CVXPY` that implements convex optimization.

# In[69]:



# Posing the dual problem
    x = cp.Variable(N)
    dual_prob = cp.Problem(
        cp.Minimize(
            cp.quad_form(x, Q) + p.T @ x
            ),
        [A @ x <= b]
    )

    # Solving the problem
    print()
    print("Solving the dual problem with CVXPY")
    dual_prob.solve()

    # Printing the optimal solution
    print()
    print(f"Min value of the problem with CVXPY: {dual_prob.value:.6g}")
    print(f"Optimal point: {x.value}")


# We can see that the found optimal values are really close, less than $5 \times 10^{-11}$.

# In[70]:

    # Difference between custom implementation and CVXPy
    print()
    print(f"Difference between custom implementation and CVXPy: {f(v_star, Q, p) - dual_prob.value:.6g}")
    


# And the norm of the difference between the points is less than $2 \times 10^{-10}$.

# In[71]:


    # Difference between custom implementation and CVXPY
    print()
    print(f"Difference between custom implementation and CVXPY: {np.linalg.norm(v_star - x.value):.6g}")
    


# CVXPY problem on the LASSO problem.

# In[72]:


    # Pose the LASSO problem
    w = cp.Variable(D)
    prob = cp.Problem(
        cp.Minimize(
            cp.norm2(X @ w - y)**2 + ld * cp.norm1(w)
        )
    )

    # Solve the problem
    print()
    print("Solving the LASSO problem with CVXPY")
    prob.solve()

    # Optimal solution
    print()
    print(f"Min value of the LASSO problem with CVXPY: {prob.value:.6g}")
    print(f"Optimal point (>= 1e-5):")
    print(np.where(w.value>=1e-5, w.value, 0))


# In[73]:
    print()
    print(f"Found optimal point (>= 1e-5):")
    print(np.where(w_star>=1e-5, w_star, 0))


# We notice some differences between the solutions of the dual problem and the solution of the LASSO implemented by CVXPY. But, we still have the larger coefficients that are similar.

# ## Results
# 
# In this section, we will display plots resuming the optimisation process.
# 
# The code implementing this is different from the functions above.
# I wanted to get more familiar with Object Oriented Programming with Python, so I made a package that implements the same method with classes.
# I checked that the ouptuts were the same for a given dataset.
# The plots come from there.
# 
# You can view the entire code on this Github repository: [https://github.com/gwatkinson/HW3](https://github.com/gwatkinson/HW3).

# ## Centering step
# 
# In this section, we will focus more on the centering step used on the objective function with $t=100$.
# 
# This are just the results obtained by a Newton method on the objective function with $\lambda=10$, $\mu=10$, $t=100$, $\alpha=0.1$ and $\beta=0.5$
# 
# ![Centering step output](outputs/example_output_centering-10.png)
# 
# We clearly see the two parts of the method, as described in the course.

# ### Other example
# 
# On another note, I also used this method and a basic gradient descent on a 2D function, given as an example in the course, to plot the descent.
# 
# This is just a visual way to confirm that the method is working as intended.
# 
# ![alt text](outputs/comparaison_newton_gradient_4.png)

# ## Barrier method
# 
# This section presents results of the barrier method on the QP problem.
# 
# We plot the dual gap between the optimal value of the dual problem and the value of the LASSO function.
# 
# ![Barrier method results](outputs/lasso_pb_outputs_10.png)
# 
# We can see that the barrier method converged successfully. Furthermore, we observe that for small values of $\mu=2, 5$, many centering steps are needed, which also increases the number of Newton iterations.
# 
# From this graph, we can suppose that $\mu=50$ is a sweetspot for this problem, as it only required 60 Newton iterations, where smaller values need a lot more, and bigger values tend to take longer as well.
