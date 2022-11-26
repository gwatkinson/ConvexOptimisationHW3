"""Module that provides optimization functions."""

from __future__ import absolute_import
from src.function_helper import FunctionHelper
from src.custom_types import Vector, Matrix, Line
import numpy as np
import matplotlib.pyplot as plt

class UnconstrainedOptimizer:
    """Unconstrained optimizer."""

    # Init function
    def __init__(self, func: FunctionHelper, x0: Vector, method="newton", epsilon=1e-10, max_iter=100):
        """Initiate a new instance."""
        self.func = func
        self.x = x0.copy()
        self.y = self.func(self.x)
        self.grad = self.func.g(self.x)
        self.hess = self.func.h(self.x)
        self.max_iter = max_iter
        self.iter = 0
        self.iter_t = 0
        self.epsilon = epsilon
        
        if method == "newton":
            self.dir_func = self.newton_step
            self.stop_func = self.newton_decrement
        elif method == "gradient":
            self.dir_func = self.gradient_step
            self.stop_func = self.gradient_stop
        else:
            raise ValueError("Wrong step type, not in ['newton', 'gradient'].")
    
        self.xs = [self.x]
        self.ys = [self.y]
        self.grads = [self.grad]
        self.hesss = [self.hess]
        self.directions = []
        self.lines = []
        self.ts = []
        self.backtracking_ts = []
        self.stop_criterion = self.stop_func()
        self.stop_criterions = [self.stop_criterion]
        

    # Step functions
    def newton_step(self):
        return -np.linalg.inv(self.hess) @ self.grad
    
    def gradient_step(self):
        return -self.grad
    
    # Stopping functions
    def newton_decrement(self):
        return self.grad.T @ np.linalg.inv(self.hess) @ self.grad / 2
    
    def gradient_stop(self):
        return np.linalg.norm(self.grad)
        
        
    # Line subfunctions
    def line_func(self):
        return self.func.line_func(self.line)
    
    def alpha_tangent(self, alpha):
        return self.func.line_alpha_tangent(self.line, alpha)
    
    
    # Update functions
    def update_dir(self):
        self.direction = self.dir_func()
        self.line = Line(self.x, self.direction)
    
    def backtracking_line_search(self, alpha: float, beta: float):
        t = 1
        ts = [1]
        
        line_func = self.line_func()
        alpha_tangent = self.alpha_tangent(alpha)
        
        self.iter_t = 0
        while line_func(t) - alpha_tangent(t) >= self.epsilon:
            t *= beta
            ts.append(t)
            self.iter_t += 1
            if self.iter_t == self.max_iter:
                self.backtracking_t = ts
                self.t = t
                raise ValueError("Backtracking line search not converging.")
        
        self.backtracking_t = ts
        self.t = t
    
    def update_x(self):
        self.x = self.x.copy() + self.t * self.direction
        self.y = self.func(self.x)
        self.grad = self.func.g(self.x)
        self.hess = self.func.h(self.x)
        self.stop_criterion = self.stop_func()
    
    def update_lists(self):
        self.xs.append(self.x)
        self.ys.append(self.y)
        self.grads.append(self.grad)
        self.hesss.append(self.hess)
        self.directions.append(self.direction)
        self.lines.append(self.line)
        self.ts.append(self.t)
        self.backtracking_ts.append(self.backtracking_t)
        self.stop_criterions.append(self.stop_criterion)
    
    def optimisation_step(self, alpha, beta):
        self.update_dir()
        self.backtracking_line_search(alpha, beta)
        self.update_x()
        self.update_lists()
        
    def optimise(self, error, alpha, beta, verbose=False):
        while self.stop_criterion - error > self.epsilon:
            try:
                self.optimisation_step(alpha, beta)
                self.iter += 1
                if self.iter == self.max_iter:
                    raise ValueError("Optimisation not converging.")
                if verbose:
                    print(f"Step {self.iter}:")
                    print(f"\t Criterion = {self.stop_criterion:.2e}")
                    print(f"\t y_{self.iter} = {self.y:.2g}")
                    print(f"\t y_{self.iter-1} - y_{self.iter} = {self.ys[-2] - self.y:.2g}")
            except ValueError as e:
                print("Break in the algorithm")
                print(e)
                break
        
        print(f"Last step {self.iter}:")
        print(f"\t Criterion = {self.stop_criterion:.2e}")
        print(f"\t y_{self.iter} = {self.y:.2g}")
        print(f"\t y_{self.iter-1} - y_{self.iter} = {self.ys[-2] - self.y:.2g}")
        
        
    # Plotting functions
    def plot_criterion(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        ax.set(xlabel='$k$', ylabel=r'$\lambda^2$')
        ax.plot(self.stop_criterions, **kwargs)
        ax.set_yscale('log')
        
    def plot_gap(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        gap = np.array(self.ys) - np.min(np.array(self.ys))
        ax.set(xlabel='$k$', ylabel=r'$f(x^{(k)}) - p^*$')
        ax.plot(gap, **kwargs)
        ax.set_yscale('log')

    def plot_line(self, alpha, t_range=None, n=None, **kwargs):
        self.update_dir()
        self.line.t_range = t_range or self.line.t_range
        self.line.n = n or self.line.n
        self.func.plot_line(self.line, label="Function", **kwargs)
        self.func.plot_taylor_approximation(self.line, label="Tangent", ls='--', **kwargs)
        self.func.plot_alpha_tangent(self.line, alpha=alpha, label=r"$\alpha$-tangent", ls='-.', **kwargs)
        plt.title(rf"Line function in the chosen direction with $\alpha = {alpha}$")
        