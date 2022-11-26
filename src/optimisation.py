"""Module that provides optimization functions."""

from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np

from src.custom_types import Line, Vector
from src.function_helper import FunctionHelper


class UnconstrainedOptimizer:
    """Unconstrained optimizer."""

    # Init function
    def __init__(
        self,
        func: FunctionHelper,
        x0: Vector,
        method="newton",
        epsilon=1e-10,
        max_iter=100,
    ):
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
        self.direction = None
        self.line = None
        self.backtracking_t = None
        self.t = None

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
        """Calculate the direction with the Newton method."""
        return -np.linalg.inv(self.hess) @ self.grad

    def gradient_step(self):
        """Calculate the direction with the gradient method."""
        return -self.grad

    # Stopping functions
    def newton_decrement(self):
        """Calculate the stopping criterion for the Newton method."""
        return self.grad.T @ np.linalg.inv(self.hess) @ self.grad / 2

    def gradient_stop(self):
        """Calculate the stopping criterion for the gradient method."""
        return np.linalg.norm(self.grad)

    # Line subfunctions
    def line_func(self):
        """Return the function restricted to the current line."""
        return self.func.line_func(self.line)

    def alpha_tangent(self, alpha):
        """Return the alpha tangent of the line function."""
        return self.func.line_alpha_tangent(self.line, alpha)

    # Update functions
    def update_dir(self):
        """Update the search direction."""
        self.direction = self.dir_func()
        self.line = Line(self.x, self.direction)

    def backtracking_line_search(self, alpha: float, beta: float):
        """Calculate the step size with backtracking_line_search."""
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
        """Update the current point, criterion, gradient, hessian and function value."""
        self.x = self.x.copy() + self.t * self.direction
        self.y = self.func(self.x)
        self.grad = self.func.g(self.x)
        self.hess = self.func.h(self.x)
        self.stop_criterion = self.stop_func()

    def update_lists(self):
        """Append the current values to the lists."""
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
        """Chain the previous function to create a optimisation step."""
        self.update_dir()
        self.backtracking_line_search(alpha, beta)
        self.update_x()
        self.update_lists()

    def optimise(self, error, alpha, beta, verbose=False):
        """Define the whole optimisation program."""
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
                    print(
                        f"\t y_{self.iter-1} - y_{self.iter} = {self.ys[-2] - self.y:.2g}"
                    )
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
        """Plot the evolution of the criterion."""
        ax = ax or plt.gca()
        xticks = list(range(len(self.stop_criterions)))
        ax.set(
            xlabel="$k$",
            ylabel=r"$\lambda^2$",
            xticks=xticks,
            yscale="log",
            title="Stopping criterion",
        )
        ax.plot(xticks, self.stop_criterions, **kwargs)

    def plot_gap(self, ax=None, **kwargs):
        """Plot the evolution of the gap."""
        ax = ax or plt.gca()
        gap = np.array(self.ys) - np.min(np.array(self.ys))
        xticks = list(range(1, len(gap) + 1))
        ax.set(
            xlabel="$k$",
            ylabel=r"$f(x^{(k)}) - p^*$",
            xticks=xticks,
            yscale="log",
            title="Gap to $p^*$",
        )
        ax.plot(xticks, gap, **kwargs)

    def plot_direction_norm(self, ax=None, **kwargs):
        """Plot the evolution of the direction norm."""
        ax = ax or plt.gca()
        step = np.linalg.norm(self.directions, axis=1)
        xticks = list(range(1, len(step) + 1))
        ax.set(
            xlabel="Iteration",
            ylabel=r"Norm of the direction",
            xticks=xticks,
            yscale="log",
            title="Norm of the directions",
        )
        ax.plot(xticks, step, **kwargs)

    def plot_step_size(self, ax=None, **kwargs):
        """Plot the evolution of the step size."""
        ax = ax or plt.gca()
        step = np.array(self.ts)
        xticks = list(range(1, len(step) + 1))
        ax.set(
            xlabel="Iteration",
            ylabel=r"Step size",
            ylim=[0, 2],
            xticks=xticks,
            title="Step size",
        )
        ax.plot(xticks, step, **kwargs)

    def plot_ouptuts(self, **kwargs):
        """Create a figure with all the previous plots."""
        fig, axs = plt.subplots(2, 2)
        fig.suptitle("Results of the optimisation steps")
        self.plot_criterion(ax=axs[0][0], **kwargs)
        self.plot_gap(ax=axs[0][1], **kwargs)
        self.plot_direction_norm(ax=axs[1][0], **kwargs)
        self.plot_step_size(ax=axs[1][1], **kwargs)
        fig.tight_layout()
        return fig

    def plot_line(self, alpha, t_range=None, n=None, **kwargs):
        """Plot the current line function with the tangent and alpha tangent."""
        self.update_dir()
        self.line.t_range = t_range or self.line.t_range
        self.line.n = n or self.line.n
        self.func.plot_line(self.line, label="Function", **kwargs)
        self.func.plot_taylor_approximation(
            self.line, label="Tangent", ls="--", **kwargs
        )
        self.func.plot_alpha_tangent(
            self.line, alpha=alpha, label=r"$\alpha$-tangent", ls="-.", **kwargs
        )
        plt.title(rf"Line function in the chosen direction with $\alpha = {alpha}$")
