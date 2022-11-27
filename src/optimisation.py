"""Module that provides optimization functions."""

from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np

from src.custom_types import Line, Vector
from src.function_helper import FunctionHelper, LogBarrier, Quadratic


class UnconstrainedOptimizer:
    """Unconstrained optimizer."""

    # Init function
    def __init__(
        self,
        func: FunctionHelper,
        x0: Vector,
        method="newton",
        alpha=0.4,
        beta=0.6,
        error=1e-10,
        epsilon=1e-14,
        max_iter=100,
        max_t_iter=100
    ):
        """Initiate a new instance."""
        self.func = func
        self.x = x0.copy()
        self.y = self.func(self.x)
        self.grad = self.func.g(self.x)
        self.hess = self.func.h(self.x)
        self.max_iter = max_iter
        self.max_t_iter = max_t_iter
        self.iter = 0
        self.iter_t = 0
        self.epsilon = epsilon
        self.direction = None
        self.line = None
        self.backtracking_t = None
        self.t = None
        self.alpha = alpha
        self.beta = beta
        self.error = error

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

    def alpha_tangent(self, alpha=None):
        """Return the alpha tangent of the line function."""
        alpha = alpha or self.alpha
        return self.func.line_alpha_tangent(self.line, alpha)

    # Update functions
    def update_dir(self):
        """Update the search direction."""
        self.direction = self.dir_func()
        self.line = Line(self.x, self.direction)

    def backtracking_line_search(self, alpha: float=None, beta: float=None):
        """Calculate the step size with backtracking_line_search."""
        t = 1
        ts = [1]
        
        alpha = alpha or self.alpha
        beta = beta or self.beta

        line_func = self.line_func()
        alpha_tangent = self.alpha_tangent(alpha)

        self.iter_t = 0
        while line_func(t) - alpha_tangent(t) >= self.epsilon:
            t *= beta
            ts.append(t)
            self.iter_t += 1
            if self.iter_t >= self.max_t_iter:
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

    def optimisation_step(self, alpha=None, beta=None):
        """Chain the previous function to create a optimisation step."""
        alpha = alpha or self.alpha
        beta = beta or self.beta
        self.update_dir()
        self.backtracking_line_search(alpha, beta)
        self.update_x()
        self.update_lists()
        self.iter += 1

    def optimise(self, error=None, alpha=None, beta=None, max_iter=None, verbose=False):
        """Define the whole optimisation program."""
        error = error or self.error
        max_iter = max_iter or self.max_iter
        alpha = alpha or self.alpha
        beta = beta or self.beta
        while self.stop_criterion - error > self.epsilon:
            try:
                self.optimisation_step(alpha, beta)
                if self.iter >= self.max_iter:
                    raise ValueError("Unconstrained optimisation not converging.")
                if verbose:
                    print(f"Step {self.iter}:")
                    print(f"\t Criterion = {self.stop_criterion:.2e}")
                    print(f"\t y_{self.iter} = {self.y:.2g}")
                    print(
                        f"\t y_{self.iter-1} - y_{self.iter} = {self.ys[-2] - self.y:.2g}"
                    )
            except ValueError as e:
                print(e)
                break

        if verbose:
            print(f"Last step {self.iter}:")
            print(f"\t Criterion = {self.stop_criterion:.2e}")
            print(f"\t y_{self.iter} = {self.y:.2g}")

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

    def plot_line(self, alpha=None, t_range=None, n=None, **kwargs):
        """Plot the current line function with the tangent and alpha tangent."""
        self.update_dir()
        self.alpha = alpha or self.alpha
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

class BarrierMethod:
    """Constrained optimizer."""

    def __init__(
        self,
        func,
        A,
        b,
        x0,
        t0,
        mu,
        centering_kwargs,
        error=1e-10,
        tol=1e-14
    ):
        """Initiate the optimizer."""
        self.func = func
        self.A = A
        self.b = b
        self.log_barrier = LogBarrier(A, b)
        self.m = len(b)
        self.x = x0
        self.y = self.func(self.x)
        self.t = t0
        self.stop_criterion = self.m / self.t
        
        self.mu = mu
        self.error = error
        self.tol = tol
        self.iter = 0
        self.kwargs = centering_kwargs
        
        self.center_f = None
        self.center_pb = None
        self.number_centering_step = None
        
        self.xs = [self.x]
        self.ys = [self.y]
        self.ts = []
        self.stop_criterions = []
        self.center_pbs = []
        self.number_centering_steps = []


    def update_centering_function(self):
        """Compute the optimal point of the current centering problem."""
        self.center_f = self.t * self.func + self.log_barrier
        self.center_pb = UnconstrainedOptimizer(self.center_f, self.x, **self.kwargs)
        
    def optimise_centering_problem(self):
        self.center_pb.optimise()
        self.number_centering_step = self.center_pb.iter
    
    def centering_step(self):
        self.x = self.center_pb.x
        self.y = self.func(self.x)
        self.stop_criterion = self.m / self.t
    
    def update_lists(self):
        self.xs.append(self.x)
        self.ys.append(self.y)
        self.ts.append(self.t)
        self.stop_criterions.append(self.stop_criterion)
        self.center_pbs.append(self.center_pb)
        self.number_centering_steps.append(self.number_centering_step)
    
    def update_t(self):
        self.t *= self.mu
    
    def optimisation_step(self):
        self.update_centering_function()
        self.optimise_centering_problem()
        self.centering_step()
        self.update_lists()
        self.update_t()
        self.iter += 1
        
    def optimise(self, error=None, verbose=False):
        """Optimise the problem."""
        error = error or self.error
        while self.stop_criterion - self.error >= self.tol:
            try:
                self.optimisation_step()
                if verbose:
                    print(f"Step {self.iter}:")
                    print(f"\t Criterion = {self.stop_criterion:.2e}")
                    print(f"\t y_{self.iter} = {self.y:.2g}")
                    print(f"\t Centering step final criterion = {self.center_pb.stop_criterion:.2g}")
                    print(f"\t Number of steps in centering = {self.number_centering_step}")
            except ValueError as e:
                print("Centering step not converging")
                print(e)
                break
        
        
class QuadraticBarrierMethod(BarrierMethod):
    """Barrier method for a quadratic function."""
    
    def __init__(
        self,
        Q, p, A, b, x0, t0, mu, centering_kwargs, error=1e-10, tol=1e-14
    ):
        """Initiate a quadratic function and the associated barrier method."""
        self.Q = Q
        self.p = p
        quadratic_func = Quadratic(Q, p)

        super().__init__(
            quadratic_func,
            A,
            b,
            x0,
            t0,
            mu,
            centering_kwargs,
            error,
            tol
        )