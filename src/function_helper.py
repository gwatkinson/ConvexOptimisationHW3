"""Module that defines function classes that are used in the optimization algorithms."""

from numbers import Number
from typing import Union

import numpy as np
import seaborn as sns

from .custom_types import (
    Function,
    Gradient,
    Hessian,
    Line,
    Matrix,
    ScalarFunctionType,
    Vector,
)


class ScalarFunction:
    """Class representing a scalar function."""

    def __init__(
        self, f: ScalarFunctionType, g: ScalarFunctionType, h: ScalarFunctionType
    ):
        """Initiate a new scalar function."""
        self.f = f
        self.g = g
        self.h = h

    def __call__(self, x: float) -> float:
        """Return the value of the function evaluated at x."""
        return self.f(x)


class FunctionHelper:
    """Class representing a function with its gradient and hessian."""

    def __init__(
        self,
        f: Function,
        g: Gradient,
        h: Hessian,
    ) -> None:
        """Class representing a function with its gradient and hessian.

        Args:
            f (Function): The function that takes a numpy array as input and returns a real.
            g (Gradient): The gradient of the function.
            h (Hessian): The Hessian of the function.
        """
        self.f = f
        self.g = g
        self.h = h

    # Operations
    def __mul__(self, other: Union["FunctionHelper", Number]) -> "FunctionHelper":
        """Define the multiplication for a function.

        Args:
            other (FunctionHelper | Number): Another function or a Number.

        Returns:
            FunctionHelper: The function that is the product of the two functions,
            with the correct gradient and hessian.
        """
        if isinstance(other, FunctionHelper):
            f = lambda x: self.f(x) * other.f(x)
            g = lambda x: self.g(x) * other.f(x) + self.f(x) * other.g(x)
            h = (
                lambda x: self.h(x) * other.f(x)
                + 2 * self.g(x).T @ other.g(x)
                + self.f(x) * other.h(x)
            )
        elif isinstance(other, Number):
            f = lambda x: self.f(x) * other
            g = lambda x: self.g(x) * other
            h = lambda x: self.h(x) * other
        return FunctionHelper(f, g, h)

    def __rmul__(self, other: Union["FunctionHelper", Number]) -> "FunctionHelper":
        """Define the right multiplication function (same as the left multiplication)."""
        return self.__mul__(other)

    def __add__(self, other: Union["FunctionHelper", Number]) -> "FunctionHelper":
        """Define the addition for a function.

        Args:
            other (FunctionHelper | Number): Another function or a Number.

        Returns:
            FunctionHelper: The function resulting from the addition,
            with the correct gradient and hessian.
        """
        if isinstance(other, FunctionHelper):
            f = lambda x: self.f(x) + other.f(x)
            g = lambda x: self.g(x) + other.g(x)
            h = lambda x: self.h(x) + other.h(x)
        elif isinstance(other, Number):
            f = lambda x: self.f(x) + other
            g = self.g
            h = self.h
        return FunctionHelper(f, g, h)
    
    def __radd__(self, other: Union["FunctionHelper", Number]) -> "FunctionHelper":
        """Define the rightside addition for a function."""
        return self.__add__(other)
        

    def __call__(self, x: Vector) -> Number:
        """Call method to quickly use the main function.

        Args:
            x (Vector): The point at which to evaluate the function.

        Returns:
            Number: The value of f(x).
        """
        return self.f(x)

    # Line subfunctions
    def line_func(self, line: Line) -> ScalarFunction:
        """Value of the function along a line."""
        x0, direction = line.x0, line.direction
        f = lambda t: self.f(x0 + t * direction)
        g = lambda t: direction.T @ self.g(x0 + t * direction)
        h = lambda t: direction.T @ self.h(x0 + t * direction) @ direction
        return ScalarFunction(f, g, h)

    def line_tangent(self, line: Line, offset: float = 0.0) -> ScalarFunction:
        """Tangent of the line function with an offset on t."""
        line_f = self.line_func(line)
        return lambda t: line_f(offset) + (t - offset) * line_f.g(offset)

    def line_second_order_approximation(
        self, line: Line, offset: float = 0.0
    ) -> ScalarFunction:
        """Tangent of the line function with an offset on t."""
        line_f = self.line_func(line)
        return (
            lambda t: line_f(offset)
            + (t - offset) * line_f.g(offset)
            + 0.5 * (t - offset) ** 2 * line_f.h(offset)
        )

    def line_alpha_tangent(
        self, line: Line, alpha: float, offset: float = 0.0
    ) -> ScalarFunction:
        """Second derivative of the line function."""
        line_f = self.line_func(line)
        return lambda t: line_f(offset) + alpha * (t - offset) * line_f.g(offset).T

    # Plotting functions
    def plot_line(self, line: Line, t_range=None, n=None, **kwargs):
        """Plot the values of the function on a line defined by the initial point and a direction.

        Args:
            line (Line): The line to plot.
                Defined by a Line dataclass.
        """
        ts = line.ts(t_range, n)  # Array of the ts values.
        line_f = np.vectorize(self.line_func(line))
        ys = line_f(ts)
        sns.lineplot(x=ts, y=ys, **kwargs)

    def plot_taylor_approximation(
        self, line: Line, order: int = 1, offset: Number = 0, **kwargs
    ):
        """Plot the values of the taylor approximation of the function restricted to a line.

        Args:
            line (Line): The line to plot.
                Defined by a Line dataclass.
            order (int, optional): The order of the taylor approximation. Defaults to 1.
            offset (Number, optional): The offset where to apply the taylor approximation.
                Defaults to 0.
        """
        ts = line.ts()
        if order == 1:
            f = np.vectorize(self.line_tangent(line, offset))
        elif order == 2:
            f = np.vectorize(self.line_second_order_approximation(line, offset))
        ys = f(ts)  # Taylor approximation of the line function.
        sns.lineplot(x=ts, y=ys, **kwargs)

    def plot_alpha_tangent(
        self,
        line: Line,
        alpha: float = 1,
        offset: Number = 0,
        t_range=None,
        n=None,
        **kwargs
    ):
        """Plot the values of the taylor approximation of the function restricted to a line.

        Args:
            line (Line): The line to plot.
                Defined by a Line dataclass.
            order (int, optional): The order of the taylor approximation. Defaults to 1.
            offset (Number, optional): The offset where to apply the taylor approximation.
                Defaults to 0.
        """
        ts = line.ts(t_range, n)
        f = np.vectorize(self.line_alpha_tangent(line, alpha, offset))
        ys = f(ts)  # Taylor approximation of the line function.
        sns.lineplot(x=ts, y=ys, **kwargs)


class LinearFunction(FunctionHelper):
    """Generate a linear function from the coef."""

    def __init__(self, c: Vector) -> None:
        """Generate a linear function from the coef.

        The function takes the form `f(x) = c.T @ x`

        Args:
            c (Vector): The vector for the bias.
        """
        self.p = c

        f = lambda x: c.T @ x
        g = lambda x: c
        h = lambda x: 0
        super().__init__(f, g, h)


class Quadratic(FunctionHelper):
    """Generate a quadratic function from the matrix and bias."""

    def __init__(self, Q: Matrix, p: Vector) -> None:
        """Generate a quadratic function from the matrix and bias.

        The function takes the form `f(x) = x.T @ Q @ x + p.T @ x`

        Args:
            Q (Matrix): The matrix for the quadratic form, should be positive.
            p (Vector): The vector for the bias.
        """
        self.Q = Q
        self.p = p

        f = lambda x: x.T @ Q @ x + p.T @ x
        g = lambda x: 2 * Q @ x + p
        h = lambda x: 2 * Q
        super().__init__(f, g, h)


class LogAffineFunction(FunctionHelper):
    """Generate a function that is the opposite log of a affine function."""

    def __init__(self, a: Vector, b: float) -> None:
        """Generate a negative log affine function from the coef and the bias.

        The function takes the form `f(x) = - log(b - a.T @ x)`.
        It is a convex function.

        Args:
            a (Vector): The vector for the bias.
            b (float): The bias.
        """
        self.a = a
        self.b = b

        f = lambda x: float(np.nan_to_num(-np.log(b - a.T @ x), nan=np.inf))
        g = lambda x: a / (b - a.T @ x)
        h = lambda x: a.reshape(-1, 1) @ a.reshape(-1, 1).T / (b - a.T @ x) ** 2
        super().__init__(f, g, h)


class LogBarrier(FunctionHelper):
    """Generate a log barrier function.

    Which is a sum of LogAffineFunctions.
    """

    def __init__(self, A: Matrix, b: Vector) -> None:
        """Generate a negative log affine function from the coefs and the biases.

        The function takes the form `f(x) = - log(b_i - a_i.T @ x)`.
        It is a convex function.

        We note I the number of constraints and N the dimension of the data.

        Args:
            A (Matrix): The matrix for the coefs (each line is a coef).
                The format should look like (I, N)
            b (Vector): The biases, in format (I,).
        """
        self.a = A
        self.b = b
        self.n_barriers = len(b)
        self.barriers = []

        total_barrier = 0.
        for i in range(self.n_barriers):
            ai, bi = A[i, :], b[i]
            barrier = LogAffineFunction(ai, bi)
            self.barriers.append(barrier)
            total_barrier += barrier

        f, g, h = total_barrier.f, total_barrier.g, total_barrier.h
        super().__init__(f, g, h)
