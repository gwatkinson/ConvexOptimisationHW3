"""Module that defines function classes that are used in the optimization algorithms."""

from numbers import Number
from typing import Union

import numpy as np
import seaborn as sns

from .custom_types import Function, Gradient, Hessian, Line, Matrix, Vector


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

    def __call__(self, x: Vector) -> Number:
        """Call method to quickly use the main function.

        Args:
            x (Vector): The point at which to evaluate the function.

        Returns:
            Number: The value of f(x).
        """
        return self.f(x)

    def plot_line(self, line: Line, **kwargs):
        """Plot the values of the function on a line defined by the initial point and a direction.

        Args:
            line (Line): The line to plot.
                Defined by a Line dataclass.
        """
        ts = line.ts()  # Array of the ts values.
        xs = line.xs()  # Array of the xs in the vector space.
        ys = np.apply_along_axis(self.f, 0, xs)

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
        inital_point = line.x0 + offset * line.direction
        value = self.f(
            inital_point
        )  # Value of the function at the point of the approximation.
        # First derivative of the line function at the initial point.
        grad = self.g(inital_point).T @ line.direction

        ys = value + (ts - offset) * grad  # Taylor approximation of the line function.

        if order == 2:
            # Second derivative of the line function at the initial point.
            hess = line.direction.T @ self.h(inital_point) @ line.direction
            ys += 0.5 * hess * (ts - offset) ** 2

        sns.lineplot(x=ts, y=ys, **kwargs)


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


class LogBarrier(FunctionHelper):
    """Generate a log barrier function."""

    pass
