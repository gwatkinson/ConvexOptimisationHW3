"""Module that defines function classes that are used in the optimization algorithms."""

import numpy as np

from .custom_types import Function, Gradient, Hessian, Matrix, Vector


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

    def __mul__(self, other: "FunctionHelper" | float) -> "FunctionHelper":
        """Define the multiplication for a function.

        Args:
            other (FunctionHelper | float): Another function or a float.

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
        elif isinstance(other, float):
            f = lambda x: self.f(x) * other
            g = lambda x: self.g(x) * other
            h = lambda x: self.h(x) * other
        return FunctionHelper(f, g, h)

    def __rmul__(self, other: "FunctionHelper" | float) -> "FunctionHelper":
        """Define the right multiplication function (same as the left multiplication)."""
        return self.__mul__(other)

    def __add__(self, other: "FunctionHelper" | float) -> "FunctionHelper":
        """Define the addition for a function.

        Args:
            other (FunctionHelper | float): Another function or a float.

        Returns:
            FunctionHelper: The function resulting from the addition,
            with the correct gradient and hessian.
        """
        if isinstance(other, FunctionHelper):
            f = lambda x: self.f(x) + other.f(x)
            g = lambda x: self.g(x) + other.g(x)
            h = lambda x: self.h(x) + other.h(x)
        elif isinstance(other, float):
            f = lambda x: self.f(x) + other
            g = self.g
            h = self.h
        return FunctionHelper(f, g, h)

    def __call__(self, x: Vector) -> float:
        """Call method to quickly use the main function.

        Args:
            x (Vector): The point at which to evaluate the function.

        Returns:
            float: The value of f(x).
        """
        return self.f(x)


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
