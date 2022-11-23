from typing import Any, Callable, Tuple

import numpy as np

NSamples = int
DDimensions = int
Vector = np.ndarray(shape=(int,), dtype=float)
Matrix = np.ndarray(shape=(int, int), dtype=float)
DataMatrix = np.ndarray(shape=(NSamples, DDimensions), dtype=float)


class FunctionHelper:
    def __init__(
        self,
        f: Callable[[Vector], float],
        g: Callable[[Vector], Vector],
        h: Callable[[Vector], Matrix],
    ) -> None:
        self.f = f
        self.g = g
        self.h = h

    def __mul__(self, other: "FunctionHelper" | float) -> "FunctionHelper":
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
        return self.__mul__(other)

    def __add__(self, other: "FunctionHelper" | float) -> "FunctionHelper":
        if isinstance(other, FunctionHelper):
            f = lambda x: self.f(x) + other.f(x)
            g = lambda x: self.g(x) + other.g(x)
            h = lambda x: self.h(x) + other.h(x)
        elif isinstance(other, float):
            f = lambda x: self.f(x) + other
            g = lambda x: self.g(x)
            h = lambda x: self.h(x)
        return FunctionHelper(f, g, h)

    def __call__(self, x: Vector) -> float:
        return self.f(x)


class Quadratic(FunctionHelper):
    def __init__(self, Q: np.ndarray, p: np.ndarray) -> None:
        self.Q = Q
        self.p = p

        f = lambda x: x.T @ Q @ x + p.T @ x
        g = lambda x: 2 * Q @ x + p
        h = lambda x: 2 * Q
        super().__init__(f, g, h)


class LogBarrier(FunctionHelper):
    pass
