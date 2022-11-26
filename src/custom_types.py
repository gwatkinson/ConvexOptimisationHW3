"""Module for custom typing support for the other modules."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Number
from typing import Tuple

import numpy as np
from nptyping import Float, NDArray, Shape

Vector = NDArray[Shape["Size"], Float]
Matrix = NDArray[Shape["Size, Size"], Float]

Function = Callable[[Vector], float]
Gradient = Callable[[Vector], Vector]
Hessian = Callable[[Vector], Matrix]
ScalarFunctionType = Callable[[float], float]


@dataclass
class Line:
    """Class to define a line.

    Args:
        x0 (Vector): The starting point.
        direction (Vector): The direction of the line.
        t_range (Tuple[Number, Number], optional): The range around the point.
            Defaults to (-10, 10).
        n (int, optional): The number of point to sample.
            Can be specified in the following format functions.
            Defaults to 1000.
    """

    x0: Vector
    direction: Vector
    t_range: Tuple[Number, Number] = (-1, 1)
    n: int = 1000

    def ts(self, t_range=None, n=None) -> NDArray[Shape["Var"], Float]:
        """Generate the ts of the line.

        Args:
            t_range (Tuple[Number, Number], optional): The range around the point.
                Defaults to (-1, 1).
            n (int, optional): The number of points to generate.
                Defaults to self.n.

        Returns:
            NDArray[Shape["n"], Float]: Array containing the ts.
        """
        t_range = t_range or self.t_range
        n = n or self.n
        return np.linspace(*t_range, num=n)

    def xs(self, t_range=None, n=None) -> NDArray[Shape["Var"], Float]:
        """Generate the points of the line.

        Args:
            t_range (Tuple[Number, Number], optional): The range around the point.
                Defaults to (-1, 1).
            n (int, optional): The number of points to generate.
                Defaults to self.n.

        Returns:
            NDArray[Shape["Size, n"], Float]: Array containing the points of the line.
        """
        t_range = t_range or self.t_range
        n = n or self.n
        ts = self.ts(t_range=t_range, n=n)
        xs = self.x0.reshape(-1, 1) + ts * self.direction.reshape(-1, 1)
        return xs
