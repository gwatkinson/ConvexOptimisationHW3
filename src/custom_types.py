"""Module for custom typing support for the other modules."""

from collections.abc import Callable

from nptyping import Float, NDArray, Shape

Vector = NDArray[Shape["Size"], Float]
Matrix = NDArray[Shape["Size, Size"], Float]

Function = Callable[[Vector], float]
Gradient = Callable[[Vector], Vector]
Hessian = Callable[[Vector], Matrix]
