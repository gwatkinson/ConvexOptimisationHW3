import numpy as np
from typing import Callable, Any, Tuple

class FunctionHelper:
    def __init__(self, 
                 f: Callable[[np.array], float], 
                 g: Callable[[np.array], np.array], 
                 h: Callable[[np.array], np.array]) -> None:
        self.f = f
        self.g = g
        self.h = h
        
    def __mul__(self, other: "FunctionHelper" | float) -> "FunctionHelper":
        if isinstance(other, FunctionHelper):
            f = lambda x: self.f(x) * other.f(x)
            g = lambda x: self.g(x) * other.f(x) + self.f(x) * other.g(x)
            h = lambda x: self.h(x) * other.f(x) + 2 * self.g(x).T @ other.g(x) + self.f(x) * other.h(x)
        elif isinstance(other, float):
            pass