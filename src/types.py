from typing import Any, Callable, Tuple

import numpy as np

NSamples = int
DDimensions = int
Vector = np.ndarray(shape=(int,), dtype=float)
Matrix = np.ndarray(shape=(int, int), dtype=float)
DataMatrix = np.ndarray(shape=(NSamples, DDimensions), dtype=float)
