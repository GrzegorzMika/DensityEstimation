import unittest

import numpy as np
from numpy.testing import assert_allclose
from densityestimation.kernels import *


class Kernels(unittest.TestCase):
    def test_boxcar_kernel(self):
        grid = np.linspace(-2, 2, 10)
        self.assertTrue(assert_allclose(boxcar_kernel(grid), np.array([0., 0., 0., 0.5, 0.5, 0.5, 0.5, 0., 0., 0.])))
