import unittest

import numpy as np
from numpy.testing import assert_allclose
from densityestimation.kernels import *


class Kernels(unittest.TestCase):
    def test_boxcar_kernel(self):
        grid = np.linspace(-2, 2, 10)
        self.assertIsNone(assert_allclose(boxcar_kernel(grid),
                                          np.array([0., 0., 0., 0.5, 0.5, 0.5, 0.5, 0., 0., 0.])))

    def test_epanechnikov_kernel(self):
        grid = np.linspace(-2, 2, 10)
        self.assertIsNone(assert_allclose(epanechnikov_kernel(grid),
                                          np.array([-0., -0., -0., 0.41666667, 0.71296296,
                                                    0.71296296, 0.41666667, -0., -0., -0.])))

    def test_tricube_kernel(self):
        grid = np.linspace(-2, 2, 10)
        self.assertIsNone(assert_allclose(tricube_kernel(grid),
                                          np.array([-0., -0., -0., 0.30114977, 0.83605766,
                                                    0.83605766, 0.30114977, -0., -0., -0.])))

    def test_gaussian_kernel(self):
        grid = np.linspace(-2, 2, 10)
        self.assertIsNone(assert_allclose(gaussian_kernel(grid),
                                          np.array([0.05399097, 0.11897819, 0.21519246, 0.31944801, 0.38921247,
                                                    0.38921247, 0.31944801, 0.21519246, 0.11897819, 0.05399097])))
