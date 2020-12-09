from densityestimation import kernels
from densityestimation.utils.gpuoptional import array_module
from densityestimation.utils.validators import validate_kernel, validate_bandwidth


class KernelDensityEstimator:
    def __init__(self, kernel, bandwidth, module=None):
        self.validate_parameters(kernel, bandwidth)

        self.kernel = kernels.KERNELS[kernel]
        self.X = None
        self.bandwidth = bandwidth
        self.xp = array_module(module)

    @staticmethod
    def validate_parameters(kernel, bandwidth):
        validate_kernel(kernel)
        validate_bandwidth(bandwidth)

    def fit(self, X, y=None):
        self.X = self.xp.asarray(X).reshape(-1, 1)

    def predict(self, X):
        return self.xp.mean(self.kernel((X - self.X) / self.bandwidth, module=self.xp), axis=0) / self.bandwidth
