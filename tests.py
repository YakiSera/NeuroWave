import scipy.integrate as integrate
import numpy as np
import math


def gauss_func(x):
    return -x * math.exp(-x / 2)


class WaveletFromModel():
    def __init__(self, ai, bi, Ai, fun, n):
        self.n = n
        self.A = Ai
        self.a = ai
        self.b = bi
        self.func = fun

    def calculate(self, x):
        result = 0
        for i in range(0, self.n):
            result += self.A[0][i] * self.func(self.a[0][i] * x + self.b[i])
        return result


class ErrorCalculator():
    def __init__(self, f1, f2):
        self.f_calc = f1
        self.f_real = f2

    def integrated(self, x):
        return (self.f_calc(x) - self.f_real(x)) ** 2

    def calculate_integral(self):
        I = integrate.quad(self.integrated, -1, 1)
        err = np.sqrt(I[0])
        return err