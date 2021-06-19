import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from math import *


class FourierSeries():
    def __init__(self, func, coeffs_num, x_grid):
        self.func = func
        self.n = coeffs_num
        self.An = []
        self.Bn = []
        self.x = x_grid

    def calculate_coeffs(self):
        cos_func = lambda x: self.func(x) * cos(i * x)
        sin_func = lambda x: self.func(x) * sin(i * x)
        a0 = quad(self.func, -np.pi, np.pi)[0]*(1.0/np.pi)
        self.An.append(a0)
        self.Bn.append(0)

        for i in range(1, self.n):
            an = quad(cos_func, -np.pi, np.pi)[0]*(1.0/np.pi)
            bn = quad(sin_func, -np.pi, np.pi)[0]*(1.0/np.pi)

            self.An.append(an)
            self.Bn.append(bn)

    def calculate_result(self, x):
        sum = self.An[0] / 2
        for i in range(1, self.n):
            sum += self.An[i]*np.cos(i * x) + self.Bn[i]*np.sin(i * x)
        return sum

# Тестирование класса
def test_func(x):
    return x * x

def test_class(n):
    grid = np.arange(-1, 1, 0.001)
    fr = FourierSeries(func=test_func, coeffs_num=n, x_grid=grid)
    fr.calculate_coeffs()

    plt.plot(grid, [fr.calculate_result(grid[i]) for i in range(2000)], 'g', label="Фурье")
    plt.plot(grid, [test_func(grid[i]) for i in range(2000)], 'r--', label="Исходная функция")
    plt.title("Фурье преобразование")
    plt.show()

