from pywt import upcoef, downcoef
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fft import rfft, irfft
from scipy import integrate
import random
import plotstyle as plotter

def calculate_error(y_true, y_pred, grid):
    error = 0
    res_func = np.power(np.subtract(y_true,y_pred), 2)
    rand_distr = np.random.uniform(-1, 1, len(res_func))
    rand_distr = rand_distr.sort()
    error = integrate.simpson(res_func, rand_distr)
    return error

wavelet_name = 'haar'
lvl = 1

def test_function(x):
    return x * x * x

x = np.arange(-1, 1, 0.002)
y = [test_function(x[i]) for i in range(len(x))]
coeffs = downcoef('a', y, wavelet_name, level=lvl)
inverse = upcoef('a', coeffs, wavelet_name, level=lvl)

ifur = irfft(rfft(y))

err = calculate_error(y, inverse, x)
err2 = calculate_error(y, ifur, x)

print(err)
print(err2)

cur_fun_name = 'f(x) = x^3'
plotter.start_graph()
plotter.add_graph(x, inverse, 'Дискретное вейвлет-преобразование', 'r')
plotter.add_graph(x, y, cur_fun_name, '--g')
plotter.finish_graph()

plotter.start_graph()
plotter.add_graph(x, ifur, 'Быстрое преобразование Фурье', 'r')
plotter.add_graph(x, y, cur_fun_name, '--g')
plotter.finish_graph()