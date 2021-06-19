import matplotlib.pyplot as plt
import layers
from tests import WaveletFromModel, ErrorCalculator, gauss_func
import math
from modelbuilder import ModelBuilder
import time
from save_to_file import save_all
import pywt


def func_sin(x):
    return math.sin(x)


def func_x1(x):
    return x


def func_x2(x):
    return x * x


def func_x3(x):
    return x * x * x


def func_x4(x):
    return x * x * x * x


isFromML: bool = True
min_num: int = 10
max_num: int = 10
epochs: int = 5
layers.real_func = func_x3
in_val, out_val = layers.make_data_sets(-1, 1, 2000)

models = [ModelBuilder(num_of_neurons=i, data_x=in_val, data_y=out_val, infunc='gauss') for i in
          range(min_num, max_num + 1)]
wave_forms = []
errors_list = []
time_education_elapsed = []
time_calc_elapsed = []

for model in models:
    t0 = time.perf_counter()
    model.start_training(epochs)
    t1 = time.perf_counter() - t0
    time_education_elapsed.append(t1)
    wave_forms.append(
        WaveletFromModel(model.linear_layer.w.numpy(), model.linear_layer.b.numpy(), model.presum.w.numpy(),
                         gauss_func, model.num_of_neurons))

print(
    "Value from ML: {ml} | Value from WFM: {wfm} | Real value: {real}".format(ml=models[0].predict([1.5]),
                                                                              wfm=wave_forms[0].calculate(1.5),
                                                                              real=layers.real_func(1.5)))

for i in range(0, len(wave_forms)):
    t0 = time.perf_counter()
    wave_forms[i].calculate(0.5)
    t1 = time.perf_counter() - t0
    time_calc_elapsed.append(t1)
    err = ErrorCalculator(f1=models[i].predict if isFromML else wave_forms[i].calculate, f2=layers.real_func, ml_checker=isFromML)
    integral_err = err.calculate_integral()
    errors_list.append(integral_err)
    print("[{num}] [education time: {ed_time}] [calculation time: {calc_time}] Погрешность по формуле 1: {error1}"
          .format(num=wave_forms[i].n, ed_time=time_education_elapsed[i], calc_time=time_calc_elapsed[i],
                  error1=integral_err))
"""
for wave in wave_forms:
    t0 = time.perf_counter()
    wave.calculate(0.5)
    t1 = time.perf_counter() - t0
    time_calc_elapsed.append(t1)
    err = ErrorCalculator(f1=wave.calculate, f2=layers.real_func)
    integral_err = err.calculate_integral()
    errors_list.append(integral_err)
    print("[{num}] [education time: {ed_time}] [calculation time: {calc_time}] Погрешность по формуле 1: {error1}"
          .format(num=wave.n, ed_time=time_education_elapsed[wave.n - 1], calc_time=time_calc_elapsed[wave.n - 1],
                  error1=integral_err))
"""

save_all(errors_list, time_education_elapsed, time_calc_elapsed)
x_scale_factor = 250
gf_x = [i / x_scale_factor for i in range(-x_scale_factor, x_scale_factor)]
gf_y = [layers.real_func(gf_x[i]) for i in range(x_scale_factor * 2)]
predictions = [models[0].predict([gf_x[i]]) for i in range(x_scale_factor * 2)]

plt.clf()
plt.plot(gf_x, predictions, 'b', label="Аппроксимация")
plt.plot(gf_x, gf_y, "r", label="Аналитическое решение")
plt.title("Результат")
plt.legend()
plt.savefig("graph.png")
plt.show()
