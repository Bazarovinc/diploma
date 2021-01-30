# импортирование библиотеки типизации данных
from typing import List, NoReturn

# импортирование библиотеки для построения графиков
import matplotlib.pyplot as plt

# импортирование библиотеки для работы с матрицами, векторами
import numpy as np

# создание значений напряжения
V = np.linspace(0, 0.5, 30)


def draw_graphic(
    current: List[np.array], target_current: List[np.array], epohs: int, u: float
) -> NoReturn:
    """ "Функция для отрисовки 2-ух графиков: реального и того, что получаем
    с помощью нейронной сети"""
    plt.title(f"ВАХ резонансно туннельной структры {u}")
    # построение графика ВАХ с полученными значениями Тока из нейронной сети
    plt.plot(
        V,
        current,
        'r-o',
        color='r',
        label=f'Значения, полученные при помощи нейронной сети (эпохи={epohs})',
    )
    # построение графика реальных значений ВАХ
    plt.plot(V, target_current, 'r-o', color='b', label='Теоретически рассчитанные значения')
    plt.xlabel('U, В')  # подпись оси x
    plt.ylabel('I, А')  # подпись оси y
    plt.legend()  # легенда
    plt.grid()  # включаем сетку
    plt.show()  # вывод графика
