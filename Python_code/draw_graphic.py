# импортирование библиотеки для построения графиков
import matplotlib.pyplot as plt
# импортирование константных значений тока и напряжения
from constants import V, II


def draw_grraphic(I):
    plt.title("ВАХ резонансно туннельной структры")
    # построение графика ВАХ с полученными значениями Тока из нейронной сети
    plt.plot(V, I, 'r-o', color='r',
             label='Значения полученные при помощи нейронной сети')
    # построение графика реальных значений ВАХ
    plt.plot(V, II, 'r-o', color='b',
             label='Теоретически рассчитанные значения')
    # подпись оси x
    plt.xlabel('U, В')
    # подпись оси y
    plt.ylabel('I, А')
    # легенда
    plt.legend()
    plt.grid()  # включаем сетку
    plt.show()  # вывод графика