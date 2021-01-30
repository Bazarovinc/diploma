import matplotlib.pyplot as plt  # импортирование библиотеки для построения графиков


def draw_learnig_graphics(lr, loss_1, epohs):
    """Функция по отрисовке графиков связанных с обученим"""
    # создание массива эпох
    epoh = [i for i in range(epohs)]
    plt.title(f"Зависимость функции потерь от коэффициента обучения")
    # построение графика зависимости функции потерь от коэффициента обучения
    plt.plot(loss_1, lr, color='r')
    plt.xlabel('loss')  # подпись оси x
    plt.ylabel('lr')  # подпись оси y
    plt.grid()  # включаем сетку
    plt.show()  # вывод графика
    plt.title(f"Зависимость функции потерь от эпох")
    # построение графика зависимости функции потерь от эпох
    plt.plot(epoh, loss_1, color='r')
    plt.xlabel('epohs')  # подпись оси x
    plt.ylabel('loss')  # подпись оси y
    plt.grid()  # включаем сетку
    plt.show()  # вывод графика
    plt.title(f"Зависимость функции потерь от коэффициента обучения")
    # построение графика зависимости функции потерь от коэффициента обучения
    plt.plot(epoh, lr, color='r')
    plt.xlabel('epoh')  # подпись оси x
    plt.ylabel('lr')  # подпись оси y
    plt.grid()  # включаем сетку
    plt.show()  # вывод графика
