import numpy as np
from test_data import TEST_INPUT


def get_query(n_n):
    # получение значений тока РТС с помощью ИНС
    I = np.array([0])
    I = np.append(I, (n_n.query(TEST_INPUT) * 1e-5))
    return I


def train_for_epoh(n_n, input_data, output_data, order):
    for i in order:
        # вызываем метод тренировки нейронной сети
        n_n.train(input_data[i], output_data[i][1:])
    return get_query(n_n)
