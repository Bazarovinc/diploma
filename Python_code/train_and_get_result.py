# импортирование библиотеки для работы с матрицами, векторами
from typing import TYPE_CHECKING, List

# импортирование библиотеки для работы с матрицами, векторами
import numpy as np

# импортирование тестовых данных
from test_data import TEST_INPUT

if TYPE_CHECKING:
    from class_neural import neuralNetwork


def get_query(n_n) -> List[np.array]:
    # получение значений тока РТС с помощью ИНС
    current = np.array([0])
    current = np.append(current, (n_n.query(TEST_INPUT) * 1e-5))
    return current


def train_for_epoh(n_n: 'neuralNetwork', input_data: List[np.array],
                   output_data: List[np.array], order: List[int]) \
        -> List[np.array]:
    """Функция для тренировки ИНС. Возвращает значения тока, полученные после
    тренировки."""
    for i in order:
        # вызываем метод тренировки нейронной сети
        n_n.train(input_data[i], output_data[i][1:])
    return get_query(n_n)
