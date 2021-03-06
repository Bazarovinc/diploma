# импортирование библиотеки типизации данных
from typing import List, NoReturn, Union

# импортирование библиотеки для работы с матрицами, векторами
import numpy as np
#  импортирование библиотеки для получения функции сигмоиды
import scipy.special as sc


# определение класса нейронной сети
class neuralNetwork:
    # инициализирование нейронной сети
    def __init__(self, input_nodes: int, hidden_nodes: List[int],
                 output_nodes: int, learning_rate: float) -> NoReturn:
        # задаем количество узлов во входном, скрытом и выходном слоях
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        # коэффициент обучения
        self.lr = learning_rate
        # список матриц весовых коэфициентов
        self.w = self.create_w()
        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: sc.expit(x)

    # создание списка матриц весовых коэфициентов
    def create_w(self) -> List[Union[np.ndarray, float]]:
        list_w = []
        """если длина скрытых слов 0, то мы создаем одну матрицу весов с
        размерам количество выходных нейронов и входных"""
        if len(self.hnodes) == 0:
            """записываем в список весов сгенерированные случайным образом
            весовые коэффициенты с применением"""
            # нормального распределения и сдвигом -0.5
            list_w.append(np.random.normal(0.0, pow(self.onodes, -0.5),
                                           (self.onodes, self.inodes)))
        else:
            """иначе проходим по списку количества нейронов скрытом слое и
            генерируем весовые коэффициенты"""
            for i in range(len(self.hnodes)):
                if i == 0:
                    list_w.append(np.random.normal(0.0,
                                                   pow(self.hnodes[i], -0.5),
                                                   (self.hnodes[i],
                                                    self.inodes)))
                    if len(self.hnodes) > 1:
                        list_w.append(np.random.normal(0.0,
                                                       pow(self.hnodes[i + 1],
                                                           -0.5),
                                                       (self.hnodes[i + 1],
                                                        self.hnodes[i])))
                    else:
                        list_w.append(np.random.normal(0.0, pow(self.onodes,
                                                                -0.5),
                                                       (self.onodes,
                                                        self.hnodes[i])))
                elif i == len(self.hnodes) - 1:
                    list_w.append(np.random.normal(0.0, pow(self.onodes, -0.5),
                                                   (self.onodes,
                                                    self.hnodes[i])))
                else:
                    list_w.append(np.random.normal(0.0, pow(self.hnodes[i + 1],
                                                            -0.5),
                                                   (self.hnodes[i + 1],
                                                    self.hnodes[i])))
        return list_w

    # метод для тренировки нейронной сети
    def train(self, inputs_list: List[float], targets_list: List[float]) -> \
            NoReturn:
        # преобразовать список входных в вектор
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        outputs_list = []
        outputs = 0
        i = 0
        # получение выходных сигналов на каждом слое
        for hn in self.w:
            if i == 0:
                outputs = self.activation_function(np.dot(hn, inputs))
            else:
                outputs = self.activation_function(np.dot(hn, outputs))
            outputs_list.append(outputs)
            i += 1
        # получение ошибки в зависимости от целевого значения на выходе
        output_errors = targets - outputs
        errors = []
        # процесс градиентного спуска в зависимости от колличества скртых слоев
        if len(self.hnodes) == 0:
            self.w[0] += self.lr * np.dot((output_errors * outputs_list[0] *
                                           (1.0 - outputs_list[0])),
                                          np.transpose(inputs))
        else:
            i = 0
            outputs_list = outputs_list[::-1]
            for out in outputs_list:
                if i == 0:
                    self.w[-(i + 1)] += self.lr * np.dot((output_errors * out
                                                          * (1.0 - out)),
                                                         np.transpose(
                                                             outputs_list[i +
                                                                          1]))
                elif i == len(outputs_list) - 1:
                    if len(self.w) > 2:
                        errors = np.dot(self.w[1].T, errors)
                        self.w[0] += self.lr * np.dot((errors * out *
                                                       (1.0 - out)),
                                                      np.transpose(inputs))
                    elif len(self.w) == 2:
                        errors = np.dot(self.w[-1].T, output_errors)
                        self.w[-(i + 1)] += self.lr * np.dot((errors *
                                                              out * (1.0 -
                                                                     out)),
                                                             np.transpose(
                                                                 inputs))
                elif i == 1:
                    errors = np.dot(self.w[-1].T, output_errors)
                    self.w[-(i + 1)] += self.lr * np.dot((errors * out *
                                                          (1.0 - out)),
                                                         np.transpose(
                                                             outputs_list[i +
                                                                          1]))
                else:
                    errors = np.dot(self.w[-i].T, errors)
                    self.w[-(i + 1)] += self.lr * np.dot((errors * out *
                                                          (1.0 - out)),
                                                         np.transpose(
                                                             outputs_list[i +
                                                                          1]))
                i += 1

    # опрос нейронной сети
    def query(self, inputs_list: List[float]) -> List[float]:
        # преобразовать список входных значений в двухмерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        # получение выходного сигнала
        for hn in self.w:
            inputs = self.activation_function(np.dot(hn, inputs))
        return inputs
