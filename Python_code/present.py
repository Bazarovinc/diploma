# импортирование библиотеки для работы с матрицами, векторами
import numpy as np

from class_neural import neuralNetwork  # импортирование класса нейронной сети
# импортирование функции отрисовки графика
from draw_graphic import draw_graphic
# импортирование функции для считывания данных из файла
from read_data_from_file import read_data
# импортирование функции для перезаписи массивов
from rewrite_arrays import rewrite_arrays
# импортирование функции тренировки и получения данных от нейронной сети
from train_and_get_result import train_for_epoh

# задаем конфигурации нейронной сети
input_nodes = 49  # колличество нейронов во входном слое
hidden_node = [39]   # колличество нейронов и скрытые слоя
output_nodes = 29  # колличество нейронов во выходном слое
learning_rate = 0.01  # коэффициент обучения
# инициализируем нейронную сеть (создаем объект нейронной сети)
neural_network = neuralNetwork(input_nodes, hidden_node, output_nodes,
                               learning_rate)
input_file = '../data_sets/input_1.csv'
output_file = '../data_sets/output_data.csv'
# считываем входные данные из файла
input_data = read_data(input_file)
# считываем входные данные из файла
output_data = read_data(output_file)
# перезапись входных данных (в файле они содержатся с лишними \n)
input_new = []
for d in input_data:
    if d != '\n':
        input_new.append(d)
# перезапись входных и выходных данный в численный типа
input_data = rewrite_arrays(input_new)
output_data = rewrite_arrays(output_data)
# создание массива порядка, чтобы обращаться к элементам дата сетов
order = [i for i in range(len(input_data))]
answer = 1
epohs = 0
while answer:
    np.random.shuffle(order)
    # обучение НС и получение значений тока на тестовых данных
    current = train_for_epoh(neural_network, input_data, output_data, order)
    epohs += 1
    # построение графиков для оценки обучения НС
    draw_graphic(current)
    answer = int(input("Продолжать? (да-1, нет-0)\n"))
print(f"Количество эпох: {epohs}")
