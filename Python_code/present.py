from class_neural import neuralNetwork  # импортирование класса нейронной сети
import numpy as np
from rewrite_arrays import rewrite_arrays
from read_data_from_file import read_data
from train_and_get_result import train_for_epoh
from draw_graphic import draw_grraphic

# задаем конфигурации нейронной сети
input_nodes = 49  # колличество нейронов во входном слое
hidden_node = [39]   # колличество нейронов и скрытые слоя
output_nodes = 29  # колличество нейронов во выходном слое
learning_rate = 0.01  # коэффициент обучения
# инициализируем нейронную сеть (создаем объект нейронной сети)
neural_network = neuralNetwork(input_nodes, hidden_node, output_nodes,
                               learning_rate)
input_file = '..\data_sets\input_1.csv'
output_file = '..\data_sets\output_data.csv'
# считываем входные данные из файла
input_data = read_data(input_file)
# считываем входные данные из файла
output_data = read_data(output_file)
# перезапись входных данных (в файле они содержатся с лишними \n)
input_new = []
for d in input_data:
    if d != '\n':
        input_new.append(d)
# перезапись вхожных и выходных данный в численный типа
input_data = rewrite_arrays(input_new)
output_data = rewrite_arrays(output_data)
# создание массива порядка, чтобы обращаться к элементам дата сетов
order = [i for i in range(len(input_data))]
answer = 1
while answer:
    np.random.shuffle(order)
    # обучение НС и получение значений тока на тестовых данных
    I = train_for_epoh(neural_network, input_data, output_data, order)
    # построение графиков для оценки обучения НС
    draw_grraphic(I)
    answer = int(input("Продолжать? (да-1, нет-0)\n"))
