# импортирование библиотеки для работы с матрицами, векторами
import numpy as np

NS = 15
NC = 16
ND = 15
Np = NS + NC + ND
# создание массива потенциальной энергии
UB = np.concatenate((0.01 * np.ones((NS, 1)), 0.4 * np.ones((4, 1)),
                     0.01 * np.ones((NC - 8, 1)), 0.4 * np.ones((4, 1)),
                     0.01 * np.ones((ND, 1))))
UB = UB.T[0]
# создание тестовых данных
TEST_INPUT = np.concatenate(([0.2275], [0.1],	[0.3], UB))
