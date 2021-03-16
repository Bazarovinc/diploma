# импортирование библиотеки для работы с векторами
import numpy as np

# импортирование класса для образования дата сета для тестов
from torch.utils.data import DataLoader

# импортирование конфигураций сети
from configurations import net_config

# импортирование тестового дата сета
from data_set import test_data

#  импортирование функции отрисовки графиков ВАХ
from draw_graphic import draw_graphic

# импортирование классов для использования функции оптимизации и функции потерь
from torch import nn, optim


def test_net(net):
    test_loader = DataLoader(test_data)
    # перевод сети в режим проверки (тестирования)
    net.eval()
    # проход по дата сету
    # создаем объект функции потерь
    criterion = nn.SmoothL1Loss()
    for i, (data, target) in enumerate(test_loader):
        # получение выходных значений сети
        out = net(data)
        loss = criterion(out, target)
        out *= 1e-5
        print(loss.item())
        # создание массива со значениями ток и добавление 0 в начало
        current = np.array([0])
        # добавление значений полученных нейронной сетью
        current = np.append(current, out.detach().numpy())
        # создание массива с реальными значениями тока и добавление 0 в начало
        target_current = np.array([0])
        # добавление реальных значений тока
        target_current = np.append(target_current, target.detach().numpy() * 1e-5)
        # отрисовка графиков ВАХ
        draw_graphic(current, target_current, net_config.epohs, round(float(data[0][18]), 2))
