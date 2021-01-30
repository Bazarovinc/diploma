# импортирование библиотек для работы с нейронными сетями
import torch
import torch.nn as nn


# создание класса нейронной сети
class NeuralNetwork(nn.Module):
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int) -> None:
        super(NeuralNetwork, self).__init__()
        # создание слоя пакетной нормализации входных данных
        self.bn1 = nn.BatchNorm1d(input_nodes)
        # создание синапса между входным и скрытым слоями
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        # создание пакетной нормализации
        self.bn2 = nn.BatchNorm1d(hidden_nodes)
        # создание синапса между скрытым и выходным слоями
        self.fc2 = nn.Linear(hidden_nodes, output_nodes)
        # создание пакетной нормализации
        self.bn3 = nn.BatchNorm1d(output_nodes)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Метод прямого прохождения входных данных через нейронную сеть"""
        # проход через пакетную нормализацию
        x = self.bn1(x)
        # проход по синапсам
        x = self.fc1(x)
        # проход через пакетную нормализацию
        x = self.bn2(x)
        # проход через функцию активации-сигмоиду
        x = torch.sigmoid(x)
        # проход по синапсам
        x = self.fc2(x)
        # проход через пакетную нормализацию
        x = self.bn3(x)
        return x
