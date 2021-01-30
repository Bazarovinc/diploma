from configurations import net_config  # импортирование конфигураций нейронной сети
from neural_network import NeuralNetwork  # импортирование нейронной сети
from test_net import test_net  # импортирование функции тестирования
from train_net import train_net  # импортирование функции тренировки

# создание объекта нейронной сети
net = NeuralNetwork(net_config.input_nodes, net_config.hidden_nodes, net_config.output_nodes)
# вызов функции тренировки сети
train_net(net)
# вызов функции тестирования сети
test_net(net)
