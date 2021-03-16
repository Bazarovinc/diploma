# импортирование классов для использования функции оптимизации и функции потерь
from torch import nn, optim, version

# импортирование класса для образования дата сета для тренировки
from torch.utils.data import DataLoader

# импортирование конфигураций сети
from configurations import net_config

# импортирование тренировочного дата сета
from data_set import train_data

# импортирование функции отрисовки графиков тренировки
from draw_learning_graphics import draw_learnig_graphics
from test_net import test_net


def configure_train(net: nn.Module) -> None:
    print(version.__version__)
    # создание объекта для оптимизации нейронной сети (используемый метод стохастический градиентный
    # спуск) передаем параметры сети и коэффициент обучения
    optimizer = optim.SGD(net.parameters(), lr=net_config.learning_rate)
    # создаем объект для изменения коэффициента обучения
    sc = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0.001, max_lr=0.9, step_size_up=2, mode="triangular"
    )
    # создаем объект функции потерь
    criterion = nn.SmoothL1Loss()
    # цикл тренировки в эпохах
    answer = 1
    epoh = 10
    while answer:
        for e in range(epoh):
            # создание объекта тренировочного дата сета, в котором данные объеденены в пакеты по 64 и
            # перемешаны
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            # проход по мини пакетам
            for batch_idx, (data, target) in enumerate(train_loader):
                # сброс значения градиента
                optimizer.zero_grad()
                # прямой проход данных через сеть
                net_out = net(data)
                # вычисление значения функции потерь
                loss = criterion(net_out, target)
                # обратное распространение
                loss.backward()
                # шаг оптимизации сети
                optimizer.step()
            # шаг по изменению весового коэффициента
            sc.step(epoh - 10 + e)
        # добавление весового коэффициента
        #lr_array.append(float(optimizer.param_groups[0]['lr']))

        # добавление значений функции потерь в массив
            #loss_array.append(float(loss.item()))
        # вывод эпохи, коэффициента обучения и функции потерь в текущем состоянии
        print(epoh, optimizer.param_groups[0]['lr'], loss.item())
        test_net(net)
        epoh += 10
        answer = int(input("Continue? (1-yes/0-no)\n"))
