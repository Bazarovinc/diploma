# импортирование библиотеки для считывания данных
import pandas as pd

# импортирование библиотеки для создания тензора
import torch

# импортирование класса дата сета
from torch.utils.data import Dataset


# Создание класса по созданию дата сета, необходим для дальнейшей работы с ним и правильной
# обработки
class DataSet(Dataset):
    def __init__(self, input_file, output_file):
        # считывание данных их фалов
        input_file_data = pd.read_csv(input_file)
        output_file_data = pd.read_csv(output_file)
        # перезапись данных
        input_data = input_file_data.iloc[0:, 0:49].values
        output_data = output_file_data.iloc[0:, 1:30].values
        # создание переменных для дата сета
        self.x = torch.tensor(input_data, dtype=torch.float)
        self.y = torch.tensor(output_data, dtype=torch.float)

    # создание методов необходимых для работы других инструментов с классом
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# создание объекта дата сета тренировочных данных
train_data = DataSet('../data_sets/input_big1.csv', '../data_sets/output_big1.csv')
# создание объекта дата сета тестовых данных
test_data = DataSet('../data_sets/input_big_t.csv', '../data_sets/output_big_t.csv')
