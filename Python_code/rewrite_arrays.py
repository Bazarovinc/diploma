# импортирование библиотеки для работы с матрицами, векторами
# импортирование библиотеки для типизации данных
from typing import List

# импортирование библиотеки для работы с матрицами, векторами
import numpy as np


def rewrite_arrays(array: List[str]) -> List[np.array]:
    """Перезапись поданной матрицы и строкового типа, в котором она была
    извлечена из файла, в числовой (float)."""
    for i in range(len(array)):
        if array[i][-1] == '\n' and array[i][-2] == ',':
            array[i] = np.asfarray(array[i].split(',')[:-1])
        else:
            array[i] = np.asfarray(array[i].split(','))
    return array
