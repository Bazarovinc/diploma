from typing import List


def read_data(file_name: str) -> List[str]:
    """Повторяющийся код. Простое считывание строк из файла."""
    with open(file_name, 'r') as f_o:
        data = f_o.readlines()
    return data
