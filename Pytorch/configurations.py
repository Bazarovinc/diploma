from pydantic import BaseSettings


# Создание класса, который будет хранить в себе конфигурации сети
class Configurations(BaseSettings):

    input_nodes: int = 49
    hidden_nodes: int = 39
    output_nodes: int = 29
    learning_rate: float = 0.9
    epohs: int = 40


# Создание объекта
net_config = Configurations()
