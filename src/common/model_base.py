from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, x, y, epochs: int):
        pass

    @abstractmethod
    def predict(self, x):
        pass
