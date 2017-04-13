from enum import Enum

class Models(Enum):
    LINEAR='linear'
    LOGISTIC='logistic'

class DataTypes(Enum):
    MATLAB = 'matlab'
    CSV = 'csv'

    def __init__(self, dtype):

        self.dtype=dtype

    @property
    def DataType(self):
        return self.dtype
