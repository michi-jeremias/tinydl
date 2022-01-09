from enum import Enum, auto


class Stage(Enum):
    """Stages of Runner."""

    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()
    UNDEFINED = auto()
