from dataclasses import dataclass
from torch import Tensor
from tinydl.stage import Stage


@dataclass
class Report:
    metric_name: str
    metric_value: Tensor
    stage: Stage
