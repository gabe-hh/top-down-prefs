import math
from typing import Callable

class Scheduler:
    """Base class for parameter schedulers"""
    def __call__(self, step: int) -> float:
        raise NotImplementedError

class ConstantScheduler(Scheduler):
    """Keeps parameter constant"""
    def __init__(self, value: float):
        self.value = value
    
    def __call__(self, step: int) -> float:
        return self.value

class LinearScheduler(Scheduler):
    """Linear interpolation between start and end values"""
    def __init__(self, start_value: float, end_value: float, num_steps: int):
        self.start = start_value
        self.end = end_value
        self.num_steps = num_steps
    
    def __call__(self, step: int) -> float:
        progress = min(step / self.num_steps, 1.0)
        return self.start + (self.end - self.start) * progress

class CosineScheduler(Scheduler):
    """Cosine annealing schedule"""
    def __init__(self, start_value: float, end_value: float, num_steps: int):
        self.start = start_value
        self.end = end_value
        self.num_steps = num_steps
    
    def __call__(self, step: int) -> float:
        progress = min(step / self.num_steps, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.end + (self.start - self.end) * cosine_decay