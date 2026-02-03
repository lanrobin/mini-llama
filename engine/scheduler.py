
from utils.config import Config
from .sequence import Sequence

class Scheduler:
    def __init__(self, config:Config):
        self.config = config

    def add_task(self, seq:Sequence):
        pass

    def schedule(self) -> tuple[list[Sequence], bool]:
        raise NotImplementedError("This is a placeholder method. Implement the schedule method.")
    
    def is_finished(self) -> bool:
        raise NotImplementedError("This is a placeholder method. Implement the is_finished method.")
    
    def preempt(self, seq:Sequence):
        raise NotImplementedError("This is a placeholder method. Implement the preempt method.")
    
    def post_process(self, token_ids: list[int]):
        raise NotImplementedError("This is a placeholder method. Implement the post_process method.")