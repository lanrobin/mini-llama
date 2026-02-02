
from utils.config import Config
from .sequence import Sequence

class Scheduler:
    def __init__(self, config:Config):
        self.config = config

    def add_task(self, seq:Sequence):
        pass

    def run(self):
        pass