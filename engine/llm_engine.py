

import atexit
from dataclasses import fields
from engine.model_runner import MasterModelRunner, SlaveModelRunner
from engine.scheduler import Scheduler
from utils import SamplingParams, Logger, Config
from transformers import AutoTokenizer
import torch.multiprocessing as mp

class LLMEngine:
    def __init__(self, model_path: str, **kwargs):
        self.logger = Logger()
        config_fiels = {f.name for f in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fiels}
        config = Config(model_path=model_path, **config_kwargs)
        self.ps = []
        self.events = []

        # Initialize parallel model runners if needed
        if config.tensor_parallel_size > 1:
            ctx = mp.get_context("spawn")
            for rank in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                p = ctx.Process(target=SlaveModelRunner, args=(config, rank, event))
                p.start()
                self.ps.append(p)
                self.events.append(event)

        self.master_runner = MasterModelRunner(config, rank=0, events=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        config.eos_token_id = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)


    def generate(self, prompt: list[list[int]], sampling_params: list[SamplingParams]) -> list[str]:

       assert len(prompt) == len(sampling_params), f"Number of prompts:{len(prompt)} must match number of sampling parameter sets:{len(sampling_params)}"

       return [ f"Generated text for prompt:{i} has {len(p)} tokens." for i, p in enumerate(prompt) ]
    
    def exit(self):
        self.master_runner.call("exit")
        del self.master_runner
        for p in self.ps:
            p.join()
        self.logger.info("LLMEngine exited cleanly.")
