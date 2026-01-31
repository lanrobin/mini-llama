

from dataclasses import fields
from utils import SamplingParams, Logger, Config
from transformers import AutoTokenizer

class LLMEngine:
    def __init__(self, model_path: str, **kwargs):
        self.logger = Logger()
        config_fiels = {f.name for f in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fiels}
        self.config = Config(model_path=model_path, **config_kwargs)
        self.model_path = model_path
        self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def load_model(self):
        # Placeholder for model loading logic
        self.logger.info(f"Loading model: {self.model_path}")

    def generate(self, prompt: list[list[int]], sampling_params: list[SamplingParams]) -> list[str]:

       assert len(prompt) == len(sampling_params), f"Number of prompts:{len(prompt)} must match number of sampling parameter sets:{len(sampling_params)}"

       return [ f"Generated text for prompt:{i} has {len(p)} tokens." for i, p in enumerate(prompt) ]