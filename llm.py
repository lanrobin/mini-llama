from engine import LLMEngine
from utils import SamplingParams

class LLM(LLMEngine):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)


    def generate_texts(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        sampling_params = [sampling_params] * len(prompts)
        
        tokenized_prompts = [self.tokenizer.encode(prompt) for prompt in prompts]
        return self.generate_tokens(tokenized_prompts, sampling_params)


    def generate_tokens(self, prompts: list[list[int]], sampling_params: list[SamplingParams]) -> list[str]:
        return super().generate(prompts, sampling_params)