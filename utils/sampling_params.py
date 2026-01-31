from dataclasses import dataclass

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 512
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature < 0.0:
            raise ValueError("Temperature must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be a positive integer")