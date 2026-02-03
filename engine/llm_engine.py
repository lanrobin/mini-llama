

import atexit
from dataclasses import fields

from typer import prompt
from engine.model_runner import MasterModelRunner, SlaveModelRunner
from engine.scheduler import Scheduler
from engine.sequence import Sequence
from utils import SamplingParams, Logger, Config, sampling_params, sampling_params
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


    def generate(self, prompt: list[list[int]], sampling_params: list[SamplingParams]) -> list[dict]:

        assert len(prompt) == len(sampling_params), f"Number of prompts:{len(prompt)} must match number of sampling parameter sets:{len(sampling_params)}"

        for p, sp in zip(prompt, sampling_params):
           self.add_request(p, sp)

        outputs = {}
        while not self.is_finished():
           step_outputs, num_tokens = self.step()
           self.logger.info(f"Step produced {len(step_outputs)} finished sequences, processed {num_tokens} tokens.")
           for seq_id, completion_tokens in step_outputs:
               outputs[seq_id] = completion_tokens

        self.logger.info(f"Generation completed for all sequences. Total sequences: {len(outputs)}")
        sorted_outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        decoded_texts = [{"text": self.tokenizer.decode(tokens, skip_special_tokens=True), "token_ids": tokens} for tokens in sorted_outputs]
        return decoded_texts

    def exit(self):
        self.master_runner.call("exit")
        del self.master_runner
        for p in self.ps:
            p.join()
        self.logger.info("LLMEngine exited cleanly.")

    def add_request(self, prompt: list[int], sampling_params: SamplingParams):
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add_task(seq)

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.master_runner.run(seqs, is_prefill)
        self.scheduler.post_process(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_tokens) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens
    
    def is_finished(self) -> bool:
        return self.scheduler.is_finished()