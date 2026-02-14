

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

        '''
         If tensor parallelism is enabled, we spawn multiple processes for the slave model runners. 
         Each slave process will run a portion of the model and communicate with the master process. 
         The master process will coordinate the generation across all processes and aggregate the results.
         This allows us to leverage multiple GPUs for larger models that cannot fit into a single GPU's memory,
         and also to speed up generation by parallelizing the computation across multiple devices.
         
         The code path are:
         1. p.start() -> p.target(*args) -> SlaveModelRunner.__init__() -> SlaveModelRunner.loop(), loop is infinite loop waiting for events from master process, when event is set, it runs the model and then waits for next event.
         2. When the run command is exit, the loop will break and the process will terminate.
         
        '''
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
        '''
        In LLAMA models, there are multiple end-of-sequence (EOS):
        "eos_token_id":
                    [
                        128001,
                        128008,
                        128009
                    ]
        the tokens are: ['<|end_of_text|>','<|eom_id|>','<|eot_id|>']
        so we need to use tokenizer.convert_tokens_to_ids(['<|end_of_text|>','<|eom_id|>','<|eot_id|>']) to get all of them instead just tokenizer.eos_token_id which is 128009.
        This way we can properly handle all end-of-sequence tokens during generation and ensure that the model stops generating when any of the EOS tokens are produced.
        '''
        # config.eos_token_id = self.tokenizer.eos_token_id
        config.eos_token_ids.update(self.get_all_eos_token_ids())
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)


    def generate(self, prompt: list[list[int]], sampling_params: list[SamplingParams]) -> list[dict]:
        '''
        It will generate text based on the input prompt and sampling parameters. 
        As this is just a demo, we aggressively try to finish all the task and then exit.
        In each step (while loop):
           1. call scheduler.schedule() to get the sequences that are ready for processing, and whether it's in prefill stage or decode stage.
           2. call master_runner.run() to run the model and get the generated token ids for each sequence.
           3. call scheduler.post_process() to update the sequence states based on the generated token ids, and check if there are finished sequences;
           if YES, remove them from the running queue and free their blocks and set the sequence as finished.
           4. Put the finished sequences' generated token ids into the outputs dict, and return the decoded texts after all sequences are finished.
        '''
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
        '''
        Each time, the number of token_ids generated from self.master_runner.run() MUST equal the number of sequences returned by self.scheduler.schedule().
        For each sequence, we append the newly generated token_id to the sequence, and then check if the sequence is EOS;
        if YES, we set the sequence state to finished, free its blocks and remove it from running sequences.
        if NO, we keep the sequence in running state and wait for next step to generate the next token.
        '''
        seqs, is_prefill = self.scheduler.schedule()
        # Get next token ids for each sequence from the model runner
        token_ids = self.master_runner.run(seqs, is_prefill)
        
        # Add the newly generated token ids to the sequences and check if they are finished.
        self.scheduler.post_process(seqs, token_ids)
        
        # return the finished sequences' generated token ids and the number of tokens processed in this step for logging.
        outputs = [(seq.seq_id, seq.completion_tokens) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens
    
    def is_finished(self) -> bool:
        return self.scheduler.is_finished()
    
    def get_all_eos_token_ids(self) -> list[int]:
        '''
        Usually, you need to check the model folder's config.json for the "eos_token_id" field, which may contain multiple token ids for end-of-sequence.
        For example, in LLAMA models, the "eos_token_id" field contains [128001, 128008, 128009], 
        which correspond to the tokens ['<|end_of_text|>','<|eom_id|>','<|eot_id|>'].
        Therefore, to get all the EOS token ids, you can use tokenizer.convert_tokens_to_ids(['<|end_of_text|>','<|eom_id|>','<|eot_id|>']) to retrieve their corresponding token ids. 
        This way, you can ensure that your generation process correctly identifies all possible end-of-sequence tokens and stops generating when any of them is produced.
        '''
        return self.tokenizer.convert_tokens_to_ids(['<|end_of_text|>','<|eom_id|>','<|eot_id|>'])