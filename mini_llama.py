import os
from transformers import AutoTokenizer
from utils import Logger
from llm import LLM
from utils.sampling_params import SamplingParams

def main():
    logger = Logger()
    model_path = os.path.expanduser('~/huggingface/Llama-3.2-3B-Instruct')
    logger.info(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    llm = LLM(model_path=model_path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.7, max_tokens=1024, ignore_eos=False)

    prompt_texts = ["can you help to sum up from 1 to 10?", 
               "list all the prime numbers bweteen 1 and 100?",
               "can you write the first 20 digits of pi?"
               ]
    
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize = False, add_generation_prompt=True)
            for text in prompt_texts
    ]

    outputs = llm.generate_texts(prompts, sampling_params)

    for prompt, output in zip(prompt_texts, outputs):
        logger.info(f"Prompt: {prompt!r}")
        logger.info(f"Output: {output!r}")

if __name__ == "__main__":
    main()