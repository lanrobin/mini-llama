import torch
from transformers import pipeline

# model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_path = "/home/lan/huggingface/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_path,
    dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
for output in outputs:
    print(output["generated_text"][-1])
