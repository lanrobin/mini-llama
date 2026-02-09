from dataclasses import dataclass
import torch

from utils.log import Logger
from utils.singleton import SingletonMeta

@dataclass
class Context:
    is_prefill : bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None



class ContextManager(metaclass=SingletonMeta):
    def __init__(self):
        self.contexts = dict()
        self.logger = Logger()

    def set_context(self, context_name: str, context: Context):
        self.contexts[context_name] = context
        #self.logger.info(f"Context '{context_name}' set.")

    def get_context(self, context_name: str) -> Context | None:
        return self.contexts.get(context_name, None)
    
    def clear_context(self, context_name: str):
        if context_name in self.contexts:
            del self.contexts[context_name]
            #self.logger.info(f"Context '{context_name}' cleared.")

    def set_default_context(self, context: Context):
        self.set_context("default", context)

    def get_default_context(self) -> Context | None:
        return self.get_context("default")
    
    def clear_default_context(self):
        self.clear_context("default")