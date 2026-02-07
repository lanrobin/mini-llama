
import torch

from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig

from layers import ParallelLMHead, VocabParallelEmbeddingHead, RMSNorm, QKVParallelLinear, RowParallelLinear, get_rope, Attention, SiluAndMultiply, MergedColumnParallelLinear
from utils.log import Logger

class Llama32Attention(nn.Module):
    def __init__(self, hidden_size: int,
                num_attention_heads: int,
                num_kv_heads: int,
                max_position: int,
                head_dim: int | None = None,
                rms_norm_eps: float = 1e-6,
                qkv_bias: bool = False,
                rope_theta: float = 10000.0,
                rope_scaling:dict| None = None):
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_attention_heads
        assert num_attention_heads % tp_size == 0, "num_attention_heads must be divisible by tp_size"
        self.num_heads_in_current_tp = num_attention_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert num_kv_heads % tp_size == 0, "num_kv_heads must be divisible by tp_size"
        self.num_kv_heads_in_current_tp = num_kv_heads // tp_size
        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.q_size = self.num_heads_in_current_tp * self.head_dim
        self.kv_size = self.num_kv_heads_in_current_tp * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias)
        
        self.o_proj = RowParallelLinear( self.total_num_heads * self.head_dim, hidden_size, bias=False)
        
        self.rotary_embedding = get_rope(self.head_dim, self.head_dim, max_position = max_position, base=rope_theta, rope_scaling=tuple(rope_scaling.items()) if rope_scaling is not None else None)
        
        self.attn = Attention(num_heads=self.num_heads_in_current_tp, head_dim=self.head_dim, scale=self.scaling, num_kv_heads=self.num_kv_heads_in_current_tp)
        
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)


    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        qv = q.view(-1, self.num_heads_in_current_tp, self.head_dim)
        kv = k.view(-1, self.num_kv_heads_in_current_tp, self.head_dim)
        vv = v.view(-1, self.num_kv_heads_in_current_tp, self.head_dim)
        if self.qkv_bias:
            qv = self.q_norm(qv)
            kv = self.k_norm(kv)
        qr,kr = self.rotary_embedding(qv, kv, positions)
        o = self.attn(qr, kr, vv)
        #o = self.o_proj(o.contiguous().view(-1, self.total_num_heads * self.head_dim))
        output = self.o_proj(o.flatten(1, -1))
        return output

class Llama32MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        
        assert hidden_act == "silu", "Currently only 'silu' activation is supported."
        self.activation_fn = SiluAndMultiply()
        

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        ax = self.activation_fn(gate_up)
        axd = self.down_proj(ax)
        return axd

class Llama32DecoderLayer(nn.Module):
    '''
    Llama32DecoderLayer
    Its weight map looks like this:
    "weight_map": {
                    # "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.post_attention_layernorm.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00002.safetensors",
                    # "model.norm.weight": "model-00002-of-00002.safetensors"
                    }
    '''
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.logger = Logger()
        assert config is not None, "LlamaConfig must be provided."
        self.config = config
        # Define layer components here (e.g., attention, feed-forward, etc.)
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)


        self.mlp = Llama32MLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act)

        self.self_attn = Llama32Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=config.attention_bias,
            rope_theta=config.rope_parameters.get("rope_theta", 500000.0),
            rope_scaling=config.rope_scaling
        )

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        # More components would be defined here

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            
        hidden_states = self.self_attn(hidden_states, positions)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Llama32Model(nn.Module):
    '''
    Llama32Model
    Its weight map looks like this:
    "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.post_attention_layernorm.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00002.safetensors",
                    "model.norm.weight": "model-00002-of-00002.safetensors"
                    }
    '''
    def __init__(self, config: LlamaConfig ):
        super().__init__()
        self.logger = Logger()
        assert config is not None, "LlamaConfig must be provided."
        self.config = config
        self.embed_tokens = VocabParallelEmbeddingHead(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size)
        
        self.layers = nn.ModuleList([Llama32DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.logger.info(f"Llama32Model initialized with {config.num_hidden_layers} layers. rms_norm_eps={config.rms_norm_eps}")


    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    
    def __init__(self, config: LlamaConfig | None = None):
        super().__init__()
        self.logger = Logger()
        assert config is not None, "LlamaConfig must be provided."
        self.config = config
        self.model = Llama32Model(config)
        self.lm_head = ParallelLMHead(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            bias=False
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data.copy_(self.model.embed_tokens.weight.data)


    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)