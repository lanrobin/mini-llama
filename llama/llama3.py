
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
    def __init__(self, config: LlamaConfig, layer_id: int):
        super().__init__()
        self.logger = Logger()
        assert config is not None, "LlamaConfig must be provided."
        self.config = config
        self.layer_id = layer_id
        self.logger.info(f">>>>>>>>>>>>>>>>======Llama32DecoderLayer {layer_id} with hidden_size={config.hidden_size}, intermediate_size={config.intermediate_size}, num_attention_heads={config.num_attention_heads}, num_key_value_heads={config.num_key_value_heads}, head_dim={config.head_dim}, rms_norm_eps={config.rms_norm_eps}")
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
        self.logger.info(f"<<<<<<<<<<<<<<<==========Llama32DecoderLayer {layer_id} initialized.")

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            
        hidden_states_1 = self.self_attn(hidden_states, positions)
        hidden_states_2, residual = self.post_attention_layernorm(hidden_states_1, residual)
        hidden_states_3 = self.mlp(hidden_states_2)
        
        #if self.layer_id == 0:
        #    self.logger.info(f"After first layer, hidden_states_1 shape: {hidden_states_1.shape}, hidden_states_2 shape: {hidden_states_2.shape}, hidden_states_3 shape: {hidden_states_3.shape}")  
        
        return hidden_states_3, residual


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
        
        self.layers = nn.ModuleList([Llama32DecoderLayer(config, layer_id=layer_id) for layer_id in range(config.num_hidden_layers)])

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
        '''
        两行代码实现了一个经典的深度学习优化技术，称为 权重绑定（Weight Tying）。就是将语言模型的输出层（lm_head）的权重与输入嵌入层（model.embed_tokens）的权重共享。这种技术有几个好处：
        1. 减少模型参数：通过共享权重，模型的总参数量减少了，这有助于降低内存占用和计算成本。
        2. 提高泛化能力：权重绑定可以帮助模型更好地泛化，因为它强制模型在输入和输出之间学习相似的表示。这对于语言模型来说尤其有用，因为输入和输出通常是相同的词汇表。
        3. 加速训练：共享权重可以加速训练过程，因为模型不需要学习两个独立的权重矩阵，而是只需要学习一个。这可以导致更快的收敛和更好的性能。
        总的来说，这两行代码通过权重绑定技术优化了模型的参数效率和训练效率，同时也有助于提高模型的泛化能力。

        如果 config 中的 tie_word_embeddings 设置为 False, 那么在模型初始化时，lm_head 的权重将不会与 embed_tokens 的权重共享。这意味着 lm_head 将拥有自己独立的权重矩阵，而不是使用 embed_tokens 的权重。
        这可能会增加模型的参数量，但也允许 lm_head 学习与 embed_tokens 不同的表示，这在某些情况下可能是有益的。
        '''
        if config.tie_word_embeddings:
            self.lm_head.weight.data= self.model.embed_tokens.weight.data


    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)