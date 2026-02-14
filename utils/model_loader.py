import os
from safetensors import safe_open
import torch
from torch import nn
from glob import glob

def default_weight_loader(param:nn.Parameter, weight:torch.Tensor):
    if param.shape != weight.shape:
        raise ValueError(f"Shape mismatch: expected {param.shape}, got {weight.shape}")
    param.data.copy_(weight)
    
    
def load_weights_from_safetensors(model:nn.Module, model_path:str, default_weight_loader=default_weight_loader):
    '''
    This function loads the model weights from the safetensors files in the specified model_path.
    It is sample, except for the packed modules, which need to be handled specially according to the packed_modules_mapping.
    Usually, we just load the weights to its corresponding model parameter, but for the packed modules,
    we need to load the weights to the corresponding sub-parameter according to the mapping.
    
    For example, for the qkv_proj module in the attention layer, we have three weight tensors in the safetensors file: q_proj, k_proj and v_proj.
    k_proj.shape = [num_key_value_heads * head_dim, hidden_size] =[1024, 3072]
    q_proj.shape = [hidden_size, hidden_size] = [3072, 3072]
    v_proj.shape = [num_key_value_heads * head_dim, hidden_size] =[1024, 3072]
    We packed them into a single weight tensor qkv_proj with shape =[3072 + 1024 + 1024]= [5120, 3072]
    
    Similarly, for the gate_up_proj = [16384, 3072] module in the MLP layer,
    we have two weight tensors in the safetensors file: gate_proj.shape = [8192, 3072] and up_proj.shape = [8192, 3072].
    '''
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    '''
    packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    }
    '''
    for file in glob(os.path.join(model_path, "*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as f:
            for weigh_name in f.keys():
                for mapping_key, (weigh_variable, shard_id) in packed_modules_mapping.items():
                    if mapping_key in weigh_name:
                        param_name = weigh_name.replace(mapping_key, weigh_variable)
                        model_param = model.get_parameter(param_name)
                        weight_loader = getattr(model_param, "weight_loader")
                        assert weight_loader is not None, f"{model_param} must have a weight_loader method"
                        weight_loader(model_param, f.get_tensor(weigh_name), shard_id)
                        break
                else:
                    model_param = model.get_parameter(weigh_name)
                    weight_loader = getattr(model_param, "weight_loader", default_weight_loader)
                    weight_loader(model_param, f.get_tensor(weigh_name))