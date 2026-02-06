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